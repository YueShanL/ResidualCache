from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from .contracts import BlockRange


KV_SCHEMA_VERSION = 1


def _safe_file_id(block_id: str) -> str:
    digest = hashlib.sha256(block_id.encode("utf-8")).hexdigest()[:20]
    return f"block-{digest}.pt"


@dataclass(frozen=True)
class KVBlock:
    block: BlockRange
    sequence_id: str
    token_ids: tuple[int, ...]
    logical_positions: tuple[int, ...]
    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    residual_summary: torch.Tensor
    model_fingerprint: dict[str, Any]
    metadata: dict[str, Any]

    def validate(self) -> "KVBlock":
        expected = self.block.length
        if len(self.token_ids) != expected or len(self.logical_positions) != expected:
            raise ValueError("block token and logical-position counts must match its range")
        if self.logical_positions != tuple(range(self.block.start_position, self.block.end_position)):
            raise ValueError("logical positions must exactly match the block range")
        if not self.layer_kv:
            raise ValueError("KV block must contain at least one physical cache layer")
        for layer_index, (key, value) in enumerate(self.layer_kv):
            if key.ndim != 4 or value.ndim != 4:
                raise ValueError(f"KV layer {layer_index} must have shape [batch, heads, tokens, dim]")
            if key.shape != value.shape:
                raise ValueError(f"KV layer {layer_index} has mismatched key/value shapes")
            if key.shape[0] != 1 or key.shape[2] != expected:
                raise ValueError(f"KV layer {layer_index} does not contain exactly the block tokens")
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError(f"KV layer {layer_index} contains non-finite values")
        if self.residual_summary.ndim != 1 or not torch.isfinite(self.residual_summary).all():
            raise ValueError("residual_summary must be a finite vector")
        if not self.model_fingerprint:
            raise ValueError("model_fingerprint is required")
        return self

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": KV_SCHEMA_VERSION,
            "block": self.block.to_dict(),
            "sequence_id": self.sequence_id,
            "token_ids": list(self.token_ids),
            "logical_positions": list(self.logical_positions),
            "layer_kv": tuple(
                (key.detach().cpu(), value.detach().cpu()) for key, value in self.layer_kv
            ),
            "residual_summary": self.residual_summary.detach().float().cpu(),
            "model_fingerprint": dict(self.model_fingerprint),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "KVBlock":
        if int(payload.get("schema_version", 0)) != KV_SCHEMA_VERSION:
            raise ValueError("unsupported KV block schema")
        return cls(
            block=BlockRange.from_dict(payload["block"]),
            sequence_id=str(payload["sequence_id"]),
            token_ids=tuple(int(value) for value in payload["token_ids"]),
            logical_positions=tuple(int(value) for value in payload["logical_positions"]),
            layer_kv=tuple(
                (torch.as_tensor(key), torch.as_tensor(value)) for key, value in payload["layer_kv"]
            ),
            residual_summary=torch.as_tensor(payload["residual_summary"]).float(),
            model_fingerprint=dict(payload["model_fingerprint"]),
            metadata=dict(payload.get("metadata", {})),
        ).validate()


class KVBlockStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.blocks_dir = self.root / "blocks"
        self.manifest_path = self.root / "manifest.json"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            if int(self.manifest.get("schema_version", 0)) != KV_SCHEMA_VERSION:
                raise ValueError("unsupported KV store schema")
        else:
            self.manifest = {"schema_version": KV_SCHEMA_VERSION, "blocks": {}}

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        for attempt in range(6):
            try:
                temporary.replace(self.manifest_path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                # Windows virus scanners and indexers can transiently hold the
                # just-flushed file. Keep the operation atomic and retry with
                # bounded backoff instead of falling back to an in-place write.
                time.sleep(0.05 * (2**attempt))

    def contains(self, block_id: str) -> bool:
        return block_id in self.manifest["blocks"]

    def save(self, block: KVBlock) -> Path:
        block.validate()
        existing = self.manifest["blocks"].get(block.block.block_id)
        file_name = existing["file"] if existing else _safe_file_id(block.block.block_id)
        path = self.blocks_dir / file_name
        torch.save(block.to_payload(), path)
        self.manifest["blocks"][block.block.block_id] = {
            "file": file_name,
            "sequence_id": block.sequence_id,
            "start_position": block.block.start_position,
            "end_position": block.block.end_position,
            "physical_layer_count": len(block.layer_kv),
            "model_fingerprint": block.model_fingerprint,
        }
        self._write_manifest()
        return path

    def load(self, block_id: str) -> KVBlock:
        try:
            entry = self.manifest["blocks"][block_id]
        except KeyError as error:
            raise KeyError(f"KV block is not present: {block_id}") from error
        try:
            payload = torch.load(self.blocks_dir / entry["file"], map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.blocks_dir / entry["file"], map_location="cpu")
        block = KVBlock.from_payload(payload)
        if block.block.block_id != block_id:
            raise RuntimeError("KV manifest and payload block ids disagree")
        return block

    def load_many(self, block_ids: Iterable[str]) -> list[KVBlock]:
        blocks = [self.load(block_id) for block_id in block_ids]
        blocks.sort(key=lambda item: item.block.start_position)
        for previous, current in zip(blocks, blocks[1:]):
            if previous.sequence_id != current.sequence_id:
                raise ValueError("cannot merge KV blocks from different sequences")
            if previous.block.end_position > current.block.start_position:
                raise ValueError("selected KV blocks overlap")
        if blocks:
            fingerprints = {json.dumps(block.model_fingerprint, sort_keys=True) for block in blocks}
            if len(fingerprints) != 1:
                raise ValueError("selected KV blocks have different model fingerprints")
            layer_counts = {len(block.layer_kv) for block in blocks}
            if len(layer_counts) != 1:
                raise ValueError("selected KV blocks have different physical layer counts")
        return blocks


def merge_layer_kv(
    blocks: Iterable[KVBlock],
    *,
    device: torch.device | str,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    block_list = list(blocks)
    if not block_list:
        return ()
    layer_count = len(block_list[0].layer_kv)
    merged: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index in range(layer_count):
        keys = [block.layer_kv[layer_index][0].to(device) for block in block_list]
        values = [block.layer_kv[layer_index][1].to(device) for block in block_list]
        merged.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(merged)
