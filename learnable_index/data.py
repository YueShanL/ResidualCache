from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .contracts import RetrievalSample, SCHEMA_VERSION


DATA_FILENAME = "samples.pt"
MANIFEST_FILENAME = "manifest.json"


class RetrievalDataset(Dataset[RetrievalSample]):
    def __init__(self, samples: Sequence[RetrievalSample]) -> None:
        if not samples:
            raise ValueError("retrieval dataset cannot be empty")
        self.samples = [sample.validate() for sample in samples]
        dimensions = {sample.residual_dim for sample in self.samples}
        if len(dimensions) != 1:
            raise ValueError("every sample must use the same residual dimension")

    @property
    def residual_dim(self) -> int:
        return self.samples[0].residual_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RetrievalSample:
        return self.samples[index]


@dataclass(frozen=True)
class RetrievalBatch:
    sample_ids: tuple[str, ...]
    query_summaries: torch.Tensor
    block_summaries: torch.Tensor
    candidate_mask: torch.Tensor
    absolute_teacher_block_mass: torch.Tensor
    total_teacher_historical_mass: torch.Tensor
    conditional_teacher_distribution: torch.Tensor
    per_future_teacher_block_mass: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "RetrievalBatch":
        return RetrievalBatch(
            sample_ids=self.sample_ids,
            query_summaries=self.query_summaries.to(device),
            block_summaries=self.block_summaries.to(device),
            candidate_mask=self.candidate_mask.to(device),
            absolute_teacher_block_mass=self.absolute_teacher_block_mass.to(device),
            total_teacher_historical_mass=self.total_teacher_historical_mass.to(device),
            conditional_teacher_distribution=self.conditional_teacher_distribution.to(device),
            per_future_teacher_block_mass=(
                None
                if self.per_future_teacher_block_mass is None
                else self.per_future_teacher_block_mass.to(device)
            ),
        )


def collate_retrieval_samples(samples: Sequence[RetrievalSample]) -> RetrievalBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    samples = [sample.validate() for sample in samples]
    residual_dims = {sample.residual_dim for sample in samples}
    if len(residual_dims) != 1:
        raise ValueError("batch samples must have the same residual dimension")
    residual_dim = samples[0].residual_dim
    max_blocks = max(len(sample.candidate_blocks) for sample in samples)
    batch_size = len(samples)

    blocks = torch.zeros(batch_size, max_blocks, residual_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_blocks, dtype=torch.bool)
    absolute = torch.zeros(batch_size, max_blocks, dtype=torch.float32)
    conditional = torch.zeros(batch_size, max_blocks, dtype=torch.float32)
    per_future_present = [sample.per_future_teacher_block_mass is not None for sample in samples]
    if any(per_future_present) and not all(per_future_present):
        raise ValueError("a batch cannot mix samples with and without per-future teacher mass")
    per_future = None
    if all(per_future_present):
        horizons = {sample.future_horizon_length for sample in samples}
        if len(horizons) != 1:
            raise ValueError("per-future batching requires a shared future horizon length")
        per_future = torch.zeros(
            batch_size,
            samples[0].future_horizon_length,
            max_blocks,
            dtype=torch.float32,
        )
    for row, sample in enumerate(samples):
        count = len(sample.candidate_blocks)
        blocks[row, :count] = sample.block_summaries.float()
        mask[row, :count] = True
        absolute[row, :count] = sample.absolute_teacher_block_mass.float()
        conditional[row, :count] = sample.conditional_teacher_distribution.float()
        if per_future is not None:
            per_future[row, :, :count] = sample.per_future_teacher_block_mass.float()

    return RetrievalBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        query_summaries=torch.stack([sample.query_summary.float() for sample in samples]),
        block_summaries=blocks,
        candidate_mask=mask,
        absolute_teacher_block_mass=absolute,
        total_teacher_historical_mass=torch.stack(
            [sample.total_teacher_historical_mass.float().reshape(()) for sample in samples]
        ),
        conditional_teacher_distribution=conditional,
        per_future_teacher_block_mass=per_future,
    )


def save_dataset(
    output_dir: Path | str,
    samples: Iterable[RetrievalSample],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    sample_list = [sample.validate() for sample in samples]
    dataset = RetrievalDataset(sample_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "samples": [sample.to_payload() for sample in sample_list],
        },
        output_dir / DATA_FILENAME,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(dataset),
        "residual_dim": dataset.residual_dim,
        "tensor_file": DATA_FILENAME,
        "metadata": metadata or {},
    }
    with (output_dir / MANIFEST_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_dir


def load_dataset(input_dir: Path | str) -> tuple[RetrievalDataset, dict[str, Any]]:
    input_dir = Path(input_dir)
    with (input_dir / MANIFEST_FILENAME).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema: {manifest['schema_version']}")
    try:
        payload = torch.load(input_dir / manifest["tensor_file"], map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only was added
        payload = torch.load(input_dir / manifest["tensor_file"], map_location="cpu")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("manifest and tensor payload schema versions do not match")
    dataset = RetrievalDataset([RetrievalSample.from_payload(row) for row in payload["samples"]])
    if len(dataset) != int(manifest["sample_count"]):
        raise ValueError("manifest sample count does not match tensor payload")
    if dataset.residual_dim != int(manifest["residual_dim"]):
        raise ValueError("manifest residual dimension does not match tensor payload")
    return dataset, manifest


def split_dataset(
    dataset: RetrievalDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[RetrievalDataset, RetrievalDataset | None]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0 or len(dataset) < 2:
        return dataset, None
    sequence_ids = sorted({sample.sequence_id for sample in dataset.samples})
    # Splitting retrieval points from one sequence across train/validation leaks
    # adjacent residuals and repeated block summaries. With one sequence, report
    # no validation split instead of presenting a contaminated metric.
    if len(sequence_ids) < 2:
        return dataset, None
    random.Random(seed).shuffle(sequence_ids)
    validation_group_count = max(1, round(len(sequence_ids) * validation_fraction))
    validation_group_count = min(validation_group_count, len(sequence_ids) - 1)
    validation_sequences = set(sequence_ids[:validation_group_count])
    train = [
        sample for sample in dataset.samples if sample.sequence_id not in validation_sequences
    ]
    validation = [
        sample for sample in dataset.samples if sample.sequence_id in validation_sequences
    ]
    return RetrievalDataset(train), RetrievalDataset(validation)
