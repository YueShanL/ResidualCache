from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import BlockInput, KVRecordPayload, MemoryRecordInput, Vector


MemoryIndexer = Callable[[int, Any, Any], Sequence[float]]


def flattened_key_index(_layer: int, key_token: Any, _value_token: Any) -> Vector:
    """Use the record's own flattened K as its memory classification feature."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project runtime has torch.
        raise RuntimeError("K/V block splitting requires torch") from error
    if not isinstance(key_token, torch.Tensor):
        raise TypeError("key_token must be a torch tensor")
    return tuple(float(value) for value in key_token.detach().float().cpu().reshape(-1))


def mean_kv_index(_layer: int, key_token: Any, value_token: Any) -> Vector:
    """Mean the record's own K and V without consulting its router tag."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project runtime has torch.
        raise RuntimeError("K/V block splitting requires torch") from error
    if not isinstance(key_token, torch.Tensor) or not isinstance(value_token, torch.Tensor):
        raise TypeError("key_token and value_token must be torch tensors")
    if key_token.shape != value_token.shape:
        raise ValueError("mean K/V indexing requires matching K/V shapes")
    feature = (key_token.detach().float() + value_token.detach().float()) * 0.5
    return tuple(float(value) for value in feature.cpu().reshape(-1))


class TorchKVBlockInputBuilder:
    """Split an all-layer K/V block into layer-local token records.

    The supplied ``memory_indexer`` sees only the record's layer-local K/V.
    The learned router key is attached later by :class:`BlockMemoryIngestor`
    and cannot affect memory classification.
    """

    def __init__(
        self,
        memory_indexer: MemoryIndexer = flattened_key_index,
        *,
        time_from_logical_position: bool = False,
    ) -> None:
        self.memory_indexer = memory_indexer
        self.time_from_logical_position = bool(time_from_logical_position)

    @staticmethod
    def _layers(layer_kv: Mapping[int, tuple[Any, Any]] | Sequence[tuple[Any, Any]]):
        if isinstance(layer_kv, Mapping):
            return sorted((int(layer), pair) for layer, pair in layer_kv.items())
        return list(enumerate(layer_kv))

    def build(
        self,
        *,
        block_id: str,
        start_position: int,
        router_key: Sequence[float],
        layer_kv: Mapping[int, tuple[Any, Any]] | Sequence[tuple[Any, Any]],
    ) -> BlockInput:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - project runtime has torch.
            raise RuntimeError("K/V block splitting requires torch") from error

        layers = self._layers(layer_kv)
        if not layers:
            raise ValueError("layer_kv must contain at least one layer")
        records: list[MemoryRecordInput] = []
        expected: dict[int, int] = {}
        token_count: int | None = None
        for layer, (key, value) in layers:
            if layer < 0:
                raise ValueError("layer indices must be non-negative")
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError("layer K/V payloads must be torch tensors")
            if key.ndim != 4 or value.shape != key.shape:
                raise ValueError("layer K/V must have shape [batch, heads, tokens, dim]")
            if key.shape[0] != 1:
                raise ValueError("block ingestion currently requires batch size 1")
            current_tokens = int(key.shape[2])
            if current_tokens <= 0:
                raise ValueError("layer K/V blocks must be non-empty")
            if token_count is None:
                token_count = current_tokens
            elif current_tokens != token_count:
                raise ValueError("all layers in one mechanical block must align in token count")
            expected[layer] = current_tokens
            for offset in range(current_tokens):
                position = start_position + offset
                key_token = key[:, :, offset : offset + 1, :]
                value_token = value[:, :, offset : offset + 1, :]
                memory_index = self.memory_indexer(layer, key_token, value_token)
                original_key = tuple(
                    float(item)
                    for item in key_token.detach().float().cpu().reshape(-1)
                )
                original_value = tuple(
                    float(item)
                    for item in value_token.detach().float().cpu().reshape(-1)
                )
                records.append(
                    MemoryRecordInput(
                        layer=layer,
                        logical_positions=(position,),
                        memory_index=tuple(memory_index),
                        original_key=original_key,
                        original_value=original_value,
                        kv_payload=KVRecordPayload(
                            key=key_token.detach(),
                            value=value_token.detach(),
                            logical_positions=(position,),
                        ),
                        time=float(position) if self.time_from_logical_position else None,
                        source_token_or_span=(position,),
                    )
                )
        assert token_count is not None
        return BlockInput(
            block_id=block_id,
            start_position=start_position,
            end_position=start_position + token_count,
            router_key=tuple(float(value) for value in router_key),
            records=tuple(records),
            expected_records_by_layer=expected,
        )
