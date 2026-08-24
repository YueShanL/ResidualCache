from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Hashable, Mapping, Sequence


Vector = tuple[float, ...]


def finite_vector(values: Sequence[float], *, name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class RecordRef:
    """Stable identity of one layer-local memory record."""

    layer: int
    record_id: Hashable

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        hash(self.record_id)


@dataclass(frozen=True)
class KVRecordPayload:
    """Physical tensor K/V payload kept by the independent bridge.

    Tensor validation is intentionally deferred to :class:`LayerKVCacheBuilder`
    so importing the contracts does not import torch.
    """

    key: Any
    value: Any
    logical_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        positions = tuple(int(position) for position in self.logical_positions)
        if not positions:
            raise ValueError("logical_positions must be non-empty")
        if min(positions) < 0 or any(
            right <= left for left, right in zip(positions, positions[1:])
        ):
            raise ValueError("logical_positions must be strictly increasing")
        object.__setattr__(self, "logical_positions", positions)


@dataclass(frozen=True)
class MemoryRecordInput:
    """One record written through the memory's existing public write path.

    This contract deliberately contains no router key because that key belongs
    to the completed block, not an individual adapter record. The bridge passes
    it directly to memory's write API, where it becomes record metadata without
    participating in native K/V classification.
    """

    layer: int
    logical_positions: tuple[int, ...]
    memory_index: Vector
    original_key: Vector
    original_value: Vector
    kv_payload: KVRecordPayload | None = None
    head_or_kv_group: Hashable = 0
    time: float | None = None
    source_token_or_span: Hashable | None = None
    write_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        positions = tuple(int(position) for position in self.logical_positions)
        if not positions or min(positions) < 0:
            raise ValueError("logical_positions must be non-empty and non-negative")
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError("logical_positions must be strictly increasing")
        object.__setattr__(self, "logical_positions", positions)
        object.__setattr__(
            self, "memory_index", finite_vector(self.memory_index, name="memory_index")
        )
        object.__setattr__(
            self, "original_key", finite_vector(self.original_key, name="original_key")
        )
        object.__setattr__(
            self,
            "original_value",
            finite_vector(self.original_value, name="original_value"),
        )
        if self.time is not None and not math.isfinite(float(self.time)):
            raise ValueError("time must be finite")
        forbidden = {
            "layer",
            "head_or_kv_group",
            "time",
            "source_token_or_span",
            "router_key",
            "router_block_id",
            "router_block_size",
        }
        overlap = forbidden.intersection(self.write_kwargs)
        if overlap:
            raise ValueError(f"write_kwargs overrides reserved fields: {sorted(overlap)}")
        if self.kv_payload is not None:
            if self.kv_payload.logical_positions != positions:
                raise ValueError("K/V payload positions must match the record positions")


@dataclass(frozen=True)
class BlockInput:
    """Completed mechanical block ready for router binding."""

    block_id: str
    start_position: int
    end_position: int
    router_key: Vector
    records: tuple[MemoryRecordInput, ...]
    expected_records_by_layer: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if self.start_position < 0 or self.end_position <= self.start_position:
            raise ValueError("block positions must form a non-empty half-open range")
        key = finite_vector(self.router_key, name="router_key")
        if math.sqrt(math.fsum(value * value for value in key)) <= 0.0:
            raise ValueError("router_key must have non-zero norm")
        object.__setattr__(self, "router_key", key)
        records = tuple(self.records)
        if not records:
            raise ValueError("a block must contain at least one memory record")
        for record in records:
            if min(record.logical_positions) < self.start_position or max(
                record.logical_positions
            ) >= self.end_position:
                raise ValueError("record positions must lie inside the block")
        object.__setattr__(self, "records", records)

        actual: dict[int, int] = {}
        for record in records:
            actual[record.layer] = actual.get(record.layer, 0) + 1
        expected = (
            dict(actual)
            if self.expected_records_by_layer is None
            else {
                int(layer): int(count)
                for layer, count in self.expected_records_by_layer.items()
            }
        )
        if any(layer < 0 or count <= 0 for layer, count in expected.items()):
            raise ValueError("expected record counts must be positive")
        for layer, count in actual.items():
            if expected.get(layer, 0) < count:
                raise ValueError("expected record count cannot be smaller than actual writes")
        object.__setattr__(self, "expected_records_by_layer", expected)


@dataclass(frozen=True)
class MemoryWritePlacement:
    record: RecordRef
    cluster_id: Hashable

    def __post_init__(self) -> None:
        hash(self.cluster_id)


@dataclass(frozen=True)
class BlockWriteResult:
    block_id: str
    placements: tuple[MemoryWritePlacement, ...]


@dataclass(frozen=True)
class ClusterMembership:
    layer: int
    cluster_id: Hashable
    record_ids: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        hash(self.cluster_id)
        record_ids = tuple(self.record_ids)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("cluster membership contains duplicate record ids")
        object.__setattr__(self, "record_ids", record_ids)


@dataclass(frozen=True)
class ClusterSelection:
    layer: int
    cluster_id: Hashable
    probability: float
    log_score: float
    record_count: int
    total_weight: float
    record_refs: tuple[RecordRef, ...] = ()

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        hash(self.cluster_id)
        refs = tuple(self.record_refs)
        if any(ref.layer != self.layer for ref in refs):
            raise ValueError("selected records must belong to the selection layer")
        if len(set(refs)) != len(refs):
            raise ValueError("selection contains duplicate record refs")
        object.__setattr__(self, "record_refs", refs)


@dataclass(frozen=True)
class LayerKVView:
    """Packed variable-length historical K/V for one decoder layer."""

    layer: int
    key: Any
    value: Any
    logical_positions: tuple[int, ...]
    record_refs: tuple[RecordRef, ...]
    selected_cluster_ids: tuple[Hashable, ...]

    @property
    def token_count(self) -> int:
        return len(self.logical_positions)
