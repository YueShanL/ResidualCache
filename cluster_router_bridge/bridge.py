from __future__ import annotations

from typing import Any, Hashable, Iterable, Mapping, Protocol, Sequence

from .contracts import (
    BlockInput,
    BlockWriteResult,
    ClusterMembership,
    ClusterSelection,
    MemoryRecordInput,
    MemoryWritePlacement,
    RecordRef,
)


class MemoryWriter(Protocol):
    """Boundary used by the independent learned-router/memory bridge."""

    def write(
        self,
        record: MemoryRecordInput,
        *,
        router_key: Sequence[float],
        router_block_id: Hashable,
        router_block_size: int,
    ) -> MemoryWritePlacement: ...

    def memberships(self) -> tuple[ClusterMembership, ...]: ...

    def select_router_clusters(
        self,
        query: Sequence[float],
        *,
        top_n: int = 4,
        layers: Iterable[int] | None = None,
        head_or_kv_group: Hashable = 0,
    ) -> dict[int, tuple[ClusterSelection, ...]]: ...


class HierarchicalLayerMemoryAdapter:
    """Map external decoder layers to independent hierarchical memories.

    The adapter transports a completed block's opaque router metadata into the
    memory public API. Distribution ownership remains entirely inside memory:
    the adapter caches no router keys, memberships, or vMF state.
    """

    def __init__(self, memories: Mapping[int, Any]) -> None:
        if not memories:
            raise ValueError("at least one layer memory is required")
        self.memories = {int(layer): memory for layer, memory in memories.items()}
        if min(self.memories) < 0:
            raise ValueError("layer indices must be non-negative")

    def write(
        self,
        record: MemoryRecordInput,
        *,
        router_key: Sequence[float],
        router_block_id: Hashable,
        router_block_size: int,
    ) -> MemoryWritePlacement:
        try:
            memory = self.memories[record.layer]
        except KeyError as error:
            raise KeyError(f"no memory is configured for layer {record.layer}") from error
        source = (
            record.source_token_or_span
            if record.source_token_or_span is not None
            else record.logical_positions
        )
        decision = memory.write(
            record.memory_index,
            record.original_key,
            record.original_value,
            layer=0,
            head_or_kv_group=record.head_or_kv_group,
            source_token_or_span=source,
            time=record.time,
            router_key=router_key,
            router_block_id=router_block_id,
            router_block_size=router_block_size,
            **dict(record.write_kwargs),
        )
        return MemoryWritePlacement(
            record=RecordRef(record.layer, decision.record_id),
            cluster_id=decision.slot_id,
        )

    def memberships(self) -> tuple[ClusterMembership, ...]:
        memberships: list[ClusterMembership] = []
        for layer, memory in sorted(self.memories.items()):
            for leaf in memory.leaves.values():
                memberships.append(
                    ClusterMembership(
                        layer=layer,
                        cluster_id=leaf.id,
                        record_ids=tuple(leaf.record_ids),
                    )
                )
        return tuple(memberships)

    def select_router_clusters(
        self,
        query: Sequence[float],
        *,
        top_n: int = 4,
        layers: Iterable[int] | None = None,
        head_or_kv_group: Hashable = 0,
    ) -> dict[int, tuple[ClusterSelection, ...]]:
        selected_layers = (
            set(self.memories) if layers is None else {int(layer) for layer in layers}
        )
        unknown = selected_layers.difference(self.memories)
        if unknown:
            raise KeyError(f"no memory is configured for layers {sorted(unknown)}")
        selected: dict[int, tuple[ClusterSelection, ...]] = {}
        for layer in sorted(selected_layers):
            rows = self.memories[layer].retrieve_router_clusters(
                query,
                layer=0,
                head_or_kv_group=head_or_kv_group,
                top_n=top_n,
            )
            selected[layer] = tuple(
                ClusterSelection(
                    layer=layer,
                    cluster_id=row.slot_id,
                    probability=row.probability,
                    log_score=row.log_score,
                    record_count=row.router_record_count,
                    total_weight=row.router_mass,
                    record_refs=tuple(
                        RecordRef(layer, record_id) for record_id in row.record_ids
                    ),
                )
                for row in rows
            )
        return selected

    @staticmethod
    def records_for_selections(
        selections: Mapping[int, Sequence[ClusterSelection]],
    ) -> dict[int, tuple[RecordRef, ...]]:
        result: dict[int, tuple[RecordRef, ...]] = {}
        for layer, layer_selections in selections.items():
            records: list[RecordRef] = []
            seen: set[RecordRef] = set()
            for selection in layer_selections:
                if selection.layer != layer:
                    raise ValueError("selection is stored under the wrong layer")
                for record in selection.record_refs:
                    if record.layer != layer:
                        raise ValueError("selected record is stored under the wrong layer")
                    if record not in seen:
                        records.append(record)
                        seen.add(record)
            result[layer] = tuple(records)
        return result


class BlockMemoryIngestor:
    """Write blocks while memory owns router metadata and distributions."""

    def __init__(
        self,
        writer: MemoryWriter,
        *,
        payload_store: Any | None = None,
    ) -> None:
        self.writer = writer
        self.payload_store = payload_store

    def ingest(self, block: BlockInput) -> BlockWriteResult:
        placements: list[MemoryWritePlacement] = []
        expected = dict(block.expected_records_by_layer or {})
        for record in block.records:
            placement = self.writer.write(
                record,
                router_key=block.router_key,
                router_block_id=block.block_id,
                router_block_size=expected[record.layer],
            )
            placements.append(placement)
            if self.payload_store is not None and record.kv_payload is not None:
                self.payload_store.register(placement.record, record.kv_payload)
        return BlockWriteResult(block.block_id, tuple(placements))

    def select(
        self,
        query: Sequence[float],
        *,
        top_n: int = 4,
        layers: Iterable[int] | None = None,
        head_or_kv_group: Hashable = 0,
    ) -> dict[int, tuple[ClusterSelection, ...]]:
        return self.writer.select_router_clusters(
            query,
            top_n=top_n,
            layers=layers,
            head_or_kv_group=head_or_kv_group,
        )

    def records_for_selections(
        self, selections: Mapping[int, Sequence[ClusterSelection]]
    ) -> dict[int, tuple[RecordRef, ...]]:
        resolver = getattr(self.writer, "records_for_selections", None)
        if resolver is None:
            raise TypeError("memory writer cannot resolve cluster selections")
        return resolver(selections)

    def synchronize(self) -> None:
        """Drop tensor payloads for records no longer retained by memory."""

        if self.payload_store is None:
            return
        memberships = self.writer.memberships()
        active = {
            RecordRef(membership.layer, record_id)
            for membership in memberships
            for record_id in membership.record_ids
        }
        self.payload_store.retain(active)
