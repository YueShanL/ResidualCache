"""Independent block-record GPU memory classified by learned router keys.

Each complete layer-local mechanical block is one record.  Its learned block
key is the only native clustering feature and is posterior-classified once;
original K/V remain an opaque replay payload.  This module shares only the
generic vMF/locality mechanism with ``gpu_local_cluster_memory`` and contains
no token-vs-block mode branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Hashable, Sequence

from .gpu_vmf_posterior import (
    CpuLocalityIndex,
    locality_codes,
    vmf_posterior_assignments,
    vmf_slot_parameters,
)


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - project runtime includes torch.
        raise RuntimeError("GPU block cluster memory requires torch") from error
    return torch, functional


@dataclass(frozen=True)
class GpuBlockClusterMemoryConfig:
    """Posterior, allocation, and cluster-local eviction settings."""

    memory_budget_bytes: int | None = None
    eviction_enabled: bool | None = None
    block_size: int = 64
    slot_capacity: int = 128
    initial_record_capacity: int = 64
    candidate_capacity: int = 8
    locality_bits: int = 8
    locality_probe_radius: int = 1
    alpha: float = 0.1
    tau_new: float = 0.5
    count_exponent: float = 0.5
    concentration_prior_mass: float = 1.0
    maximum_concentration: float = 1_000.0
    locality_seed: int = 13
    usage_ema_rate: float = 0.25
    eviction_usage_threshold: float = 1e-4
    eviction_min_recall_count: int = 1
    eviction_min_records_per_cluster: int = 1

    def __post_init__(self) -> None:
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when provided")
        for name in (
            "block_size",
            "slot_capacity",
            "initial_record_capacity",
            "candidate_capacity",
            "locality_bits",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.locality_bits > 20:
            raise ValueError("locality_bits above 20 would make CPU probing impractical")
        if self.locality_probe_radius not in {0, 1}:
            raise ValueError("locality_probe_radius currently supports only 0 or 1")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.tau_new <= 1.0:
            raise ValueError("tau_new must be in [0, 1]")
        if not 0.0 <= self.count_exponent < 1.0:
            raise ValueError("count_exponent must be in [0, 1)")
        if self.concentration_prior_mass < 0.0:
            raise ValueError("concentration_prior_mass cannot be negative")
        if self.maximum_concentration <= 0.0:
            raise ValueError("maximum_concentration must be positive")
        if not 0.0 < self.usage_ema_rate <= 1.0:
            raise ValueError("usage_ema_rate must be in (0, 1]")
        if self.eviction_usage_threshold < 0.0:
            raise ValueError("eviction_usage_threshold cannot be negative")
        if self.eviction_min_recall_count <= 0:
            raise ValueError("eviction_min_recall_count must be positive")
        if self.eviction_min_records_per_cluster < 0:
            raise ValueError("eviction_min_records_per_cluster cannot be negative")

    @property
    def usage_eviction_enabled(self) -> bool:
        if self.eviction_enabled is not None:
            return bool(self.eviction_enabled)
        return self.memory_budget_bytes is not None


@dataclass(frozen=True)
class GpuBlockClusterSummary:
    cluster_id: str
    slot_index: int
    record_ids: tuple[str, ...]
    block_ids: tuple[Hashable, ...]
    block_lengths: tuple[int, ...]
    logical_positions: tuple[int, ...]
    probability: float
    log_score: float
    total_weight: float


@dataclass(frozen=True)
class PackedBlockKV:
    """Packed complete-block K/V plus explicit block-to-token alignment."""

    keys: Any
    values: Any
    logical_positions: tuple[int, ...]
    record_ids: tuple[str, ...]
    block_ids: tuple[Hashable, ...]
    record_slices: tuple[tuple[str, int, int], ...]
    token_record_ids: tuple[str, ...]


class GpuBlockClusterMemory:
    """One layer's dynamically growing block-record memory."""

    def __init__(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        router_dim: int,
        device: Any,
        dtype: Any,
        config: GpuBlockClusterMemoryConfig | None = None,
    ) -> None:
        torch, functional = _require_torch()
        self.config = config or GpuBlockClusterMemoryConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.router_dim = int(router_dim)
        self.block_size = int(self.config.block_size)
        if min(self.kv_heads, self.head_dim, self.router_dim) <= 0:
            raise ValueError("K/V and router dimensions must be positive")
        if self.router_dim < 32:
            raise ValueError("vMF router dimension must be at least 32")

        self.slot_capacity = int(self.config.slot_capacity)
        self.record_capacity = int(self.config.initial_record_capacity)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.locality_seed))
        projection = torch.randn(
            self.config.locality_bits,
            self.router_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.locality_projection = functional.normalize(projection, dim=-1).to(
            self.device
        )
        self._locality = CpuLocalityIndex(
            bits=self.config.locality_bits,
            probe_radius=self.config.locality_probe_radius,
            candidate_capacity=self.config.candidate_capacity,
        )

        self.slot_resultants = torch.zeros(
            self.slot_capacity,
            self.router_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.slot_counts = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.int64
        )
        self.slot_kappas = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.float64
        )
        self.slot_log_bases = torch.full(
            (self.slot_capacity,), -torch.inf, device=self.device, dtype=torch.float64
        )
        self.slot_active = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.bool
        )

        self.record_keys = torch.zeros(
            self.record_capacity,
            self.kv_heads,
            self.block_size,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.record_values = torch.zeros_like(self.record_keys)
        self.record_block_keys = torch.zeros(
            self.record_capacity,
            self.router_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.record_slots = torch.full(
            (self.record_capacity,), -1, device=self.device, dtype=torch.int64
        )
        self.record_start_positions = torch.full(
            (self.record_capacity,), -1, device=self.device, dtype=torch.int64
        )
        self.record_lengths = torch.zeros(
            self.record_capacity, device=self.device, dtype=torch.int64
        )
        self.record_active = torch.zeros(
            self.record_capacity, device=self.device, dtype=torch.bool
        )
        self.record_usage_ema = torch.zeros(
            self.record_capacity, device=self.device, dtype=torch.float32
        )
        self.record_recall_counts = torch.zeros(
            self.record_capacity, device=self.device, dtype=torch.int64
        )

        self._record_ids: list[str | None] = [None] * self.record_capacity
        self._record_block_ids: list[Hashable | None] = [None] * self.record_capacity
        self._record_id_to_index: dict[str, int] = {}
        self._block_id_to_index: dict[Hashable, int] = {}
        self._slot_records: list[set[int]] = [
            set() for _ in range(self.slot_capacity)
        ]
        self._slot_ids: list[str | None] = [None] * self.slot_capacity
        self._slot_generations = [0] * self.slot_capacity
        self._slot_last_position = [-1] * self.slot_capacity
        self._free_slots = list(reversed(range(self.slot_capacity)))
        self._free_records: list[int] = []
        self._next_unused_record_index = 0
        self._next_record_serial = 0

        self.ingested_records = 0
        self.ingested_tokens = 0
        self.evicted_records = 0
        self.evicted_tokens = 0
        self.evicted_slots = 0
        self.created_slots = 0
        self.assigned_existing = 0
        self.local_candidate_requests = 0
        self.local_candidate_slots_considered = 0
        self.maximum_candidate_slots_considered = 0
        self.global_assignment_scans = 0
        self.recall_observations = 0
        self.recalled_records = 0
        self.usage_eviction_passes = 0
        self.capacity_growth_events = 0

    @staticmethod
    def _grow_tensor(tensor, new_capacity: int, *, fill_value: float | int | bool = 0):
        torch, _functional = _require_torch()
        shape = (int(new_capacity),) + tuple(tensor.shape[1:])
        grown = torch.full(
            shape, fill_value, device=tensor.device, dtype=tensor.dtype
        )
        grown[: tensor.shape[0]].copy_(tensor)
        return grown

    def _ensure_slot_capacity(self, required: int) -> None:
        if required <= self.slot_capacity:
            return
        old = self.slot_capacity
        new = max(int(required), old * 2)
        self.slot_resultants = self._grow_tensor(self.slot_resultants, new)
        self.slot_counts = self._grow_tensor(self.slot_counts, new)
        self.slot_kappas = self._grow_tensor(self.slot_kappas, new)
        self.slot_log_bases = self._grow_tensor(
            self.slot_log_bases, new, fill_value=-math.inf
        )
        self.slot_active = self._grow_tensor(self.slot_active, new)
        self._slot_records.extend(set() for _ in range(new - old))
        self._slot_ids.extend([None] * (new - old))
        self._slot_generations.extend([0] * (new - old))
        self._slot_last_position.extend([-1] * (new - old))
        self._free_slots.extend(reversed(range(old, new)))
        self.slot_capacity = new
        self.capacity_growth_events += 1

    def _ensure_record_capacity(self, required: int) -> None:
        if required <= self.record_capacity:
            return
        old = self.record_capacity
        new = max(int(required), old * 2)
        self.record_keys = self._grow_tensor(self.record_keys, new)
        self.record_values = self._grow_tensor(self.record_values, new)
        self.record_block_keys = self._grow_tensor(self.record_block_keys, new)
        self.record_slots = self._grow_tensor(self.record_slots, new, fill_value=-1)
        self.record_start_positions = self._grow_tensor(
            self.record_start_positions, new, fill_value=-1
        )
        self.record_lengths = self._grow_tensor(self.record_lengths, new)
        self.record_active = self._grow_tensor(self.record_active, new)
        self.record_usage_ema = self._grow_tensor(self.record_usage_ema, new)
        self.record_recall_counts = self._grow_tensor(
            self.record_recall_counts, new
        )
        self._record_ids.extend([None] * (new - old))
        self._record_block_ids.extend([None] * (new - old))
        self.record_capacity = new
        self.capacity_growth_events += 1

    def _codes(self, directions) -> list[int]:
        return locality_codes(directions, self.locality_projection)

    def _refresh_slots(self, slot_ids) -> None:
        torch, _functional = _require_torch()
        if slot_ids.numel() == 0:
            return
        slot_ids = torch.unique(slot_ids.to(device=self.device, dtype=torch.int64))
        kappas, log_bases, active = vmf_slot_parameters(
            self.slot_resultants[slot_ids],
            self.slot_counts[slot_ids],
            dimension=self.router_dim,
            concentration_prior_mass=self.config.concentration_prior_mass,
            maximum_concentration=self.config.maximum_concentration,
            count_exponent=self.config.count_exponent,
        )
        self.slot_kappas[slot_ids] = kappas
        self.slot_log_bases[slot_ids] = log_bases
        self.slot_active[slot_ids] = active

    def _posterior(self, directions, candidate_ids, candidate_valid):
        return vmf_posterior_assignments(
            directions,
            candidate_ids,
            candidate_valid,
            slot_resultants=self.slot_resultants,
            slot_kappas=self.slot_kappas,
            slot_log_bases=self.slot_log_bases,
            dimension=self.router_dim,
            alpha=self.config.alpha,
            tau_new=self.config.tau_new,
        )

    def _remove_slot_from_cpu_index(self, slot: int) -> None:
        self._locality.remove(slot)
        self._slot_ids[slot] = None
        self._slot_last_position[slot] = -1

    def _deactivate_empty_slots(self, touched: Sequence[int]) -> None:
        for slot in set(int(value) for value in touched):
            if self._slot_records[slot]:
                continue
            if self._slot_ids[slot] is not None:
                self.evicted_slots += 1
            self._remove_slot_from_cpu_index(slot)
            self._free_slots.append(slot)

    def _evict_record_indices(self, indices: Sequence[int]) -> None:
        torch, _functional = _require_torch()
        active = [
            int(index)
            for index in indices
            if bool(self.record_active[int(index)].item())
        ]
        if not active:
            return
        record_tensor = torch.tensor(active, device=self.device, dtype=torch.int64)
        slots = self.record_slots[record_tensor]
        lengths = self.record_lengths[record_tensor]
        self.slot_resultants.index_add_(
            0, slots, -self.record_block_keys[record_tensor]
        )
        self.slot_counts.index_add_(
            0, slots, -torch.ones_like(slots, dtype=torch.int64)
        )
        self.record_active[record_tensor] = False
        self.record_slots[record_tensor] = -1
        self.record_start_positions[record_tensor] = -1
        self.record_lengths[record_tensor] = 0
        self.record_usage_ema[record_tensor] = 0.0
        self.record_recall_counts[record_tensor] = 0
        touched = slots.cpu().tolist()
        for record_index, slot in zip(active, touched):
            self._slot_records[slot].discard(record_index)
            record_id = self._record_ids[record_index]
            block_id = self._record_block_ids[record_index]
            if record_id is not None:
                self._record_id_to_index.pop(record_id, None)
            if block_id is not None:
                self._block_id_to_index.pop(block_id, None)
            self._record_ids[record_index] = None
            self._record_block_ids[record_index] = None
            self._free_records.append(record_index)
        self.evicted_records += len(active)
        self.evicted_tokens += int(lengths.sum().item())
        self._refresh_slots(slots)
        self._deactivate_empty_slots(touched)

    def _allocate_slot(self, *, code: int, position: int) -> int:
        if not self._free_slots:
            self._ensure_slot_capacity(self.slot_capacity + 1)
        slot = self._free_slots.pop()
        self._slot_generations[slot] += 1
        generation = self._slot_generations[slot]
        self._slot_ids[slot] = f"block-slot-{slot:06d}-g{generation:06d}"
        self._slot_last_position[slot] = int(position)
        self._slot_records[slot].clear()
        self.slot_resultants[slot].zero_()
        self.slot_counts[slot] = 0
        self.slot_active[slot] = True
        self._locality.update(slot, int(code))
        self.created_slots += 1
        return slot

    def _reserve_record_index(self) -> int:
        if self._free_records:
            return self._free_records.pop()
        index = self._next_unused_record_index
        self._ensure_record_capacity(index + 1)
        self._next_unused_record_index += 1
        return index

    def _candidate_tensor(self, code: int):
        torch, _functional = _require_torch()
        candidates = self._locality.candidates(int(code))
        count = len(candidates)
        self.local_candidate_requests += 1
        self.local_candidate_slots_considered += count
        self.maximum_candidate_slots_considered = max(
            self.maximum_candidate_slots_considered, count
        )
        padded = list(candidates) + [0] * (self.config.candidate_capacity - count)
        return (
            torch.tensor([padded], device=self.device, dtype=torch.int64),
            torch.tensor(
                [[True] * count + [False] * (self.config.candidate_capacity - count)],
                device=self.device,
                dtype=torch.bool,
            ),
        )

    def _refresh_cpu_locality(self, slot: int, position: int) -> None:
        if self._slot_ids[int(slot)] is None or not self._slot_records[int(slot)]:
            return
        torch, functional = _require_torch()
        direction = functional.normalize(
            self.slot_resultants[int(slot)][None, :], dim=-1, eps=1e-12
        )
        self._locality.update(int(slot), self._codes(direction)[0])
        self._slot_last_position[int(slot)] = max(
            self._slot_last_position[int(slot)], int(position)
        )

    def _block_direction(self, router_key):
        torch, functional = _require_torch()
        vector = torch.as_tensor(
            router_key, device=self.device, dtype=torch.float32
        ).detach().reshape(-1)
        if vector.numel() != self.router_dim:
            raise ValueError("router block key dimension mismatch")
        if not bool(torch.isfinite(vector).all().item()):
            raise ValueError("router block key must be finite")
        if float(torch.linalg.vector_norm(vector).item()) == 0.0:
            raise ValueError("router block key must have non-zero norm")
        return functional.normalize(vector, dim=0, eps=1e-12)

    def ingest_block(
        self,
        keys,
        values,
        *,
        router_key,
        block_id: Hashable,
        logical_positions: Sequence[int],
    ) -> str:
        """Classify and store exactly one complete block record."""

        torch, _functional = _require_torch()
        if block_id is None:
            raise ValueError("block_id cannot be None")
        try:
            hash(block_id)
        except TypeError as error:
            raise TypeError("block_id must be hashable") from error
        if block_id in self._block_id_to_index:
            raise ValueError(f"block_id {block_id!r} is already active")
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("K/V must have shape [1, heads, tokens, dim]")
        if keys.shape[0] != 1 or int(keys.shape[1]) != self.kv_heads:
            raise ValueError("K/V shape is incompatible with this layer memory")
        if int(keys.shape[3]) != self.head_dim:
            raise ValueError("K/V head dimension is incompatible with this memory")
        token_count = int(keys.shape[2])
        if token_count != self.block_size:
            raise ValueError(
                "block-record memory accepts only complete config.block_size blocks"
            )
        positions = tuple(int(position) for position in logical_positions)
        if len(positions) != token_count:
            raise ValueError("logical positions must align with K/V tokens")
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise ValueError("block positions must be contiguous")

        direction = self._block_direction(router_key)
        code = self._codes(direction[None, :])[0]
        candidate_ids, candidate_valid = self._candidate_tensor(code)
        selected, create_new = self._posterior(
            direction[None, :], candidate_ids, candidate_valid
        )
        if bool(create_new[0].item()):
            slot = self._allocate_slot(code=code, position=positions[-1])
        else:
            slot = int(selected[0].item())
            self.assigned_existing += 1
        record_index = self._reserve_record_index()
        record_tensor = torch.tensor(
            [record_index], device=self.device, dtype=torch.int64
        )
        slot_tensor = torch.tensor([slot], device=self.device, dtype=torch.int64)

        with torch.no_grad():
            self.record_keys[record_index].zero_()
            self.record_values[record_index].zero_()
            self.record_keys[record_index, :, :token_count, :].copy_(
                keys.detach().to(device=self.device, dtype=self.dtype)[0]
            )
            self.record_values[record_index, :, :token_count, :].copy_(
                values.detach().to(device=self.device, dtype=self.dtype)[0]
            )
            self.record_block_keys[record_index] = direction
            self.record_slots[record_index] = slot
            self.record_start_positions[record_index] = positions[0]
            self.record_lengths[record_index] = token_count
            self.record_active[record_index] = True
            self.slot_resultants.index_add_(0, slot_tensor, direction[None, :])
            self.slot_counts.index_add_(
                0, slot_tensor, torch.ones_like(slot_tensor, dtype=torch.int64)
            )
            self._refresh_slots(slot_tensor)

        self._next_record_serial += 1
        record_id = f"block-record-{self._next_record_serial:012d}"
        self._record_ids[record_index] = record_id
        self._record_block_ids[record_index] = block_id
        self._record_id_to_index[record_id] = record_index
        self._block_id_to_index[block_id] = record_index
        self._slot_records[slot].add(record_index)
        self._refresh_cpu_locality(slot, positions[-1])
        self.ingested_records += 1
        self.ingested_tokens += token_count
        return record_id

    def router_clusters(self, query_router_key) -> tuple[GpuBlockClusterSummary, ...]:
        """Rank block-key clusters with a learned query key."""

        torch, functional = _require_torch()
        active = torch.nonzero(self.slot_active, as_tuple=False).flatten()
        if active.numel() == 0:
            return ()
        query = self._block_direction(query_router_key)
        resultants = self.slot_resultants[active]
        norms = torch.linalg.vector_norm(resultants, dim=-1).clamp_min(1e-12)
        cosine = (resultants * query[None, :]).sum(dim=-1) / norms
        log_scores = (
            self.slot_log_bases[active]
            + self.slot_kappas[active] * cosine.to(torch.float64)
        )
        probabilities = torch.softmax(log_scores, dim=0)
        rows: list[GpuBlockClusterSummary] = []
        for slot, probability, log_score in zip(
            active.cpu().tolist(),
            probabilities.cpu().tolist(),
            log_scores.cpu().tolist(),
        ):
            ordered = sorted(
                self._slot_records[slot],
                key=lambda index: int(self.record_start_positions[index].item()),
            )
            positions = tuple(
                position
                for index in ordered
                for position in range(
                    int(self.record_start_positions[index].item()),
                    int(self.record_start_positions[index].item())
                    + int(self.record_lengths[index].item()),
                )
            )
            rows.append(
                GpuBlockClusterSummary(
                    cluster_id=str(self._slot_ids[slot]),
                    slot_index=int(slot),
                    record_ids=tuple(str(self._record_ids[index]) for index in ordered),
                    block_ids=tuple(self._record_block_ids[index] for index in ordered),
                    block_lengths=tuple(
                        int(self.record_lengths[index].item()) for index in ordered
                    ),
                    logical_positions=positions,
                    probability=float(probability),
                    log_score=float(log_score),
                    total_weight=float(self.slot_counts[slot].item()),
                )
            )
        rows.sort(key=lambda row: (-row.probability, row.cluster_id))
        return tuple(rows)

    def _pack_indices(self, indices: Sequence[int]) -> PackedBlockKV:
        torch, _functional = _require_torch()
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: int(self.record_start_positions[index].item()),
        )
        if not ordered:
            empty = self.record_keys.new_empty((1, self.kv_heads, 0, self.head_dim))
            return PackedBlockKV(empty, empty.clone(), (), (), (), (), ())
        key_parts = []
        value_parts = []
        positions: list[int] = []
        record_ids: list[str] = []
        block_ids: list[Hashable] = []
        slices: list[tuple[str, int, int]] = []
        token_record_ids: list[str] = []
        cursor = 0
        for index in ordered:
            length = int(self.record_lengths[index].item())
            start = int(self.record_start_positions[index].item())
            record_id = str(self._record_ids[index])
            key_parts.append(self.record_keys[index, :, :length, :][None, :])
            value_parts.append(self.record_values[index, :, :length, :][None, :])
            positions.extend(range(start, start + length))
            record_ids.append(record_id)
            block_ids.append(self._record_block_ids[index])
            slices.append((record_id, cursor, cursor + length))
            token_record_ids.extend([record_id] * length)
            cursor += length
        return PackedBlockKV(
            keys=torch.cat(key_parts, dim=2),
            values=torch.cat(value_parts, dim=2),
            logical_positions=tuple(positions),
            record_ids=tuple(record_ids),
            block_ids=tuple(block_ids),
            record_slices=tuple(slices),
            token_record_ids=tuple(token_record_ids),
        )

    def selected_kv_blocks(self, cluster_ids: Sequence[str]) -> PackedBlockKV:
        selected = set(str(value) for value in cluster_ids)
        indices: set[int] = set()
        for slot, slot_id in enumerate(self._slot_ids):
            if slot_id in selected:
                indices.update(self._slot_records[slot])
        return self._pack_indices(tuple(indices))

    def selected_kv(self, cluster_ids: Sequence[str]):
        packed = self.selected_kv_blocks(cluster_ids)
        return packed.keys, packed.values, packed.logical_positions

    def all_kv_blocks(self) -> PackedBlockKV:
        return self.selected_kv_blocks(
            tuple(slot_id for slot_id in self._slot_ids if slot_id is not None)
        )

    def kv_for_record_ids(self, record_ids: Sequence[str]) -> PackedBlockKV:
        try:
            indices = [self._record_id_to_index[str(value)] for value in record_ids]
        except KeyError as error:
            raise KeyError(f"unknown or inactive block record ID {error.args[0]!r}") from error
        if len(set(indices)) != len(indices):
            raise ValueError("record_ids cannot contain duplicates")
        return self._pack_indices(indices)

    def _prepare_recall_usage(self, record_ids: Sequence[str], usage_rates):
        torch, _functional = _require_torch()
        external_ids = tuple(str(value) for value in record_ids)
        usage = torch.as_tensor(
            usage_rates, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if len(external_ids) != int(usage.numel()):
            raise ValueError("record_ids and usage_rates must have the same length")
        if not external_ids:
            return external_ids, usage, [], set()
        if not bool(torch.isfinite(usage).all().item()) or bool((usage < 0).any().item()):
            raise ValueError("usage_rates must be finite and non-negative")
        try:
            indices = [self._record_id_to_index[record_id] for record_id in external_ids]
        except KeyError as error:
            raise KeyError(f"unknown or inactive block record ID {error.args[0]!r}") from error
        if len(set(indices)) != len(indices):
            raise ValueError("record_ids cannot contain duplicates")
        slots = {int(self.record_slots[index].item()) for index in indices}
        supplied = set(indices)
        for slot in slots:
            expected = self._slot_records[slot]
            if not expected.issubset(supplied):
                missing = len(expected - supplied)
                raise ValueError(
                    "usage observation must cover every active block record in each "
                    f"recalled cluster; slot {slot} is missing {missing} records"
                )
        return external_ids, usage, indices, slots

    def plan_recall_eviction(
        self,
        record_ids: Sequence[str],
        usage_rates,
        *,
        usage_threshold: float | None = None,
    ) -> dict[str, Any]:
        torch, _functional = _require_torch()
        _external_ids, usage, indices, slots = self._prepare_recall_usage(
            record_ids, usage_rates
        )
        if not indices:
            return {"retained_record_ids": (), "evicted_record_ids": ()}
        threshold = (
            self.config.eviction_usage_threshold
            if usage_threshold is None
            else float(usage_threshold)
        )
        if threshold < 0.0:
            raise ValueError("usage_threshold cannot be negative")
        index_tensor = torch.tensor(indices, device=self.device, dtype=torch.int64)
        previous_counts = self.record_recall_counts[index_tensor]
        previous_usage = self.record_usage_ema[index_tensor]
        rate = float(self.config.usage_ema_rate)
        updated = torch.where(
            previous_counts == 0,
            usage,
            previous_usage * (1.0 - rate) + usage * rate,
        )
        prospective_counts = previous_counts + 1
        updated_by_index = {
            index: float(value)
            for index, value in zip(indices, updated.detach().cpu().tolist())
        }
        count_by_index = {
            index: int(value)
            for index, value in zip(
                indices, prospective_counts.detach().cpu().tolist()
            )
        }
        evict: list[int] = []
        for slot in slots:
            members = sorted(self._slot_records[slot])
            eligible = [
                index
                for index in members
                if count_by_index[index] >= self.config.eviction_min_recall_count
                and updated_by_index[index] < threshold
            ]
            maximum = max(
                0, len(members) - self.config.eviction_min_records_per_cluster
            )
            eligible.sort(
                key=lambda index: (
                    updated_by_index[index],
                    int(self.record_start_positions[index].item()),
                )
            )
            evict.extend(eligible[:maximum])
        evicted = set(evict)
        retained = sorted(
            (index for index in indices if index not in evicted),
            key=lambda index: int(self.record_start_positions[index].item()),
        )
        evict.sort(key=lambda index: int(self.record_start_positions[index].item()))
        return {
            "retained_record_ids": tuple(
                str(self._record_ids[index]) for index in retained
            ),
            "evicted_record_ids": tuple(
                str(self._record_ids[index]) for index in evict
            ),
            "recalled_clusters": len(slots),
            "recalled_records": len(indices),
            "usage_threshold": threshold,
        }

    def observe_recall_usage(
        self,
        record_ids: Sequence[str],
        usage_rates,
        *,
        apply_eviction: bool | None = None,
    ) -> dict[str, Any]:
        torch, _functional = _require_torch()
        external_ids, usage, indices, slots = self._prepare_recall_usage(
            record_ids, usage_rates
        )
        if not external_ids:
            return {
                "recalled_clusters": 0,
                "recalled_records": 0,
                "evicted_records": 0,
            }
        enabled = (
            self.config.usage_eviction_enabled
            if apply_eviction is None
            else bool(apply_eviction)
        )
        evict: list[int] = []
        if enabled:
            plan = self.plan_recall_eviction(external_ids, usage)
            evict = [
                self._record_id_to_index[record_id]
                for record_id in plan["evicted_record_ids"]
            ]
        index_tensor = torch.tensor(indices, device=self.device, dtype=torch.int64)
        previous_counts = self.record_recall_counts[index_tensor]
        previous_usage = self.record_usage_ema[index_tensor]
        rate = float(self.config.usage_ema_rate)
        updated = torch.where(
            previous_counts == 0,
            usage,
            previous_usage * (1.0 - rate) + usage * rate,
        )
        self.record_usage_ema[index_tensor] = updated
        self.record_recall_counts[index_tensor] = previous_counts + 1
        self.recall_observations += len(slots)
        self.recalled_records += len(indices)
        if enabled:
            self.usage_eviction_passes += 1
            self._evict_record_indices(evict)
        return {
            "recalled_clusters": len(slots),
            "recalled_records": len(indices),
            "evicted_records": len(evict),
            "eviction_enabled": enabled,
            "usage_threshold": float(self.config.eviction_usage_threshold),
        }

    @property
    def active_record_count(self) -> int:
        return sum(len(records) for records in self._slot_records)

    @property
    def active_token_count(self) -> int:
        return sum(
            int(self.record_lengths[index].item())
            for records in self._slot_records
            for index in records
        )

    @property
    def active_slot_count(self) -> int:
        return sum(slot_id is not None for slot_id in self._slot_ids)

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.locality_projection,
            self.slot_resultants,
            self.slot_counts,
            self.slot_kappas,
            self.slot_log_bases,
            self.slot_active,
            self.record_keys,
            self.record_values,
            self.record_block_keys,
            self.record_slots,
            self.record_start_positions,
            self.record_lengths,
            self.record_active,
            self.record_usage_ema,
            self.record_recall_counts,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "gpu_router_key_block_vmf_clusters",
            "device": str(self.device),
            "record_unit": "layer_local_block",
            "classification_feature": "learned_router_block_key",
            "payload": "original_complete_block_kv",
            "block_size": self.block_size,
            "active_slots": self.active_slot_count,
            "allocated_slot_capacity": self.slot_capacity,
            "active_records": self.active_record_count,
            "active_tokens": self.active_token_count,
            "allocated_record_capacity": self.record_capacity,
            "ingested_records": self.ingested_records,
            "ingested_tokens": self.ingested_tokens,
            "evicted_records": self.evicted_records,
            "evicted_tokens": self.evicted_tokens,
            "evicted_slots": self.evicted_slots,
            "created_slots": self.created_slots,
            "assigned_existing": self.assigned_existing,
            "local_candidate_requests": self.local_candidate_requests,
            "local_candidate_slots_considered": self.local_candidate_slots_considered,
            "maximum_candidate_slots_considered": self.maximum_candidate_slots_considered,
            "candidate_capacity": self.config.candidate_capacity,
            "global_assignment_scans": self.global_assignment_scans,
            "cluster_capacity_policy": "dynamic_unbounded",
            "record_capacity_policy": "dynamic_unbounded",
            "assignment_commit_order": "one_block_one_posterior",
            "assignment_transaction_size": 1,
            "memory_bytes": self.memory_bytes,
            "memory_budget_bytes": self.config.memory_budget_bytes,
            "memory_budget_role": "legacy_eviction_enable_switch_only",
            "usage_eviction_enabled": self.config.usage_eviction_enabled,
            "eviction_scope": "recalled_cluster_only",
            "usage_metric": "mean_attention_probability_per_historical_block",
            "recall_observations": self.recall_observations,
            "recalled_records": self.recalled_records,
            "usage_eviction_passes": self.usage_eviction_passes,
            "capacity_growth_events": self.capacity_growth_events,
            "cpu_prefetch_payload": "bounded_cluster_ids_only",
            "numerical_ingestion_device": str(self.device),
        }


__all__ = [
    "GpuBlockClusterMemory",
    "GpuBlockClusterMemoryConfig",
    "GpuBlockClusterSummary",
    "PackedBlockKV",
]
