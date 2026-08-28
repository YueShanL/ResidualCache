"""GPU-resident token records with CPU-prefetched local vMF writes.

The numerical ingestion path deliberately never scores an incoming record
against every cluster.  A small CPU locality index maps a compact random-
hyperplane code to a bounded candidate list.  Only those candidate slot IDs are
sent back to the GPU, where exact vMF posterior assignment and all sufficient-
statistic updates are performed.

The learned router key is independent metadata.  It contributes to a second
per-slot vMF distribution, normalized by the mechanical block size, and never
participates in native K/V classification.
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
        raise RuntimeError("GPU local cluster memory requires torch") from error
    return torch, functional


@dataclass(frozen=True)
class GpuLocalClusterMemoryConfig:
    """Allocation hints, posterior policy, and cluster-local eviction policy.

    ``memory_budget_bytes`` is retained as a compatibility switch only.  A
    non-``None`` value enables usage-based eviction, but is never treated as a
    global byte cap.  The tensor capacities are allocation hints which grow on
    demand and therefore do not bound the number of records or clusters.
    """

    memory_budget_bytes: int | None = None
    eviction_enabled: bool | None = None
    slot_capacity: int = 128
    initial_record_capacity: int = 1024
    candidate_capacity: int = 8
    locality_bits: int = 8
    locality_probe_radius: int = 1
    # One mechanical block is one transaction: all records obtain their
    # posterior from the same pre-commit state and commit together.
    write_chunk_size: int = 64
    alpha: float = 0.1
    tau_new: float = 0.5
    count_exponent: float = 0.5
    concentration_prior_mass: float = 1.0
    maximum_concentration: float = 1_000.0
    router_count_exponent: float = 0.5
    router_concentration_prior_mass: float = 1.0
    router_maximum_concentration: float = 1_000.0
    index_mode: str = "mean_kv"
    locality_seed: int = 13
    usage_ema_rate: float = 0.25
    eviction_usage_threshold: float = 1e-4
    eviction_min_recall_count: int = 1
    eviction_min_records_per_cluster: int = 1

    def __post_init__(self) -> None:
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when provided")
        for name in (
            "slot_capacity",
            "initial_record_capacity",
            "candidate_capacity",
            "locality_bits",
            "write_chunk_size",
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
        if not 0.0 <= self.router_count_exponent < 1.0:
            raise ValueError("router_count_exponent must be in [0, 1)")
        if self.concentration_prior_mass < 0.0:
            raise ValueError("concentration_prior_mass cannot be negative")
        if self.router_concentration_prior_mass < 0.0:
            raise ValueError("router_concentration_prior_mass cannot be negative")
        if min(self.maximum_concentration, self.router_maximum_concentration) <= 0.0:
            raise ValueError("maximum concentrations must be positive")
        if self.index_mode not in {"key", "mean_kv"}:
            raise ValueError("index_mode must be 'key' or 'mean_kv'")
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
class GpuClusterSummary:
    cluster_id: str
    slot_index: int
    record_ids: tuple[str, ...]
    logical_positions: tuple[int, ...]
    block_ids: tuple[Hashable, ...]
    probability: float
    log_score: float
    total_weight: float


class GpuLocalClusterMemory:
    """One layer's dynamically growing GPU K/V database and vMF indices."""

    def __init__(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        router_dim: int,
        device: Any,
        dtype: Any,
        config: GpuLocalClusterMemoryConfig | None = None,
    ) -> None:
        torch, functional = _require_torch()
        self.config = config or GpuLocalClusterMemoryConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.flat_dim = self.kv_heads * self.head_dim
        self.router_dim = int(router_dim)
        if min(self.kv_heads, self.head_dim, self.router_dim) <= 0:
            raise ValueError("K/V and router dimensions must be positive")
        if min(self.flat_dim, self.router_dim) < 32:
            raise ValueError("vMF dimensions must be at least 32")

        self.slot_capacity = int(self.config.slot_capacity)
        self.record_capacity = max(
            int(self.config.initial_record_capacity),
            int(self.config.write_chunk_size),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.locality_seed))
        projection = torch.randn(
            self.config.locality_bits,
            self.flat_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.locality_projection = functional.normalize(projection, dim=-1).to(self.device)
        self._locality = CpuLocalityIndex(
            bits=self.config.locality_bits,
            probe_radius=self.config.locality_probe_radius,
            candidate_capacity=self.config.candidate_capacity,
        )

        self.slot_resultants = torch.zeros(
            self.slot_capacity, self.flat_dim, device=self.device, dtype=torch.float32
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
        self.slot_router_resultants = torch.zeros(
            self.slot_capacity, self.router_dim, device=self.device, dtype=torch.float32
        )
        self.slot_router_weights = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.float32
        )
        self.slot_router_kappas = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.float64
        )
        self.slot_router_log_bases = torch.full(
            (self.slot_capacity,), -torch.inf, device=self.device, dtype=torch.float64
        )
        self.slot_active = torch.zeros(
            self.slot_capacity, device=self.device, dtype=torch.bool
        )

        self.record_keys = torch.zeros(
            self.record_capacity, self.kv_heads, self.head_dim,
            device=self.device, dtype=dtype,
        )
        self.record_values = torch.zeros_like(self.record_keys)
        self.record_router_keys = torch.zeros(
            self.record_capacity, self.router_dim, device=self.device, dtype=torch.float32
        )
        self.record_router_weights = torch.zeros(
            self.record_capacity, device=self.device, dtype=torch.float32
        )
        self.record_slots = torch.full(
            (self.record_capacity,), -1, device=self.device, dtype=torch.int64
        )
        self.record_positions = torch.full(
            (self.record_capacity,), -1, device=self.device, dtype=torch.int64
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
        self._slot_records: list[set[int]] = [set() for _ in range(self.slot_capacity)]
        self._slot_ids: list[str | None] = [None] * self.slot_capacity
        self._slot_generations = [0] * self.slot_capacity
        self._slot_last_position = [-1] * self.slot_capacity
        self._free_slots = list(reversed(range(self.slot_capacity)))
        self._free_records: list[int] = []
        self._next_unused_record_index = 0
        self._next_record_serial = 0

        self.ingested_records = 0
        self.evicted_records = 0
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
            shape,
            fill_value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        grown[: tensor.shape[0]].copy_(tensor)
        return grown

    def _ensure_slot_capacity(self, required: int) -> None:
        if required <= self.slot_capacity:
            return
        old = self.slot_capacity
        new = max(required, old * 2)
        self.slot_resultants = self._grow_tensor(self.slot_resultants, new)
        self.slot_counts = self._grow_tensor(self.slot_counts, new)
        self.slot_kappas = self._grow_tensor(self.slot_kappas, new)
        self.slot_log_bases = self._grow_tensor(
            self.slot_log_bases, new, fill_value=-math.inf
        )
        self.slot_router_resultants = self._grow_tensor(
            self.slot_router_resultants, new
        )
        self.slot_router_weights = self._grow_tensor(self.slot_router_weights, new)
        self.slot_router_kappas = self._grow_tensor(self.slot_router_kappas, new)
        self.slot_router_log_bases = self._grow_tensor(
            self.slot_router_log_bases, new, fill_value=-math.inf
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
        new = max(required, old * 2)
        self.record_keys = self._grow_tensor(self.record_keys, new)
        self.record_values = self._grow_tensor(self.record_values, new)
        self.record_router_keys = self._grow_tensor(self.record_router_keys, new)
        self.record_router_weights = self._grow_tensor(self.record_router_weights, new)
        self.record_slots = self._grow_tensor(self.record_slots, new, fill_value=-1)
        self.record_positions = self._grow_tensor(
            self.record_positions, new, fill_value=-1
        )
        self.record_active = self._grow_tensor(self.record_active, new)
        self.record_usage_ema = self._grow_tensor(self.record_usage_ema, new)
        self.record_recall_counts = self._grow_tensor(
            self.record_recall_counts, new
        )
        self._record_ids.extend([None] * (new - old))
        self._record_block_ids.extend([None] * (new - old))
        self.record_capacity = new
        self.capacity_growth_events += 1

    def _classification_vectors(self, flat_keys, flat_values):
        torch, functional = _require_torch()
        values = flat_keys if self.config.index_mode == "key" else (flat_keys + flat_values) * 0.5
        return functional.normalize(values.float(), dim=-1, eps=1e-12)

    def _codes(self, directions) -> list[int]:
        return locality_codes(directions, self.locality_projection)

    def _refresh_slots(self, slot_ids) -> None:
        torch, _functional = _require_torch()
        if slot_ids.numel() == 0:
            return
        slot_ids = torch.unique(slot_ids.to(device=self.device, dtype=torch.int64))
        kappa, log_base, active = vmf_slot_parameters(
            self.slot_resultants[slot_ids],
            self.slot_counts[slot_ids],
            dimension=self.flat_dim,
            concentration_prior_mass=self.config.concentration_prior_mass,
            maximum_concentration=self.config.maximum_concentration,
            count_exponent=self.config.count_exponent,
        )
        self.slot_kappas[slot_ids] = kappa
        self.slot_log_bases[slot_ids] = log_base

        router_weights = self.slot_router_weights[slot_ids].to(torch.float64)
        router_kappa, router_log_base, router_active = vmf_slot_parameters(
            self.slot_router_resultants[slot_ids],
            router_weights,
            dimension=self.router_dim,
            concentration_prior_mass=self.config.router_concentration_prior_mass,
            maximum_concentration=self.config.router_maximum_concentration,
            count_exponent=self.config.router_count_exponent,
        )
        router_active = active & router_active
        self.slot_router_kappas[slot_ids] = torch.where(
            router_active, router_kappa, torch.zeros_like(router_kappa)
        )
        self.slot_router_log_bases[slot_ids] = torch.where(
            router_active,
            router_log_base,
            torch.full_like(router_log_base, -torch.inf),
        )
        self.slot_active[slot_ids] = active

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
        torch, functional = _require_torch()
        active = [int(index) for index in indices if bool(self.record_active[int(index)].item())]
        if not active:
            return
        record_ids = torch.tensor(active, device=self.device, dtype=torch.int64)
        slots = self.record_slots[record_ids]
        flat_keys = self.record_keys[record_ids].reshape(len(active), self.flat_dim)
        flat_values = self.record_values[record_ids].reshape(len(active), self.flat_dim)
        directions = self._classification_vectors(flat_keys, flat_values)
        router_directions = functional.normalize(
            self.record_router_keys[record_ids], dim=-1, eps=1e-12
        )
        router_weights = self.record_router_weights[record_ids]
        self.slot_resultants.index_add_(0, slots, -directions)
        self.slot_counts.index_add_(
            0, slots, -torch.ones_like(slots, dtype=torch.int64)
        )
        self.slot_router_resultants.index_add_(
            0, slots, -router_directions * router_weights[:, None]
        )
        self.slot_router_weights.index_add_(0, slots, -router_weights)
        self.slot_router_weights.clamp_min_(0.0)
        self.record_active[record_ids] = False
        self.record_slots[record_ids] = -1
        self.record_positions[record_ids] = -1
        self.record_router_weights[record_ids] = 0.0
        self.record_usage_ema[record_ids] = 0.0
        self.record_recall_counts[record_ids] = 0
        touched = slots.cpu().tolist()
        for record_index, slot in zip(active, touched):
            self._slot_records[slot].discard(record_index)
            record_id = self._record_ids[record_index]
            if record_id is not None:
                self._record_id_to_index.pop(record_id, None)
            self._record_ids[record_index] = None
            self._record_block_ids[record_index] = None
            self._free_records.append(record_index)
        self.evicted_records += len(active)
        self._refresh_slots(slots)
        self._deactivate_empty_slots(touched)

    def _allocate_slot(
        self,
        *,
        code: int,
        position: int,
        protected_slots: set[int],
    ) -> int:
        del protected_slots
        if not self._free_slots:
            self._ensure_slot_capacity(self.slot_capacity + 1)
        slot = self._free_slots.pop()
        self._slot_generations[slot] += 1
        generation = self._slot_generations[slot]
        self._slot_ids[slot] = f"slot-{slot:06d}-g{generation:06d}"
        self._slot_last_position[slot] = int(position)
        self._slot_records[slot].clear()
        self.slot_resultants[slot].zero_()
        self.slot_counts[slot] = 0
        self.slot_router_resultants[slot].zero_()
        self.slot_router_weights[slot] = 0
        self.slot_active[slot] = True
        self._locality.update(slot, int(code))
        self.created_slots += 1
        return slot

    def _reserve_record_indices(self, count: int) -> list[int]:
        indices: list[int] = []
        while self._free_records and len(indices) < count:
            indices.append(self._free_records.pop())
        remaining = count - len(indices)
        if remaining:
            end = self._next_unused_record_index + remaining
            self._ensure_record_capacity(end)
            indices.extend(range(self._next_unused_record_index, end))
            self._next_unused_record_index = end
        return indices

    def _candidate_tensor(self, codes: Sequence[int]):
        torch, _functional = _require_torch()
        width = self.config.candidate_capacity
        rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        for code in codes:
            candidates = self._locality.candidates(int(code))
            count = len(candidates)
            self.local_candidate_requests += 1
            self.local_candidate_slots_considered += count
            self.maximum_candidate_slots_considered = max(
                self.maximum_candidate_slots_considered, count
            )
            padded = list(candidates) + [0] * (width - count)
            rows.append(padded)
            valid_rows.append([True] * count + [False] * (width - count))
        return (
            torch.tensor(rows, device=self.device, dtype=torch.int64),
            torch.tensor(valid_rows, device=self.device, dtype=torch.bool),
        )

    def _posterior(self, directions, candidate_ids, candidate_valid):
        return vmf_posterior_assignments(
            directions,
            candidate_ids,
            candidate_valid,
            slot_resultants=self.slot_resultants,
            slot_kappas=self.slot_kappas,
            slot_log_bases=self.slot_log_bases,
            dimension=self.flat_dim,
            alpha=self.config.alpha,
            tau_new=self.config.tau_new,
        )

    def _refresh_cpu_locality(self, touched: Sequence[int], position: int) -> None:
        active = sorted(
            {
                int(slot)
                for slot in touched
                if self._slot_ids[int(slot)] is not None
                and self._slot_records[int(slot)]
            }
        )
        if not active:
            return
        torch, functional = _require_torch()
        slot_ids = torch.tensor(active, device=self.device, dtype=torch.int64)
        directions = functional.normalize(
            self.slot_resultants[slot_ids], dim=-1, eps=1e-12
        )
        codes = self._codes(directions)
        for slot, code in zip(active, codes):
            self._locality.update(slot, code)
            self._slot_last_position[slot] = max(
                self._slot_last_position[slot], int(position)
            )

    def ingest_block(
        self,
        keys,
        values,
        *,
        router_key,
        block_id: Hashable,
        logical_positions: Sequence[int],
        router_block_size: int | None = None,
    ) -> None:
        """Ingest an evicted span as one or more pre-commit block transactions.

        Every record still obtains its own candidate posterior. Records in the
        same transaction are intentionally scored against one shared memory
        state and their sufficient statistics are committed together.
        """

        torch, functional = _require_torch()
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("K/V must have shape [1, heads, tokens, dim]")
        if keys.shape[0] != 1 or tuple(keys.shape[1::2]) != (
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("K/V shape is incompatible with this layer memory")
        token_count = int(keys.shape[2])
        positions = tuple(int(position) for position in logical_positions)
        if len(positions) != token_count or not positions:
            raise ValueError("logical positions must align with K/V tokens")
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise ValueError("block positions must be contiguous")
        block_size = token_count if router_block_size is None else int(router_block_size)
        if block_size <= 0:
            raise ValueError("router_block_size must be positive")
        router = torch.as_tensor(
            router_key, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if router.numel() != self.router_dim:
            raise ValueError("router key dimension mismatch")
        router_direction = functional.normalize(router, dim=0, eps=1e-12)
        router_weight = 1.0 / block_size
        flat_keys = keys.detach().to(device=self.device, dtype=self.dtype).transpose(1, 2)[
            0
        ].reshape(token_count, self.flat_dim)
        flat_values = values.detach().to(device=self.device, dtype=self.dtype).transpose(1, 2)[
            0
        ].reshape(token_count, self.flat_dim)

        with torch.no_grad():
            for start in range(0, token_count, self.config.write_chunk_size):
                end = min(token_count, start + self.config.write_chunk_size)
                chunk_keys = flat_keys[start:end]
                chunk_values = flat_values[start:end]
                chunk_positions = positions[start:end]
                directions = self._classification_vectors(chunk_keys, chunk_values)
                codes = self._codes(directions)
                record_indices = self._reserve_record_indices(end - start)
                candidate_ids, candidate_valid = self._candidate_tensor(codes)
                selected, create_new = self._posterior(
                    directions, candidate_ids, candidate_valid
                )
                selected_cpu = selected.cpu().tolist()
                create_cpu = create_new.cpu().tolist()
                # Posterior decisions are made together on the GPU.  A later
                # create-new row must therefore not recycle a slot already
                # selected by an earlier row before the chunk is committed.
                protected_slots = {
                    int(slot)
                    for slot, should_create in zip(selected_cpu, create_cpu)
                    if not should_create
                }
                for offset, should_create in enumerate(create_cpu):
                    if should_create:
                        selected_cpu[offset] = self._allocate_slot(
                            code=codes[offset],
                            position=chunk_positions[offset],
                            protected_slots=protected_slots,
                        )
                        protected_slots.add(selected_cpu[offset])
                    else:
                        self.assigned_existing += 1
                selected = torch.tensor(
                    selected_cpu, device=self.device, dtype=torch.int64
                )
                record_tensor = torch.tensor(
                    record_indices, device=self.device, dtype=torch.int64
                )
                self.record_keys[record_tensor] = chunk_keys.reshape(
                    -1, self.kv_heads, self.head_dim
                )
                self.record_values[record_tensor] = chunk_values.reshape(
                    -1, self.kv_heads, self.head_dim
                )
                self.record_router_keys[record_tensor] = router_direction
                self.record_router_weights[record_tensor] = router_weight
                self.record_slots[record_tensor] = selected
                self.record_positions[record_tensor] = torch.tensor(
                    chunk_positions, device=self.device, dtype=torch.int64
                )
                self.record_active[record_tensor] = True
                self.slot_resultants.index_add_(0, selected, directions)
                self.slot_counts.index_add_(
                    0, selected, torch.ones_like(selected, dtype=torch.int64)
                )
                self.slot_router_resultants.index_add_(
                    0,
                    selected,
                    router_direction[None, :].expand(end - start, -1) * router_weight,
                )
                self.slot_router_weights.index_add_(
                    0,
                    selected,
                    torch.full(
                        (end - start,),
                        router_weight,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
                self._refresh_slots(selected)
                for offset, (record_index, slot, position) in enumerate(
                    zip(record_indices, selected_cpu, chunk_positions)
                ):
                    self._next_record_serial += 1
                    record_id = f"record-{self._next_record_serial:012d}"
                    self._record_ids[record_index] = record_id
                    self._record_id_to_index[record_id] = record_index
                    self._record_block_ids[record_index] = block_id
                    self._slot_records[slot].add(record_index)
                    self._slot_last_position[slot] = max(
                        self._slot_last_position[slot], int(position)
                    )
                self._refresh_cpu_locality(selected_cpu, max(chunk_positions))
                self.ingested_records += end - start

    def router_clusters(self, query_router_key) -> tuple[GpuClusterSummary, ...]:
        """Rank current clusters using only their memory-owned router vMF."""

        torch, functional = _require_torch()
        active = torch.nonzero(self.slot_active, as_tuple=False).flatten()
        if active.numel() == 0:
            return ()
        query = torch.as_tensor(
            query_router_key, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if query.numel() != self.router_dim:
            raise ValueError("query router key dimension mismatch")
        query = functional.normalize(query, dim=0, eps=1e-12)
        resultants = self.slot_router_resultants[active]
        norms = torch.linalg.vector_norm(resultants, dim=-1).clamp_min(1e-12)
        cosine = (resultants * query[None, :]).sum(dim=-1) / norms
        log_scores = (
            self.slot_router_log_bases[active]
            + self.slot_router_kappas[active] * cosine.to(torch.float64)
        )
        probabilities = torch.softmax(log_scores, dim=0)
        active_cpu = active.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()
        scores_cpu = log_scores.cpu().tolist()
        rows: list[GpuClusterSummary] = []
        for slot, probability, log_score in zip(
            active_cpu, probabilities_cpu, scores_cpu
        ):
            ordered = sorted(
                self._slot_records[slot],
                key=lambda index: int(self.record_positions[index].item()),
            )
            rows.append(
                GpuClusterSummary(
                    cluster_id=str(self._slot_ids[slot]),
                    slot_index=slot,
                    record_ids=tuple(str(self._record_ids[index]) for index in ordered),
                    logical_positions=tuple(
                        int(self.record_positions[index].item()) for index in ordered
                    ),
                    block_ids=tuple(self._record_block_ids[index] for index in ordered),
                    probability=float(probability),
                    log_score=float(log_score),
                    total_weight=float(self.slot_router_weights[slot].item()),
                )
            )
        rows.sort(key=lambda row: (-row.probability, row.cluster_id))
        return tuple(rows)

    def selected_kv_records(self, cluster_ids: Sequence[str]):
        """Pack complete selected clusters and return aligned record IDs."""

        torch, _functional = _require_torch()
        selected = set(str(value) for value in cluster_ids)
        indices: set[int] = set()
        for slot, slot_id in enumerate(self._slot_ids):
            if slot_id in selected:
                indices.update(self._slot_records[slot])
        ordered = sorted(indices, key=lambda index: int(self.record_positions[index].item()))
        if not ordered:
            empty = self.record_keys.new_empty((1, self.kv_heads, 0, self.head_dim))
            return empty, empty.clone(), (), ()
        record_ids = torch.tensor(ordered, device=self.device, dtype=torch.int64)
        keys = self.record_keys[record_ids].transpose(0, 1).unsqueeze(0)
        values = self.record_values[record_ids].transpose(0, 1).unsqueeze(0)
        positions = tuple(int(self.record_positions[index].item()) for index in ordered)
        external_ids = tuple(str(self._record_ids[index]) for index in ordered)
        return keys, values, positions, external_ids

    def selected_kv(self, cluster_ids: Sequence[str]):
        """Pack every active record from selected clusters in logical order."""

        keys, values, positions, _record_ids = self.selected_kv_records(cluster_ids)
        return keys, values, positions

    def all_kv_records(self):
        """Pack every retained record without invoking the learned router."""

        return self.selected_kv_records(
            tuple(slot_id for slot_id in self._slot_ids if slot_id is not None)
        )

    def kv_for_record_ids(self, record_ids: Sequence[str]):
        """Pack an explicit retained-record plan in logical position order."""

        torch, _functional = _require_torch()
        try:
            indices = [self._record_id_to_index[str(value)] for value in record_ids]
        except KeyError as error:
            raise KeyError(f"unknown or inactive record ID {error.args[0]!r}") from error
        if len(set(indices)) != len(indices):
            raise ValueError("record_ids cannot contain duplicates")
        ordered = sorted(indices, key=lambda index: int(self.record_positions[index].item()))
        if not ordered:
            empty = self.record_keys.new_empty((1, self.kv_heads, 0, self.head_dim))
            return empty, empty.clone(), (), ()
        index_tensor = torch.tensor(ordered, device=self.device, dtype=torch.int64)
        return (
            self.record_keys[index_tensor].transpose(0, 1).unsqueeze(0),
            self.record_values[index_tensor].transpose(0, 1).unsqueeze(0),
            tuple(int(self.record_positions[index].item()) for index in ordered),
            tuple(str(self._record_ids[index]) for index in ordered),
        )

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
            raise KeyError(f"unknown or inactive record ID {error.args[0]!r}") from error
        if len(set(indices)) != len(indices):
            raise ValueError("record_ids cannot contain duplicates")
        slots = {int(self.record_slots[index].item()) for index in indices}
        supplied = set(indices)
        for slot in slots:
            expected = self._slot_records[slot]
            if not expected.issubset(supplied):
                missing = len(expected - supplied)
                raise ValueError(
                    "usage observation must cover every active record in each "
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
        """Plan one cluster-local usage eviction pass without mutating memory."""

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
                0,
                len(members) - self.config.eviction_min_records_per_cluster,
            )
            eligible.sort(
                key=lambda index: (
                    updated_by_index[index],
                    int(self.record_positions[index].item()),
                )
            )
            evict.extend(eligible[:maximum])
        evicted = set(evict)
        retained = sorted(
            (index for index in indices if index not in evicted),
            key=lambda index: int(self.record_positions[index].item()),
        )
        evict.sort(key=lambda index: int(self.record_positions[index].item()))
        return {
            "retained_record_ids": tuple(str(self._record_ids[index]) for index in retained),
            "evicted_record_ids": tuple(str(self._record_ids[index]) for index in evict),
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
        """Update record usage and optionally evict only within recalled clusters.

        The caller must provide all current records for every cluster represented
        in ``record_ids``.  This prevents a partial recall from being mistaken
        for zero use.  Records in clusters absent from this call are untouched.
        """

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
            self.slot_router_resultants,
            self.slot_router_weights,
            self.slot_router_kappas,
            self.slot_router_log_bases,
            self.slot_active,
            self.record_keys,
            self.record_values,
            self.record_router_keys,
            self.record_router_weights,
            self.record_slots,
            self.record_positions,
            self.record_active,
            self.record_usage_ema,
            self.record_recall_counts,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "gpu_local_vmf_clusters",
            "device": str(self.device),
            "index_mode": self.config.index_mode,
            "active_slots": self.active_slot_count,
            "allocated_slot_capacity": self.slot_capacity,
            "active_records": self.active_record_count,
            "allocated_record_capacity": self.record_capacity,
            "ingested_records": self.ingested_records,
            "evicted_records": self.evicted_records,
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
            "assignment_commit_order": "block_precommit_simultaneous",
            "assignment_transaction_size": self.config.write_chunk_size,
            "memory_bytes": self.memory_bytes,
            "memory_budget_bytes": self.config.memory_budget_bytes,
            "memory_budget_role": "legacy_eviction_enable_switch_only",
            "usage_eviction_enabled": self.config.usage_eviction_enabled,
            "eviction_scope": "recalled_cluster_only",
            "usage_metric": "mean_attention_probability_per_historical_record",
            "recall_observations": self.recall_observations,
            "recalled_records": self.recalled_records,
            "usage_eviction_passes": self.usage_eviction_passes,
            "capacity_growth_events": self.capacity_growth_events,
            "cpu_prefetch_payload": "bounded_cluster_ids_only",
            "numerical_ingestion_device": str(self.device),
        }


__all__ = [
    "GpuClusterSummary",
    "GpuLocalClusterMemory",
    "GpuLocalClusterMemoryConfig",
]
