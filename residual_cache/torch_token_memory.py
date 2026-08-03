"""Torch-native token-level memories for standalone model evaluation.

This module intentionally does not import or call the residual-index pipeline.
It provides two online memories with the same model-facing contract:

* ``camelot`` stores one count-weighted average K/V payload per slot.
* ``vmf_records`` uses vMF posterior assignment for routing slots while every
  write keeps its original K/V payload as a separately retrievable record.

Both memories are scoped per decoder layer and per batch stream.  Retrieval is
performed before the current window is written, matching CAMELoT's online
evaluation protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised by the CLI.
        raise RuntimeError(
            "Model evaluation requires torch from the project environment."
        ) from exc
    return torch, functional


@dataclass(frozen=True)
class TokenMemoryConfig:
    """Configuration shared by the model-facing token memories."""

    method: str = "camelot"
    slot_capacity: int = 10_000
    record_capacity: int = 10_000
    camelot_threshold: float = 0.93
    route_top_k: int = 4
    vmf_write_chunk_size: int = 32
    alpha: float = 0.1
    tau_new: float = 0.5
    count_exponent: float = 0.5
    concentration_prior_mass: float = 1.0
    maximum_concentration: float = 1_000.0

    def __post_init__(self) -> None:
        if self.method not in {"camelot", "vmf_records"}:
            raise ValueError(f"Unknown token-memory method {self.method!r}.")
        if self.slot_capacity <= 0 or self.record_capacity <= 0:
            raise ValueError("Memory capacities must be positive.")
        if not -1.0 <= self.camelot_threshold <= 1.0:
            raise ValueError("camelot_threshold must be in [-1, 1].")
        if self.route_top_k <= 0:
            raise ValueError("route_top_k must be positive.")
        if self.vmf_write_chunk_size <= 0:
            raise ValueError("vmf_write_chunk_size must be positive.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.tau_new <= 1.0:
            raise ValueError("tau_new must be in [0, 1].")
        if not 0.0 <= self.count_exponent < 1.0:
            raise ValueError("count_exponent must be in [0, 1).")
        if self.concentration_prior_mass < 0.0:
            raise ValueError("concentration_prior_mass cannot be negative.")
        if self.maximum_concentration <= 0.0:
            raise ValueError("maximum_concentration must be positive.")


def vmf_log_normalizer_high_dim_torch(concentration, dimension: int):
    """Uniform-asymptotic approximation of log C_d(kappa).

    Gemma's flattened KV index dimensions are 512 or 1024.  At those
    dimensions the Debye expansion is both stable and substantially cheaper
    than evaluating a different high-order Bessel function for every slot.
    The first correction term is retained.  Small test-only dimensions are
    rejected so callers cannot accidentally treat this as a universal
    approximation.
    """

    if dimension < 32:
        raise ValueError("The high-dimensional vMF approximation requires dimension >= 32.")
    torch, _functional = _require_torch()
    kappa = concentration.to(dtype=torch.float64)
    order = dimension / 2.0 - 1.0
    z = kappa / order
    root = torch.sqrt(1.0 + z * z)
    t = 1.0 / root
    correction = (3.0 * t - 5.0 * t * t * t) / (24.0 * order)
    log_c = (
        order * math.log(order)
        + order * torch.log1p(root)
        - order * root
        - dimension / 2.0 * math.log(2.0 * math.pi)
        + 0.5 * math.log(2.0 * math.pi * order)
        + 0.25 * torch.log1p(z * z)
        - correction
    )
    uniform = math.lgamma(dimension / 2.0) - math.log(2.0) - (
        dimension / 2.0
    ) * math.log(math.pi)
    return torch.where(kappa <= 1e-8, torch.full_like(log_c, uniform), log_c)


class TorchCamelotMemoryBank:
    """Fixed-capacity CAMELoT memory for one decoder layer."""

    def __init__(
        self,
        *,
        batch_size: int,
        kv_heads: int,
        head_dim: int,
        device,
        dtype,
        config: TokenMemoryConfig,
    ):
        torch, _functional = _require_torch()
        self.config = config
        self.batch_size = batch_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.flat_dim = kv_heads * head_dim
        # ``slot_capacity`` is a logical upper bound.  Backing tensors grow
        # with actual novel writes so a short run does not reserve every
        # layer's full 10k-slot budget up front.
        shape = (batch_size, 0)
        self.keys = torch.zeros(
            (*shape, self.flat_dim), device=device, dtype=dtype
        )
        self.values = torch.zeros(
            (*shape, kv_heads, head_dim), device=device, dtype=dtype
        )
        self.counts = torch.zeros(shape, device=device, dtype=torch.int64)
        self.ages = torch.zeros(shape, device=device, dtype=torch.int64)
        self.active = torch.zeros(shape, device=device, dtype=torch.bool)
        self.reads = 0
        self.writes = 0
        self.consolidations = 0
        self.replacements = 0
        self.assignment_searches = 0
        self._pending_assignment = None
        self._pending_source_pointer: int | None = None
        self._pending_source_shape: tuple[int, ...] | None = None

    @property
    def allocated_capacity(self) -> int:
        return self.keys.shape[1]

    def _ensure_capacity(self, required_capacity: int) -> None:
        torch, _functional = _require_torch()
        target = min(self.config.slot_capacity, required_capacity)
        current = self.allocated_capacity
        if target <= current:
            return
        added = target - current
        device = self.keys.device
        dtype = self.keys.dtype
        self.keys = torch.cat(
            (
                self.keys,
                torch.zeros(
                    (self.batch_size, added, self.flat_dim),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=1,
        )
        self.values = torch.cat(
            (
                self.values,
                torch.zeros(
                    (
                        self.batch_size,
                        added,
                        self.kv_heads,
                        self.head_dim,
                    ),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=1,
        )
        integer_extension = torch.zeros(
            (self.batch_size, added), device=device, dtype=torch.int64
        )
        self.counts = torch.cat((self.counts, integer_extension), dim=1)
        self.ages = torch.cat(
            (self.ages, torch.zeros_like(integer_extension)), dim=1
        )
        self.active = torch.cat(
            (
                self.active,
                torch.zeros(
                    (self.batch_size, added),
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )

    def _validate(self, keys, values) -> None:
        expected = (self.batch_size, self.kv_heads, self.head_dim)
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("K/V tensors must both have shape [batch, heads, tokens, dim].")
        if (keys.shape[0], keys.shape[1], keys.shape[3]) != expected:
            raise ValueError(
                f"K/V shape {tuple(keys.shape)} is incompatible with bank {expected}."
            )

    def _nearest_assignments(self, query_keys):
        torch, functional = _require_torch()
        batch, heads, tokens, dim = query_keys.shape
        self.assignment_searches += 1
        active_batches = self.active.any(dim=1)
        if not bool(active_batches.any().item()):
            indices = torch.zeros(
                (batch, tokens), device=query_keys.device, dtype=torch.int64
            )
            best_scores = torch.full(
                (batch, tokens),
                -torch.inf,
                device=query_keys.device,
                dtype=torch.float32,
            )
            valid = active_batches[:, None].expand(batch, tokens)
            return indices, best_scores, valid
        query = query_keys.transpose(1, 2).reshape(batch, tokens, self.flat_dim)
        query = functional.normalize(query.float(), dim=-1, eps=1e-12)
        slot_keys = functional.normalize(self.keys.float(), dim=-1, eps=1e-12)
        scores = torch.matmul(query, slot_keys.transpose(1, 2))
        scores = scores.masked_fill(~self.active[:, None, :], -torch.inf)
        valid = active_batches[:, None].expand(batch, tokens)
        indices = scores.argmax(dim=-1)
        best_scores = torch.gather(scores, 2, indices[..., None]).squeeze(-1)
        return indices, best_scores, valid

    def retrieve(self, query_keys):
        torch, _functional = _require_torch()
        if query_keys.ndim != 4:
            raise ValueError("query_keys must have shape [batch, heads, tokens, dim].")
        batch, heads, tokens, dim = query_keys.shape
        if (batch, heads, dim) != (
            self.batch_size,
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("Query shape is incompatible with this memory bank.")
        if tokens == 0:
            empty = query_keys.new_empty((batch, heads, 0, dim))
            return empty, empty, torch.empty((batch, 0), device=query_keys.device, dtype=torch.bool)

        indices, best_scores, valid = self._nearest_assignments(query_keys)
        self._pending_assignment = (indices, best_scores, valid)
        self._pending_source_pointer = query_keys.data_ptr()
        self._pending_source_shape = tuple(query_keys.shape)
        if self.allocated_capacity == 0:
            empty_memory = torch.zeros_like(query_keys)
            self.reads += int(valid.sum().item())
            return empty_memory, empty_memory, valid
        key_indices = indices[..., None].expand(batch, tokens, self.flat_dim)
        selected_keys = torch.gather(self.keys, 1, key_indices)
        value_indices = indices[..., None, None].expand(
            batch, tokens, self.kv_heads, self.head_dim
        )
        selected_values = torch.gather(self.values, 1, value_indices)
        selected_keys = selected_keys.view(
            batch, tokens, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        selected_values = selected_values.transpose(1, 2)
        self.reads += int(valid.sum().item())
        return selected_keys, selected_values, valid

    def write(self, keys, values) -> None:
        torch, _functional = _require_torch()
        self._validate(keys, values)
        batch, _heads, tokens, _dim = keys.shape
        flat_keys = keys.transpose(1, 2).reshape(batch, tokens, self.flat_dim)
        flat_values = values.transpose(1, 2).reshape(
            batch, tokens, self.flat_dim
        )
        cached_assignment_matches = (
            self._pending_assignment is not None
            and self._pending_source_pointer == keys.data_ptr()
            and self._pending_source_shape == tuple(keys.shape)
        )
        if cached_assignment_matches:
            indices, best_scores, valid = self._pending_assignment
        else:
            indices, best_scores, valid = self._nearest_assignments(keys)
        self._pending_assignment = None
        self._pending_source_pointer = None
        self._pending_source_shape = None
        familiar = valid & (best_scores > self.config.camelot_threshold)
        maximum_novel = int((~familiar).sum(dim=1).max().item())
        self._ensure_capacity(self.allocated_capacity + maximum_novel)
        touched = torch.zeros_like(self.active)
        with torch.no_grad():
            for batch_index in range(batch):
                familiar_positions = torch.nonzero(
                    familiar[batch_index], as_tuple=False
                ).flatten()
                if familiar_positions.numel():
                    assigned_slots = indices[
                        batch_index, familiar_positions
                    ]
                    unique_slots, inverse = torch.unique(
                        assigned_slots,
                        sorted=True,
                        return_inverse=True,
                    )
                    slot_count = unique_slots.numel()
                    key_sums = torch.zeros(
                        (slot_count, self.flat_dim),
                        device=keys.device,
                        dtype=torch.float32,
                    )
                    value_sums = torch.zeros_like(key_sums)
                    key_sums.index_add_(
                        0,
                        inverse,
                        flat_keys[
                            batch_index, familiar_positions
                        ].float(),
                    )
                    value_sums.index_add_(
                        0,
                        inverse,
                        flat_values[
                            batch_index, familiar_positions
                        ].float(),
                    )
                    additions = torch.bincount(
                        inverse, minlength=slot_count
                    )
                    old_counts = self.counts[
                        batch_index, unique_slots
                    ]
                    new_counts = old_counts + additions
                    old_mass = old_counts.float()[:, None]
                    denominator = new_counts.float()[:, None]
                    updated_keys = (
                        self.keys[
                            batch_index, unique_slots
                        ].float()
                        * old_mass
                        + key_sums
                    ) / denominator
                    old_values = self.values[
                        batch_index, unique_slots
                    ].reshape(slot_count, self.flat_dim)
                    updated_values = (
                        old_values.float() * old_mass + value_sums
                    ) / denominator
                    self.keys[batch_index].index_copy_(
                        0,
                        unique_slots,
                        updated_keys.to(self.keys.dtype),
                    )
                    self.values[batch_index].index_copy_(
                        0,
                        unique_slots,
                        updated_values.to(self.values.dtype).reshape(
                            slot_count, self.kv_heads, self.head_dim
                        ),
                    )
                    self.counts[
                        batch_index, unique_slots
                    ] = new_counts
                    touched[batch_index, unique_slots] = True

                novel_positions = torch.nonzero(
                    ~familiar[batch_index], as_tuple=False
                ).flatten()
                offset = 0
                while offset < novel_positions.numel():
                    free_ids = torch.nonzero(
                        ~self.active[batch_index], as_tuple=False
                    ).flatten()
                    old_ids = torch.nonzero(
                        self.active[batch_index]
                        & ~touched[batch_index],
                        as_tuple=False,
                    ).flatten()
                    if old_ids.numel():
                        order = torch.argsort(
                            self.ages[batch_index, old_ids],
                            descending=True,
                            stable=True,
                        )
                        old_ids = old_ids[order]
                    recently_touched = torch.nonzero(
                        self.active[batch_index]
                        & touched[batch_index],
                        as_tuple=False,
                    ).flatten()
                    allocation_order = torch.cat(
                        (free_ids, old_ids, recently_touched)
                    )
                    chunk_size = min(
                        int(allocation_order.numel()),
                        int(novel_positions.numel()) - offset,
                    )
                    source_positions = novel_positions[
                        offset : offset + chunk_size
                    ]
                    target_slots = allocation_order[:chunk_size]
                    was_active = self.active[
                        batch_index, target_slots
                    ].clone()
                    self.keys[batch_index].index_copy_(
                        0,
                        target_slots,
                        flat_keys[batch_index, source_positions],
                    )
                    self.values[batch_index].index_copy_(
                        0,
                        target_slots,
                        flat_values[
                            batch_index, source_positions
                        ].reshape(
                            chunk_size, self.kv_heads, self.head_dim
                        ),
                    )
                    self.counts[
                        batch_index, target_slots
                    ] = 1
                    self.active[
                        batch_index, target_slots
                    ] = True
                    touched[batch_index, target_slots] = True
                    self.ages[batch_index, target_slots] = 0
                    self.replacements += int(was_active.sum().item())
                    offset += chunk_size
            self.consolidations += int(familiar.sum().item())
            self.writes += batch * tokens
            self.ages[self.active & ~touched] += 1
            self.ages[touched] = 0

    @property
    def memory_bytes(self) -> int:
        tensors = (self.keys, self.values, self.counts, self.ages, self.active)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "camelot",
            "active_slots": int(self.active.sum().item()),
            "allocated_slots_per_stream": self.allocated_capacity,
            "maximum_slots_per_stream": self.config.slot_capacity,
            "represented_writes": int(self.counts.sum().item()),
            "reads": self.reads,
            "writes": self.writes,
            "consolidations": self.consolidations,
            "replacements": self.replacements,
            "assignment_searches": self.assignment_searches,
            "memory_bytes": self.memory_bytes,
        }


class TorchVMFRecordMemoryBank:
    """vMF routing slots whose payloads remain original per-write K/V records."""

    def __init__(
        self,
        *,
        batch_size: int,
        kv_heads: int,
        head_dim: int,
        device,
        dtype,
        config: TokenMemoryConfig,
    ):
        torch, _functional = _require_torch()
        self.config = config
        self.batch_size = batch_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.flat_dim = kv_heads * head_dim
        # Capacities are logical limits.  Physical tensors grow with the
        # number of slots/records actually needed by sequential writes.
        slot_shape = (batch_size, 0)
        record_shape = (batch_size, 0)
        self.slot_resultants = torch.zeros(
            (*slot_shape, self.flat_dim), device=device, dtype=torch.float32
        )
        self.slot_norms = torch.zeros(
            slot_shape, device=device, dtype=torch.float32
        )
        self.slot_kappas = torch.zeros(
            slot_shape, device=device, dtype=torch.float64
        )
        self.slot_log_bases = torch.full(
            slot_shape, -torch.inf, device=device, dtype=torch.float64
        )
        self.slot_counts = torch.zeros(slot_shape, device=device, dtype=torch.int64)
        self.slot_ages = torch.zeros(slot_shape, device=device, dtype=torch.int64)
        self.slot_active = torch.zeros(slot_shape, device=device, dtype=torch.bool)
        self.record_keys = torch.zeros(
            (*record_shape, self.flat_dim), device=device, dtype=dtype
        )
        self.record_values = torch.zeros(
            (*record_shape, kv_heads, head_dim), device=device, dtype=dtype
        )
        self.record_slots = torch.full(
            record_shape, -1, device=device, dtype=torch.int64
        )
        self.record_ages = torch.zeros(record_shape, device=device, dtype=torch.int64)
        self.record_active = torch.zeros(record_shape, device=device, dtype=torch.bool)
        self.reads = 0
        self.writes = 0
        self.created_slots = 0
        self.assigned_existing = 0
        self.evicted_records = 0
        self.evicted_slots = 0
        self.posterior_new_sum = 0.0

    @property
    def allocated_slot_capacity(self) -> int:
        return self.slot_resultants.shape[1]

    @property
    def allocated_record_capacity(self) -> int:
        return self.record_keys.shape[1]

    def _ensure_slot_capacity(self, required_capacity: int) -> None:
        torch, _functional = _require_torch()
        target = min(self.config.slot_capacity, required_capacity)
        current = self.allocated_slot_capacity
        if target <= current:
            return
        added = target - current
        device = self.slot_resultants.device
        self.slot_resultants = torch.cat(
            (
                self.slot_resultants,
                torch.zeros(
                    (self.batch_size, added, self.flat_dim),
                    device=device,
                    dtype=torch.float32,
                ),
            ),
            dim=1,
        )
        self.slot_norms = torch.cat(
            (
                self.slot_norms,
                torch.zeros(
                    (self.batch_size, added),
                    device=device,
                    dtype=torch.float32,
                ),
            ),
            dim=1,
        )
        self.slot_kappas = torch.cat(
            (
                self.slot_kappas,
                torch.zeros(
                    (self.batch_size, added),
                    device=device,
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
        self.slot_log_bases = torch.cat(
            (
                self.slot_log_bases,
                torch.full(
                    (self.batch_size, added),
                    -torch.inf,
                    device=device,
                    dtype=torch.float64,
                ),
            ),
            dim=1,
        )
        integer_extension = torch.zeros(
            (self.batch_size, added), device=device, dtype=torch.int64
        )
        self.slot_counts = torch.cat(
            (self.slot_counts, integer_extension), dim=1
        )
        self.slot_ages = torch.cat(
            (self.slot_ages, torch.zeros_like(integer_extension)), dim=1
        )
        self.slot_active = torch.cat(
            (
                self.slot_active,
                torch.zeros(
                    (self.batch_size, added),
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )

    def _ensure_record_capacity(self, required_capacity: int) -> None:
        torch, _functional = _require_torch()
        target = min(self.config.record_capacity, required_capacity)
        current = self.allocated_record_capacity
        if target <= current:
            return
        added = target - current
        device = self.record_keys.device
        dtype = self.record_keys.dtype
        self.record_keys = torch.cat(
            (
                self.record_keys,
                torch.zeros(
                    (self.batch_size, added, self.flat_dim),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=1,
        )
        self.record_values = torch.cat(
            (
                self.record_values,
                torch.zeros(
                    (
                        self.batch_size,
                        added,
                        self.kv_heads,
                        self.head_dim,
                    ),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=1,
        )
        self.record_slots = torch.cat(
            (
                self.record_slots,
                torch.full(
                    (self.batch_size, added),
                    -1,
                    device=device,
                    dtype=torch.int64,
                ),
            ),
            dim=1,
        )
        integer_extension = torch.zeros(
            (self.batch_size, added), device=device, dtype=torch.int64
        )
        self.record_ages = torch.cat(
            (self.record_ages, integer_extension), dim=1
        )
        self.record_active = torch.cat(
            (
                self.record_active,
                torch.zeros(
                    (self.batch_size, added),
                    device=device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )

    def _refresh_slots(self, batch_ids, slot_ids) -> None:
        torch, _functional = _require_torch()
        resultants = self.slot_resultants[batch_ids, slot_ids]
        counts = self.slot_counts[batch_ids, slot_ids]
        norms64 = torch.linalg.vector_norm(
            resultants.to(torch.float64), dim=-1
        )
        counts64 = counts.to(torch.float64)
        rbar = norms64 / (
            counts64 + self.config.concentration_prior_mass
        )
        rbar = rbar.clamp(0.0, 1.0 - 1e-9)
        kappa = (
            rbar * (self.flat_dim - rbar * rbar)
            / (1.0 - rbar * rbar)
        ).clamp(0.0, self.config.maximum_concentration)
        log_base = (
            self.config.count_exponent
            * torch.log(counts64 + 1e-6)
            + vmf_log_normalizer_high_dim_torch(
                kappa, self.flat_dim
            )
        )
        active = counts > 0
        self.slot_norms[batch_ids, slot_ids] = norms64.to(
            torch.float32
        )
        self.slot_kappas[batch_ids, slot_ids] = torch.where(
            active, kappa, torch.zeros_like(kappa)
        )
        self.slot_log_bases[batch_ids, slot_ids] = torch.where(
            active,
            log_base,
            torch.full_like(log_base, -torch.inf),
        )

    def _posterior_batch(
        self,
        candidate_ids,
        candidate_valid,
        candidate_cosine,
    ):
        torch, _functional = _require_torch()
        batch_ids = torch.arange(
            self.batch_size, device=candidate_ids.device
        )[:, None, None]
        kappa = self.slot_kappas[batch_ids, candidate_ids]
        log_existing = (
            self.slot_log_bases[batch_ids, candidate_ids]
            + kappa * candidate_cosine.to(torch.float64)
        )
        log_existing = log_existing.masked_fill(
            ~candidate_valid, -torch.inf
        )
        uniform = math.lgamma(self.flat_dim / 2.0) - math.log(2.0) - (
            self.flat_dim / 2.0
        ) * math.log(math.pi)
        log_new = torch.full(
            (*candidate_ids.shape[:-1], 1),
            math.log(self.config.alpha) + uniform,
            device=candidate_ids.device,
            dtype=torch.float64,
        )
        logits = torch.cat(
            (log_new, log_existing),
            dim=-1,
        )
        probabilities = torch.softmax(logits, dim=-1)
        probability_new = probabilities[..., 0]
        best_positions = probabilities[..., 1:].argmax(dim=-1)
        selected = candidate_ids.gather(
            -1, best_positions[..., None]
        ).squeeze(-1)
        has_existing = candidate_valid.any(dim=-1)
        return probability_new, selected, has_existing

    def retrieve(self, query_keys):
        torch, functional = _require_torch()
        if query_keys.ndim != 4:
            raise ValueError("query_keys must have shape [batch, heads, tokens, dim].")
        batch, heads, tokens, dim = query_keys.shape
        if (batch, heads, dim) != (
            self.batch_size,
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("Query shape is incompatible with this memory bank.")
        flat_queries = query_keys.transpose(1, 2).reshape(batch, tokens, self.flat_dim)
        selected_keys = torch.zeros_like(flat_queries)
        selected_values = query_keys.new_zeros(
            (batch, tokens, self.kv_heads, self.head_dim)
        )
        valid = torch.zeros((batch, tokens), device=query_keys.device, dtype=torch.bool)
        if (
            tokens == 0
            or self.allocated_slot_capacity == 0
            or self.allocated_record_capacity == 0
        ):
            return (
                selected_keys.view(
                    batch, tokens, heads, dim
                ).transpose(1, 2),
                selected_values.transpose(1, 2),
                valid,
            )
        with torch.no_grad():
            query_directions = functional.normalize(
                flat_queries.float(), dim=-1, eps=1e-12
            )
            centroid_scores = torch.matmul(
                query_directions,
                self.slot_resultants.transpose(1, 2),
            )
            centroid_scores = centroid_scores / self.slot_norms[
                :, None, :
            ].clamp_min(1e-12)
            centroid_scores = centroid_scores.masked_fill(
                ~self.slot_active[:, None, :], -torch.inf
            )
            route_width = min(
                self.config.route_top_k, self.allocated_slot_capacity
            )
            routed_slots = centroid_scores.topk(
                route_width, dim=-1
            ).indices
            routed_valid = self.slot_active.gather(
                1, routed_slots.reshape(batch, -1)
            ).reshape(batch, tokens, route_width)
            route_mask = (
                self.record_slots[:, None, :, None]
                == routed_slots[:, :, None, :]
            ) & routed_valid[:, :, None, :]
            route_mask = route_mask.any(dim=-1)
            route_mask &= self.record_active[:, None, :]
            raw_scores = torch.matmul(
                flat_queries.float(),
                self.record_keys.float().transpose(1, 2),
            ) / math.sqrt(self.flat_dim)
            raw_scores = raw_scores.masked_fill(~route_mask, -torch.inf)
            best_records = raw_scores.argmax(dim=-1)
            key_indices = best_records[..., None].expand(
                batch, tokens, self.flat_dim
            )
            selected_keys = torch.gather(
                self.record_keys, 1, key_indices
            )
            value_indices = best_records[..., None, None].expand(
                batch, tokens, self.kv_heads, self.head_dim
            )
            selected_values = torch.gather(
                self.record_values, 1, value_indices
            )
            valid_batches = (
                self.slot_active.any(dim=1)
                & self.record_active.any(dim=1)
            )
            valid = valid_batches[:, None].expand(batch, tokens)
            selected_keys = selected_keys.masked_fill(
                ~valid[..., None], 0
            )
            selected_values = selected_values.masked_fill(
                ~valid[..., None, None], 0
            )
            self.reads += int(valid.sum().item())
        return (
            selected_keys.view(batch, tokens, heads, dim).transpose(1, 2),
            selected_values.transpose(1, 2),
            valid,
        )

    def write(self, keys, values) -> None:
        torch, functional = _require_torch()
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("K/V tensors must both have shape [batch, heads, tokens, dim].")
        batch, heads, tokens, dim = keys.shape
        if (batch, heads, dim) != (
            self.batch_size,
            self.kv_heads,
            self.head_dim,
        ):
            raise ValueError("K/V shape is incompatible with this memory bank.")
        flat_keys = keys.transpose(1, 2).reshape(batch, tokens, self.flat_dim)
        flat_values = values.transpose(1, 2)
        if tokens == 0:
            return
        self._ensure_slot_capacity(
            self.allocated_slot_capacity + tokens
        )
        self._ensure_record_capacity(
            self.allocated_record_capacity + tokens
        )
        touched_slots = torch.zeros_like(self.slot_active)
        touched_records = torch.zeros_like(self.record_active)
        batch_ids = torch.arange(batch, device=keys.device)
        created_count = torch.zeros((), device=keys.device, dtype=torch.int64)
        assigned_count = torch.zeros_like(created_count)
        evicted_record_count = torch.zeros_like(created_count)
        evicted_slot_count = torch.zeros_like(created_count)
        posterior_sum = torch.zeros(
            (), device=keys.device, dtype=torch.float64
        )
        slot_count = self.allocated_slot_capacity
        record_count = self.allocated_record_capacity
        slot_offsets = batch_ids * slot_count
        record_offsets = batch_ids * record_count
        chunk_size = min(
            self.config.vmf_write_chunk_size,
            self.config.slot_capacity,
            self.config.record_capacity,
        )
        with torch.no_grad():
            route_width = min(
                self.config.route_top_k, slot_count
            )
            for chunk_start in range(0, tokens, chunk_size):
                chunk_end = min(tokens, chunk_start + chunk_size)
                current_size = chunk_end - chunk_start
                chunk_keys = flat_keys[:, chunk_start:chunk_end]
                chunk_values = flat_values[:, chunk_start:chunk_end]
                direction = functional.normalize(
                    chunk_keys.float(),
                    dim=-1,
                    eps=1e-12,
                )
                similarities = torch.matmul(
                    direction,
                    self.slot_resultants.transpose(1, 2),
                )
                similarities = similarities / self.slot_norms[
                    :, None, :
                ].clamp_min(1e-12)
                similarities = similarities.masked_fill(
                    ~self.slot_active[:, None, :], -torch.inf
                )
                candidate_ids = similarities.topk(
                    route_width, dim=-1
                ).indices
                candidate_valid = self.slot_active[:, None, :].expand(
                    -1, current_size, -1
                ).gather(2, candidate_ids)
                candidate_cosine = similarities.gather(
                    2, candidate_ids
                )
                probability_new, selected, has_existing = (
                    self._posterior_batch(
                        candidate_ids,
                        candidate_valid,
                        candidate_cosine,
                    )
                )
                create_new = (
                    ~has_existing
                    | (probability_new > self.config.tau_new)
                )

                slot_category = torch.where(
                    ~self.slot_active,
                    torch.zeros_like(self.slot_ages),
                    torch.where(
                        ~touched_slots,
                        torch.ones_like(self.slot_ages),
                        torch.full_like(self.slot_ages, 2),
                    ),
                )
                slot_age_order = torch.argsort(
                    self.slot_ages,
                    dim=1,
                    descending=True,
                    stable=True,
                )
                ordered_slot_category = slot_category.gather(
                    1, slot_age_order
                )
                category_order = torch.argsort(
                    ordered_slot_category,
                    dim=1,
                    stable=True,
                )
                slot_allocation_order = slot_age_order.gather(
                    1, category_order
                )
                new_rank = create_new.to(torch.int64).cumsum(
                    dim=1
                ) - 1
                allocated_slots = slot_allocation_order.gather(
                    1, new_rank.clamp_min(0)
                )
                selected = torch.where(
                    create_new, allocated_slots, selected
                )
                new_slot_hits = torch.zeros_like(
                    self.slot_counts
                ).scatter_add_(
                    1, selected, create_new.to(torch.int64)
                )
                new_slot_mask = new_slot_hits > 0
                evicted_slot_mask = new_slot_mask & self.slot_active
                safe_record_slots = self.record_slots.clamp_min(0)
                slot_members = self.record_active & evicted_slot_mask.gather(
                    1, safe_record_slots
                )
                evicted_record_count += slot_members.sum()
                self.record_active.masked_fill_(slot_members, False)
                self.record_slots.masked_fill_(slot_members, -1)
                self.slot_resultants.masked_fill_(
                    new_slot_mask[..., None], 0
                )
                self.slot_counts.masked_fill_(new_slot_mask, 0)
                self.slot_active.masked_fill_(new_slot_mask, False)
                created_count += create_new.sum()
                assigned_count += (~create_new).sum()
                evicted_slot_count += evicted_slot_mask.sum()

                record_category = torch.where(
                    ~self.record_active,
                    torch.zeros_like(self.record_ages),
                    torch.where(
                        ~touched_records,
                        torch.ones_like(self.record_ages),
                        torch.full_like(self.record_ages, 2),
                    ),
                )
                record_age_order = torch.argsort(
                    self.record_ages,
                    dim=1,
                    descending=True,
                    stable=True,
                )
                ordered_record_category = record_category.gather(
                    1, record_age_order
                )
                record_category_order = torch.argsort(
                    ordered_record_category,
                    dim=1,
                    stable=True,
                )
                record_allocation_order = record_age_order.gather(
                    1, record_category_order
                )
                record_ids = record_allocation_order[:, :current_size]
                evict_record = self.record_active.gather(
                    1, record_ids
                )
                old_slots = self.record_slots.gather(
                    1, record_ids
                ).clamp_min(0)
                record_key_indices = record_ids[..., None].expand(
                    batch, current_size, self.flat_dim
                )
                old_directions = functional.normalize(
                    torch.gather(
                        self.record_keys, 1, record_key_indices
                    ).float(),
                    dim=-1,
                    eps=1e-12,
                )
                global_old_slots = (
                    slot_offsets[:, None] + old_slots
                ).reshape(-1)
                flat_resultants = self.slot_resultants.view(
                    batch * slot_count, self.flat_dim
                )
                flat_counts = self.slot_counts.view(-1)
                flat_resultants.index_add_(
                    0,
                    global_old_slots,
                    (
                        -old_directions
                        * evict_record[..., None]
                    ).reshape(-1, self.flat_dim),
                )
                flat_counts.index_add_(
                    0,
                    global_old_slots,
                    (-evict_record.to(torch.int64)).reshape(-1),
                )
                empty_slots = self.slot_counts <= 0
                self.slot_resultants.masked_fill_(
                    empty_slots[..., None], 0
                )
                self.slot_counts.masked_fill_(empty_slots, 0)
                self.slot_active &= ~empty_slots
                evicted_record_count += evict_record.sum()

                global_record_ids = (
                    record_offsets[:, None] + record_ids
                ).reshape(-1)
                self.record_keys.view(
                    batch * record_count, self.flat_dim
                ).index_copy_(
                    0,
                    global_record_ids,
                    chunk_keys.reshape(-1, self.flat_dim),
                )
                self.record_values.view(
                    batch * record_count,
                    self.kv_heads,
                    self.head_dim,
                ).index_copy_(
                    0,
                    global_record_ids,
                    chunk_values.reshape(
                        -1, self.kv_heads, self.head_dim
                    ),
                )
                self.record_slots.view(-1).index_copy_(
                    0, global_record_ids, selected.reshape(-1)
                )
                self.record_active.view(-1)[global_record_ids] = True

                global_selected = (
                    slot_offsets[:, None] + selected
                ).reshape(-1)
                flat_resultants.index_add_(
                    0,
                    global_selected,
                    direction.reshape(-1, self.flat_dim),
                )
                flat_counts.index_add_(
                    0,
                    global_selected,
                    torch.ones(
                        batch * current_size,
                        device=keys.device,
                        dtype=torch.int64,
                    ),
                )
                self.slot_active.view(-1)[global_selected] = True
                touched_slots.scatter_(1, selected, True)
                touched_records.scatter_(1, record_ids, True)
                affected_slots = torch.unique(
                    torch.cat(
                        (global_selected, global_old_slots), dim=0
                    )
                )
                self._refresh_slots(
                    affected_slots // slot_count,
                    affected_slots % slot_count,
                )
                posterior_sum += probability_new.sum()

            self.slot_ages[self.slot_active & ~touched_slots] += 1
            self.slot_ages[touched_slots] = 0
            self.record_ages[self.record_active & ~touched_records] += 1
            self.record_ages[touched_records] = 0
            statistics = torch.stack(
                (
                    created_count,
                    assigned_count,
                    evicted_record_count,
                    evicted_slot_count,
                )
            ).tolist()
            self.created_slots += int(statistics[0])
            self.assigned_existing += int(statistics[1])
            self.evicted_records += int(statistics[2])
            self.evicted_slots += int(statistics[3])
            self.posterior_new_sum += float(posterior_sum.item())
            self.writes += batch * tokens

    @property
    def memory_bytes(self) -> int:
        tensors = (
            self.slot_resultants,
            self.slot_norms,
            self.slot_kappas,
            self.slot_log_bases,
            self.slot_counts,
            self.slot_ages,
            self.slot_active,
            self.record_keys,
            self.record_values,
            self.record_slots,
            self.record_ages,
            self.record_active,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": "vmf_records",
            "active_slots": int(self.slot_active.sum().item()),
            "active_records": int(self.record_active.sum().item()),
            "allocated_slots_per_stream": self.allocated_slot_capacity,
            "maximum_slots_per_stream": self.config.slot_capacity,
            "allocated_records_per_stream": self.allocated_record_capacity,
            "maximum_records_per_stream": self.config.record_capacity,
            "reads": self.reads,
            "writes": self.writes,
            "created_slots": self.created_slots,
            "assigned_existing": self.assigned_existing,
            "evicted_records": self.evicted_records,
            "evicted_slots": self.evicted_slots,
            "mean_probability_new": (
                self.posterior_new_sum / self.writes if self.writes else 0.0
            ),
            "memory_bytes": self.memory_bytes,
        }


def create_token_memory_bank(
    *,
    batch_size: int,
    kv_heads: int,
    head_dim: int,
    device,
    dtype,
    config: TokenMemoryConfig,
):
    kwargs = {
        "batch_size": batch_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "device": device,
        "dtype": dtype,
        "config": config,
    }
    if config.method == "camelot":
        return TorchCamelotMemoryBank(**kwargs)
    return TorchVMFRecordMemoryBank(**kwargs)


def token_memory_config_dict(config: TokenMemoryConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "TokenMemoryConfig",
    "TorchCamelotMemoryBank",
    "TorchVMFRecordMemoryBank",
    "create_token_memory_bank",
    "token_memory_config_dict",
    "vmf_log_normalizer_high_dim_torch",
]
