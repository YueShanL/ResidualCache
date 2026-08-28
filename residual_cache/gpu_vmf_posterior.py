"""Shared bounded-candidate vMF posterior mathematics for GPU memories.

This module deliberately knows nothing about token records, block records, K/V
payload layout, learned-router ownership, replay, or eviction.  Independent
memory implementations share only this classification mechanism.
"""

from __future__ import annotations

from collections import OrderedDict
import math

from .torch_token_memory import vmf_log_normalizer_high_dim_torch


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - project runtime includes torch.
        raise RuntimeError("GPU vMF posterior classification requires torch") from error
    return torch


class CpuLocalityIndex:
    """Bounded bucket lookup over active slots; never iterates all slots."""

    def __init__(self, *, bits: int, probe_radius: int, candidate_capacity: int):
        self.bits = int(bits)
        self.probe_radius = int(probe_radius)
        self.candidate_capacity = int(candidate_capacity)
        self._buckets: dict[int, OrderedDict[int, None]] = {}
        self._slot_codes: dict[int, int] = {}

    def _probes(self, code: int):
        yield code
        if self.probe_radius == 1:
            for bit in range(self.bits):
                yield code ^ (1 << bit)

    def remove(self, slot: int) -> None:
        code = self._slot_codes.pop(int(slot), None)
        if code is None:
            return
        bucket = self._buckets.get(code)
        if bucket is None:
            return
        bucket.pop(int(slot), None)
        if not bucket:
            self._buckets.pop(code, None)

    def update(self, slot: int, code: int) -> None:
        slot = int(slot)
        code = int(code)
        previous = self._slot_codes.get(slot)
        if previous != code:
            self.remove(slot)
        bucket = self._buckets.setdefault(code, OrderedDict())
        bucket.pop(slot, None)
        bucket[slot] = None
        self._slot_codes[slot] = code

    def candidates(self, code: int) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        for probe in self._probes(int(code)):
            bucket = self._buckets.get(probe)
            if not bucket:
                continue
            for slot in reversed(bucket):
                if slot in seen:
                    continue
                result.append(slot)
                seen.add(slot)
                if len(result) >= self.candidate_capacity:
                    return tuple(result)
        return tuple(result)


def locality_codes(directions, projection) -> list[int]:
    """Return compact random-hyperplane codes for normalized directions."""

    torch = _require_torch()
    signs = (directions @ projection.transpose(0, 1)) >= 0
    powers = 1 << torch.arange(
        projection.shape[0], device=directions.device, dtype=torch.int64
    )
    return (signs.to(torch.int64) * powers).sum(dim=-1).cpu().tolist()


def vmf_slot_parameters(
    resultants,
    masses,
    *,
    dimension: int,
    concentration_prior_mass: float,
    maximum_concentration: float,
    count_exponent: float,
):
    """Compute vMF concentration/log-base tensors for selected slot state."""

    torch = _require_torch()
    mass64 = masses.to(torch.float64)
    norms = torch.linalg.vector_norm(resultants.to(torch.float64), dim=-1)
    rbar = (norms / (mass64 + float(concentration_prior_mass))).clamp(
        0.0, 1.0 - 1e-9
    )
    kappa = (
        rbar * (int(dimension) - rbar.square()) / (1.0 - rbar.square())
    ).clamp(0.0, float(maximum_concentration))
    log_base = (
        float(count_exponent) * torch.log(mass64 + 1e-6)
        + vmf_log_normalizer_high_dim_torch(kappa, int(dimension))
    )
    active = masses > 0
    return (
        torch.where(active, kappa, torch.zeros_like(kappa)),
        torch.where(active, log_base, torch.full_like(log_base, -torch.inf)),
        active,
    )


def vmf_posterior_assignments(
    directions,
    candidate_ids,
    candidate_valid,
    *,
    slot_resultants,
    slot_kappas,
    slot_log_bases,
    dimension: int,
    alpha: float,
    tau_new: float,
):
    """Score existing candidates plus the explicit new-cluster hypothesis."""

    torch = _require_torch()
    resultants = slot_resultants[candidate_ids]
    norms = torch.linalg.vector_norm(resultants, dim=-1).clamp_min(1e-12)
    cosine = (resultants * directions[:, None, :]).sum(dim=-1) / norms
    log_existing = (
        slot_log_bases[candidate_ids]
        + slot_kappas[candidate_ids] * cosine.to(torch.float64)
    ).masked_fill(~candidate_valid, -torch.inf)
    uniform = math.lgamma(int(dimension) / 2.0) - math.log(2.0) - (
        int(dimension) / 2.0
    ) * math.log(math.pi)
    log_new = torch.full(
        (directions.shape[0], 1),
        math.log(float(alpha)) + uniform,
        device=directions.device,
        dtype=torch.float64,
    )
    probabilities = torch.softmax(torch.cat((log_new, log_existing), dim=1), dim=1)
    has_existing = candidate_valid.any(dim=1)
    best = probabilities[:, 1:].argmax(dim=1)
    selected = candidate_ids.gather(1, best[:, None]).squeeze(1)
    create_new = ~has_existing | (probabilities[:, 0] > float(tau_new))
    return selected, create_new


__all__ = [
    "CpuLocalityIndex",
    "locality_codes",
    "vmf_posterior_assignments",
    "vmf_slot_parameters",
]
