from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import LossConfig
from .data import RetrievalBatch
from .model import RouterOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    conditional: torch.Tensor
    demand: torch.Tensor
    conditional_sample_count: int


def router_loss(
    output: RouterOutput,
    batch: RetrievalBatch,
    config: LossConfig,
) -> LossBreakdown:
    if output.scores.shape != batch.candidate_mask.shape:
        raise ValueError("router scores must align with the candidate mask")
    if output.demand_logits.shape != batch.total_teacher_historical_mass.shape:
        raise ValueError("demand logits must contain one value per sample")

    target = batch.conditional_teacher_distribution.masked_fill(~batch.candidate_mask, 0.0)
    target_sums = target.sum(dim=-1)
    eligible = batch.total_teacher_historical_mass > config.minimum_historical_mass
    if torch.any(eligible & ~torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-4)):
        raise ValueError("positive-demand conditional targets must sum to one")

    log_probabilities = F.log_softmax(output.scores, dim=-1)
    per_sample_conditional = -(target * log_probabilities).sum(dim=-1)
    if torch.any(eligible):
        conditional = per_sample_conditional[eligible].mean()
    else:
        conditional = output.scores.sum() * 0.0

    demand_target = batch.total_teacher_historical_mass
    if config.demand_loss == "bce":
        if torch.any((demand_target < 0) | (demand_target > 1 + 1e-5)):
            raise ValueError(
                "demand BCE requires total historical mass in [0, 1]; "
                "use demand_loss='mse' for summed future attention"
            )
        demand = F.binary_cross_entropy_with_logits(
            output.demand_logits, demand_target.clamp(0, 1)
        )
    else:
        demand = F.mse_loss(F.softplus(output.demand_logits), demand_target)
    total = config.conditional_weight * conditional + config.demand_weight * demand
    return LossBreakdown(
        total=total,
        conditional=conditional,
        demand=demand,
        conditional_sample_count=int(eligible.sum().item()),
    )
