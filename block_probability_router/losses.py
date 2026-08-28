from __future__ import annotations

from dataclasses import dataclass

import torch

from learnable_index.data import RetrievalBatch

from .config import ProbabilityLossConfig
from .model import ProbabilityRouterOutput


@dataclass(frozen=True)
class ProbabilityLossBreakdown:
    total: torch.Tensor
    conditional: torch.Tensor
    conditional_sample_count: int


def probability_router_loss(
    output: ProbabilityRouterOutput,
    batch: RetrievalBatch,
    config: ProbabilityLossConfig,
) -> ProbabilityLossBreakdown:
    if output.probabilities.shape != batch.candidate_mask.shape:
        raise ValueError("router probabilities must align with the candidate mask")

    target = batch.conditional_teacher_distribution.masked_fill(~batch.candidate_mask, 0.0)
    target_sums = target.sum(dim=-1)
    eligible = batch.total_teacher_historical_mass > config.minimum_historical_mass
    if torch.any(eligible & ~torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-4)):
        raise ValueError("positive-historical-mass conditional targets must sum to one")

    safe_log_probabilities = output.log_probabilities.masked_fill(~batch.candidate_mask, 0.0)
    per_sample = -(target * safe_log_probabilities).sum(dim=-1)
    conditional = per_sample[eligible].mean() if torch.any(eligible) else output.weights.sum() * 0.0
    return ProbabilityLossBreakdown(
        total=conditional,
        conditional=conditional,
        conditional_sample_count=int(eligible.sum().item()),
    )
