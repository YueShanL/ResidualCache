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
    conditional_sample_count: int


def router_loss(
    output: RouterOutput,
    batch: RetrievalBatch,
    config: LossConfig,
) -> LossBreakdown:
    if output.scores.shape != batch.candidate_mask.shape:
        raise ValueError("router scores must align with the candidate mask")

    target = batch.conditional_teacher_distribution.masked_fill(~batch.candidate_mask, 0.0)
    target_sums = target.sum(dim=-1)
    eligible = batch.total_teacher_historical_mass > config.minimum_historical_mass
    if torch.any(eligible & ~torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-4)):
        raise ValueError("positive-historical-mass conditional targets must sum to one")

    log_probabilities = F.log_softmax(output.scores, dim=-1)
    per_sample_conditional = -(target * log_probabilities).sum(dim=-1)
    if torch.any(eligible):
        conditional = per_sample_conditional[eligible].mean()
    else:
        conditional = output.scores.sum() * 0.0

    return LossBreakdown(
        total=conditional,
        conditional=conditional,
        conditional_sample_count=int(eligible.sum().item()),
    )
