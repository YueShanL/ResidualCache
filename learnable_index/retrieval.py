from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from .contracts import RetrievalSample
from .model import LearnableBlockIndex


@dataclass(frozen=True)
class RetrievalDecision:
    selected_block_ids: tuple[str, ...]
    selected_indices: tuple[int, ...]
    predicted_demand: float
    predicted_entropy: float
    requested_top_n: int
    policy: str
    reason: str


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    policy: Literal["fixed", "dynamic"] = "fixed"
    top_n: int = 4
    minimum_top_n: int = 1
    maximum_top_n: int = 8
    demand_threshold: float = 0.05
    cumulative_probability_target: float = 0.8

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "dynamic"}:
            raise ValueError("policy must be 'fixed' or 'dynamic'")
        if self.top_n <= 0 or self.minimum_top_n <= 0 or self.maximum_top_n < self.minimum_top_n:
            raise ValueError("invalid retrieval budget")
        if self.demand_threshold < 0:
            raise ValueError("demand_threshold must be non-negative")
        if not 0 < self.cumulative_probability_target <= 1:
            raise ValueError("cumulative_probability_target must be in (0, 1]")


@torch.no_grad()
def score_retrieval_sample(
    model: LearnableBlockIndex,
    sample: RetrievalSample,
    *,
    device: torch.device | str,
):
    model = model.to(device)
    model.eval()
    query = sample.query_summary.to(device).unsqueeze(0)
    blocks = sample.block_summaries.to(device).unsqueeze(0)
    mask = torch.ones((1, blocks.shape[1]), dtype=torch.bool, device=device)
    output = model(query, blocks, mask)
    return output.scores[0], output.demand_logits[0]


def decide_retrieval(
    sample: RetrievalSample,
    scores: torch.Tensor,
    demand_logit: torch.Tensor,
    config: RetrievalPolicyConfig,
    *,
    demand_loss: str = "bce",
) -> RetrievalDecision:
    candidate_count = len(sample.candidate_blocks)
    if scores.shape != (candidate_count,):
        raise ValueError("router scores do not match candidate blocks")
    probabilities = scores.float().softmax(dim=-1)
    demand = (
        float(demand_logit.sigmoid())
        if demand_loss == "bce"
        else float(torch.nn.functional.softplus(demand_logit))
    )
    entropy = float(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    if config.policy == "dynamic" and demand < config.demand_threshold:
        return RetrievalDecision(
            selected_block_ids=(),
            selected_indices=(),
            predicted_demand=demand,
            predicted_entropy=entropy,
            requested_top_n=0,
            policy=config.policy,
            reason="predicted historical demand below threshold",
        )

    ordered = scores.argsort(descending=True)
    if config.policy == "fixed":
        budget = min(config.top_n, candidate_count)
        reason = "fixed top_n"
    else:
        maximum = min(config.maximum_top_n, candidate_count)
        minimum = min(config.minimum_top_n, maximum)
        ordered_probabilities = probabilities[ordered[:maximum]]
        cumulative = ordered_probabilities.cumsum(dim=0)
        reached = torch.nonzero(
            cumulative >= config.cumulative_probability_target,
            as_tuple=False,
        )
        budget = maximum if reached.numel() == 0 else int(reached[0, 0]) + 1
        budget = max(minimum, budget)
        reason = "dynamic demand/concentration budget"
    indices = tuple(int(index) for index in ordered[:budget].cpu())
    return RetrievalDecision(
        selected_block_ids=tuple(sample.candidate_blocks[index].block_id for index in indices),
        selected_indices=indices,
        predicted_demand=demand,
        predicted_entropy=entropy,
        requested_top_n=budget,
        policy=config.policy,
        reason=reason,
    )


def oracle_indices(sample: RetrievalSample, top_n: int) -> tuple[int, ...]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    budget = min(top_n, len(sample.candidate_blocks))
    return tuple(
        int(index)
        for index in sample.absolute_teacher_block_mass.argsort(descending=True)[:budget]
    )


def recent_indices(sample: RetrievalSample, top_n: int) -> tuple[int, ...]:
    budget = min(top_n, len(sample.candidate_blocks))
    ordered = sorted(
        range(len(sample.candidate_blocks)),
        key=lambda index: sample.candidate_blocks[index].end_position,
        reverse=True,
    )
    return tuple(ordered[:budget])


def normalized_entropy(decision: RetrievalDecision, candidate_count: int) -> float:
    if candidate_count <= 1:
        return 0.0
    return decision.predicted_entropy / math.log(candidate_count)
