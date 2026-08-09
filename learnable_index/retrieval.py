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
    predicted_entropy: float
    requested_top_n: int
    policy: str
    reason: str


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    policy: Literal["fixed", "score_threshold"] = "fixed"
    top_n: int = 4
    score_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "score_threshold"}:
            raise ValueError("policy must be 'fixed' or 'score_threshold'")
        if self.top_n <= 0:
            raise ValueError("invalid retrieval budget")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")


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
    return output.scores[0]


def decide_retrieval(
    sample: RetrievalSample,
    scores: torch.Tensor,
    config: RetrievalPolicyConfig,
) -> RetrievalDecision:
    candidate_count = len(sample.candidate_blocks)
    if scores.shape != (candidate_count,):
        raise ValueError("router scores do not match candidate blocks")
    probabilities = scores.float().softmax(dim=-1)
    entropy = float(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    ordered = scores.argsort(descending=True)
    if config.policy == "fixed":
        budget = min(config.top_n, candidate_count)
        selected = ordered[:budget]
        reason = "fixed top_n"
    else:
        maximum = min(config.top_n, candidate_count)
        selected = ordered[:maximum][
            probabilities[ordered[:maximum]] >= config.score_threshold
        ]
        budget = int(selected.numel())
        reason = "manual query-key probability threshold"
    indices = tuple(int(index) for index in selected.cpu())
    return RetrievalDecision(
        selected_block_ids=tuple(sample.candidate_blocks[index].block_id for index in indices),
        selected_indices=indices,
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
