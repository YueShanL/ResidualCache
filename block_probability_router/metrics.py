from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import torch

from learnable_index.data import RetrievalBatch

from .model import (
    BlockProbabilityRouter,
    ProbabilityRouterOutput,
    minimum_cumulative_mass_mask,
)


class MetricAccumulator:
    def __init__(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, values: torch.Tensor | Iterable[float]) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            return
        self._sums[name] += float(finite.sum())
        self._counts[name] += int(finite.numel())

    def compute(self) -> dict[str, float]:
        return {
            name: self._sums[name] / self._counts[name]
            for name in sorted(self._sums)
            if self._counts[name] > 0
        }


def _mass_label(value: float) -> str:
    return format(value, ".6g")


@torch.no_grad()
def update_probability_metrics(
    accumulator: MetricAccumulator,
    output: ProbabilityRouterOutput,
    batch: RetrievalBatch,
    *,
    top_n: int,
    missing_mass_tolerances: tuple[float, ...],
    maximum_retrieval_blocks: int = -1,
    epsilon: float = 1e-8,
) -> None:
    mask = batch.candidate_mask
    target = batch.conditional_teacher_distribution
    probabilities = output.probabilities
    eligible = batch.total_teacher_historical_mass > epsilon

    normalization_error = (probabilities.sum(dim=-1) - 1.0).abs()
    accumulator.add("probability_normalization_error", normalization_error)
    if torch.any(eligible):
        target_positive = target[eligible]
        predicted_positive = probabilities[eligible]
        log_positive = output.log_probabilities.masked_fill(~mask, 0.0)[eligible]
        conditional_ce = -(target_positive * log_positive).sum(dim=-1)
        target_entropy = -(target_positive * target_positive.clamp_min(epsilon).log()).sum(dim=-1)
        predicted_entropy = -(
            predicted_positive * predicted_positive.clamp_min(epsilon).log()
        ).sum(dim=-1)
        accumulator.add("conditional_cross_entropy", conditional_ce)
        accumulator.add("conditional_kl", conditional_ce - target_entropy)
        accumulator.add("teacher_entropy", target_entropy)
        accumulator.add("prediction_entropy", predicted_entropy)

        teacher_top = target.argmax(dim=-1)
        predicted_top = probabilities.argmax(dim=-1)
        accumulator.add("top1_teacher_block_recall", (teacher_top == predicted_top)[eligible].float())
        top_n_hits: list[torch.Tensor] = []
        for row in torch.nonzero(eligible, as_tuple=False).flatten():
            candidate_count = int(mask[row].sum().item())
            budget = min(top_n, candidate_count)
            indices = probabilities[row, :candidate_count].topk(budget).indices
            top_n_hits.append((indices == teacher_top[row]).any().float())
        accumulator.add(f"top{top_n}_teacher_block_recall", torch.stack(top_n_hits))

        predicted_coverage: list[torch.Tensor] = []
        oracle_coverage: list[torch.Tensor] = []
        for row in torch.nonzero(eligible, as_tuple=False).flatten():
            candidate_count = int(mask[row].sum().item())
            budget = min(top_n, candidate_count)
            predicted_indices = probabilities[row, :candidate_count].topk(budget).indices
            oracle_indices = target[row, :candidate_count].topk(budget).indices
            predicted_coverage.append(target[row, predicted_indices].sum())
            oracle_coverage.append(target[row, oracle_indices].sum())
        accumulator.add(f"predicted_conditional_coverage@{top_n}", torch.stack(predicted_coverage))
        accumulator.add(f"oracle_conditional_coverage@{top_n}", torch.stack(oracle_coverage))

    for tolerance in missing_mass_tolerances:
        uncapped_selected = BlockProbabilityRouter.cumulative_mass_mask(
            output,
            mask,
            tolerance,
        )
        selected = BlockProbabilityRouter.cumulative_mass_mask(
            output,
            mask,
            tolerance,
            maximum_retrieval_blocks,
        )
        selected_count = selected.sum(dim=-1)
        uncapped_selected_count = uncapped_selected.sum(dim=-1)
        candidate_count = mask.sum(dim=-1)
        label = _mass_label(tolerance)
        prefix = f"missing_mass/{label}"
        retained_mass_target = 1.0 - tolerance
        predicted_mass = (probabilities * selected.to(probabilities.dtype)).sum(dim=-1)
        accumulator.add(f"{prefix}/retained_mass_target", torch.full_like(predicted_mass, retained_mass_target))
        accumulator.add(f"{prefix}/selected_blocks", selected_count)
        accumulator.add(
            f"{prefix}/selected_fraction",
            selected_count.float() / candidate_count.clamp_min(1),
        )
        accumulator.add(f"{prefix}/predicted_probability_mass", predicted_mass)
        accumulator.add(
            f"{prefix}/predicted_target_success_rate",
            (predicted_mass >= retained_mass_target - epsilon).float(),
        )
        if maximum_retrieval_blocks != -1:
            truncated_blocks = uncapped_selected_count - selected_count
            accumulator.add(
                f"{prefix}/maximum_retrieval_blocks",
                torch.full_like(selected_count, maximum_retrieval_blocks),
            )
            accumulator.add(
                f"{prefix}/uncapped_selected_blocks",
                uncapped_selected_count,
            )
            accumulator.add(f"{prefix}/truncated_blocks", truncated_blocks)
            accumulator.add(
                f"{prefix}/cap_applied_rate",
                (truncated_blocks > 0).float(),
            )
        if torch.any(eligible):
            captured = (target * selected.to(target.dtype)).sum(dim=-1)
            eligible_target = target[eligible]
            eligible_mask = mask[eligible]
            oracle_uncapped_selected = minimum_cumulative_mass_mask(
                eligible_target,
                eligible_mask,
                tolerance,
            )
            oracle_selected = minimum_cumulative_mass_mask(
                eligible_target,
                eligible_mask,
                tolerance,
                maximum_retrieval_blocks,
            )
            oracle_count = oracle_selected.sum(dim=-1)
            oracle_uncapped_count = oracle_uncapped_selected.sum(dim=-1)
            oracle_mass = (
                eligible_target * oracle_selected.to(eligible_target.dtype)
            ).sum(dim=-1)
            eligible_captured = captured[eligible]
            accumulator.add(f"{prefix}/teacher_mass_recall", eligible_captured)
            accumulator.add(f"{prefix}/oracle_selected_blocks", oracle_count)
            accumulator.add(f"{prefix}/oracle_teacher_mass", oracle_mass)
            accumulator.add(
                f"{prefix}/oracle_target_success_rate",
                (oracle_mass >= retained_mass_target - epsilon).float(),
            )
            accumulator.add(
                f"{prefix}/oracle_mass_shortfall",
                (retained_mass_target - oracle_mass).clamp_min(0.0),
            )
            accumulator.add(
                f"{prefix}/selected_blocks_minus_oracle",
                selected_count[eligible].float() - oracle_count.float(),
            )
            if maximum_retrieval_blocks != -1:
                oracle_truncated = oracle_uncapped_count - oracle_count
                accumulator.add(
                    f"{prefix}/oracle_uncapped_selected_blocks",
                    oracle_uncapped_count,
                )
                accumulator.add(
                    f"{prefix}/oracle_truncated_blocks",
                    oracle_truncated,
                )
                accumulator.add(
                    f"{prefix}/oracle_cap_applied_rate",
                    (oracle_truncated > 0).float(),
                )
                accumulator.add(
                    f"{prefix}/uncapped_selected_blocks_minus_oracle",
                    uncapped_selected_count[eligible].float()
                    - oracle_uncapped_count.float(),
                )
            accumulator.add(
                f"{prefix}/teacher_target_success_rate",
                (eligible_captured >= retained_mass_target - epsilon).float(),
            )
            accumulator.add(
                f"{prefix}/teacher_mass_shortfall",
                (retained_mass_target - eligible_captured).clamp_min(0.0),
            )


def finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: value for name, value in metrics.items() if math.isfinite(value)}
