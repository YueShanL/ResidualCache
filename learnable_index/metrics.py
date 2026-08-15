from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .data import RetrievalBatch
from .model import RouterOutput


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


@torch.no_grad()
def update_router_metrics(
    accumulator: MetricAccumulator,
    output: RouterOutput,
    batch: RetrievalBatch,
    *,
    top_n: int,
    epsilon: float = 1e-8,
) -> None:
    scores = output.scores
    mask = batch.candidate_mask
    target = batch.conditional_teacher_distribution
    positive = batch.total_teacher_historical_mass > epsilon
    log_probabilities = F.log_softmax(scores, dim=-1)
    predicted = log_probabilities.exp()

    if torch.any(positive):
        target_positive = target[positive]
        predicted_positive = predicted[positive]
        log_positive = log_probabilities[positive]
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
        predicted_top = scores.argmax(dim=-1)
        accumulator.add("top1_teacher_block_recall", (teacher_top == predicted_top)[positive].float())
        top_n_hits: list[torch.Tensor] = []
        for row in torch.nonzero(positive, as_tuple=False).flatten():
            candidate_count = int(mask[row].sum().item())
            budget = min(top_n, candidate_count)
            predicted_indices = scores[row, :candidate_count].topk(budget).indices
            top_n_hits.append((predicted_indices == teacher_top[row]).any().float())
        accumulator.add(f"top{top_n}_teacher_block_recall", torch.stack(top_n_hits))

    for row in range(scores.shape[0]):
        count = int(mask[row].sum().item())
        if count == 0:
            continue
        budget = min(top_n, count)
        total_mass = float(batch.absolute_teacher_block_mass[row, :count].sum())
        if total_mass <= epsilon:
            continue
        predicted_indices = scores[row, :count].topk(budget).indices
        oracle_indices = batch.absolute_teacher_block_mass[row, :count].topk(budget).indices
        predicted_coverage = (
            batch.absolute_teacher_block_mass[row, predicted_indices].sum() / total_mass
        )
        oracle_coverage = batch.absolute_teacher_block_mass[row, oracle_indices].sum() / total_mass
        accumulator.add(f"predicted_coverage@{top_n}", predicted_coverage.reshape(1))
        accumulator.add(f"oracle_coverage@{top_n}", oracle_coverage.reshape(1))
        if batch.per_future_teacher_block_mass is not None:
            for distance_index in range(batch.per_future_teacher_block_mass.shape[1]):
                if (
                    batch.per_future_mask is not None
                    and not bool(batch.per_future_mask[row, distance_index])
                ):
                    continue
                distance_mass = batch.per_future_teacher_block_mass[
                    row, distance_index, :count
                ]
                distance_total = distance_mass.sum()
                accumulator.add(
                    f"historical_mass/distance_{distance_index + 1}",
                    distance_total.reshape(1),
                )
                if float(distance_total) <= epsilon:
                    continue
                distance_oracle = distance_mass.topk(budget).indices
                accumulator.add(
                    f"predicted_coverage@{top_n}/distance_{distance_index + 1}",
                    (distance_mass[predicted_indices].sum() / distance_total).reshape(1),
                )
                accumulator.add(
                    f"oracle_coverage@{top_n}/distance_{distance_index + 1}",
                    (distance_mass[distance_oracle].sum() / distance_total).reshape(1),
                )


def finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {name: value for name, value in metrics.items() if math.isfinite(value)}
