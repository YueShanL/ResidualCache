"""Block-specific offline labels and retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class BlockFactLabels:
    primary_by_block: Mapping[str, str]
    facts_by_block: Mapping[str, tuple[str, ...]]
    overlap_tokens_by_block: Mapping[str, Mapping[str, int]]
    target_block_ids: tuple[str, ...]
    diagnostics: Mapping[str, float]


def build_block_fact_labels(
    fact_ranges: Sequence[Mapping[str, object]],
    block_positions: Mapping[str, Sequence[int]],
    *,
    target_fact_id: str,
) -> BlockFactLabels:
    """Build offline block labels from exact fact-token overlaps.

    B-cubed needs one hard label, so the largest-overlap fact is the primary
    label.  The full set is retained for ambiguity diagnostics and target-fact
    retrieval, where a block is relevant if it contains any target tokens.
    """

    normalized_ranges: list[tuple[str, int, int]] = []
    for item in fact_ranges:
        fact_id = str(item["fact_id"])
        start = int(item["start"])
        end = int(item["end"])
        if start >= end:
            raise ValueError("memory fact range must be non-empty")
        normalized_ranges.append((fact_id, start, end))

    primary: dict[str, str] = {}
    facts: dict[str, tuple[str, ...]] = {}
    overlaps_by_block: dict[str, dict[str, int]] = {}
    dominant_fractions: list[float] = []
    ambiguous = 0
    for raw_block_id, raw_positions in block_positions.items():
        block_id = str(raw_block_id)
        positions = tuple(int(value) for value in raw_positions)
        if not positions:
            raise ValueError("calibration block cannot be empty")
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise ValueError("calibration block positions must be contiguous")
        block_start = positions[0]
        block_end = positions[-1] + 1
        overlap: dict[str, int] = {}
        for fact_id, start, end in normalized_ranges:
            count = max(0, min(block_end, end) - max(block_start, start))
            if count:
                overlap[fact_id] = overlap.get(fact_id, 0) + count
        if not overlap:
            continue
        ordered = tuple(sorted(overlap))
        facts[block_id] = ordered
        overlaps_by_block[block_id] = overlap
        primary[block_id] = min(
            overlap,
            key=lambda fact_id: (-overlap[fact_id], fact_id),
        )
        total_overlap = sum(overlap.values())
        dominant_fractions.append(max(overlap.values()) / total_overlap)
        ambiguous += int(len(overlap) > 1)

    if len(set(primary.values())) < 2:
        raise ValueError("block calibration requires at least two primary fact labels")
    target_blocks = tuple(
        sorted(
            block_id
            for block_id, fact_ids in facts.items()
            if str(target_fact_id) in fact_ids
        )
    )
    if not target_blocks:
        raise ValueError("target fact has no complete block in historical memory")
    labeled_count = len(primary)
    return BlockFactLabels(
        primary_by_block=primary,
        facts_by_block=facts,
        overlap_tokens_by_block=overlaps_by_block,
        target_block_ids=target_blocks,
        diagnostics={
            "memory_block_count": float(len(block_positions)),
            "labeled_block_count": float(labeled_count),
            "fact_id_count": float(len({value for values in facts.values() for value in values})),
            "primary_fact_id_count": float(len(set(primary.values()))),
            "ambiguous_labeled_block_count": float(ambiguous),
            "ambiguous_labeled_block_ratio": (
                ambiguous / labeled_count if labeled_count else 0.0
            ),
            "mean_dominant_fact_overlap_fraction": (
                sum(dominant_fractions) / len(dominant_fractions)
                if dominant_fractions
                else 0.0
            ),
            "target_block_count": float(len(target_blocks)),
        },
    )


def target_block_retrieval_metrics(
    ranked_memberships: Sequence[Sequence[Hashable]],
    *,
    labeled_block_ids: Sequence[Hashable],
    target_block_ids: Sequence[Hashable],
    total_block_count: int,
    top_n: int,
) -> dict[str, float]:
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    selected = {
        str(block_id)
        for membership in ranked_memberships[: int(top_n)]
        for block_id in membership
    }
    labeled = {str(value) for value in labeled_block_ids}
    target = {str(value) for value in target_block_ids}
    selected_labeled = selected.intersection(labeled)
    selected_target = selected.intersection(target)
    return {
        "selected_cluster_count": float(min(int(top_n), len(ranked_memberships))),
        "selected_block_count": float(len(selected)),
        "selected_labeled_block_count": float(len(selected_labeled)),
        "selected_block_ratio": (
            len(selected) / int(total_block_count) if total_block_count else 0.0
        ),
        "target_fact_block_recall": (
            len(selected_target) / len(target) if target else 0.0
        ),
        "target_fact_block_precision": (
            len(selected_target) / len(selected_labeled)
            if selected_labeled
            else 0.0
        ),
    }


__all__ = [
    "BlockFactLabels",
    "build_block_fact_labels",
    "target_block_retrieval_metrics",
]
