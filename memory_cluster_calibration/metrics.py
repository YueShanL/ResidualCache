from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Hashable, Mapping, Sequence


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_size_metrics(memberships: Sequence[Sequence[int]]) -> dict[str, float]:
    sizes = [len(tuple(values)) for values in memberships if values]
    record_count = sum(sizes)
    singleton_clusters = sum(size == 1 for size in sizes)
    return {
        "record_count": float(record_count),
        "cluster_count": float(len(sizes)),
        "mean_cluster_size": record_count / len(sizes) if sizes else 0.0,
        "p50_cluster_size": _percentile(sizes, 0.50),
        "p90_cluster_size": _percentile(sizes, 0.90),
        "p95_cluster_size": _percentile(sizes, 0.95),
        "maximum_cluster_size": float(max(sizes, default=0)),
        "singleton_cluster_ratio": singleton_clusters / len(sizes) if sizes else 0.0,
        "singleton_record_ratio": singleton_clusters / record_count if record_count else 0.0,
    }


def bcubed_metrics(
    cluster_by_record: Mapping[Hashable, Hashable],
    label_by_record: Mapping[Hashable, Hashable],
) -> dict[str, float]:
    records = sorted(set(cluster_by_record).intersection(label_by_record), key=str)
    if not records:
        return {
            "labeled_record_count": 0.0,
            "fact_count": 0.0,
            "bcubed_precision": 0.0,
            "bcubed_recall": 0.0,
            "bcubed_f1": 0.0,
            "cluster_purity": 0.0,
            "fact_completeness": 0.0,
        }
    cluster_label_counts: dict[Hashable, Counter] = defaultdict(Counter)
    cluster_counts: Counter = Counter()
    label_counts: Counter = Counter()
    label_cluster_counts: dict[Hashable, Counter] = defaultdict(Counter)
    for record in records:
        cluster = cluster_by_record[record]
        label = label_by_record[record]
        cluster_label_counts[cluster][label] += 1
        cluster_counts[cluster] += 1
        label_counts[label] += 1
        label_cluster_counts[label][cluster] += 1
    precision = 0.0
    recall = 0.0
    for record in records:
        cluster = cluster_by_record[record]
        label = label_by_record[record]
        overlap = cluster_label_counts[cluster][label]
        precision += overlap / cluster_counts[cluster]
        recall += overlap / label_counts[label]
    precision /= len(records)
    recall /= len(records)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    purity = sum(max(counts.values()) for counts in cluster_label_counts.values()) / len(records)
    completeness = sum(max(counts.values()) for counts in label_cluster_counts.values()) / len(records)
    return {
        "labeled_record_count": float(len(records)),
        "fact_count": float(len(label_counts)),
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": f1,
        "cluster_purity": purity,
        "fact_completeness": completeness,
    }


def permutation_baseline(
    cluster_by_record: Mapping[Hashable, Hashable],
    label_by_record: Mapping[Hashable, Hashable],
    *,
    trials: int,
    seed: int,
) -> dict[str, float]:
    if trials <= 0:
        raise ValueError("permutation trials must be positive")
    records = sorted(set(cluster_by_record).intersection(label_by_record), key=str)
    labels = [label_by_record[record] for record in records]
    if not records:
        return {
            "permutation_trials": float(trials),
            "mean_bcubed_f1": 0.0,
            "p95_bcubed_f1": 0.0,
        }
    rng = random.Random(int(seed))
    values: list[float] = []
    for _ in range(trials):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        metrics = bcubed_metrics(
            cluster_by_record,
            dict(zip(records, shuffled)),
        )
        values.append(float(metrics["bcubed_f1"]))
    return {
        "permutation_trials": float(trials),
        "mean_bcubed_f1": sum(values) / len(values),
        "p95_bcubed_f1": _percentile(values, 0.95),
    }


def retrieval_fact_metrics(
    ranked_memberships: Sequence[Sequence[int]],
    label_by_record: Mapping[int, str],
    *,
    target_fact_id: str,
    top_n: int,
) -> dict[str, float]:
    selected = {
        int(record)
        for membership in ranked_memberships[: int(top_n)]
        for record in membership
    }
    selected_labeled = selected.intersection(label_by_record)
    target = {
        record
        for record, label in label_by_record.items()
        if str(label) == str(target_fact_id)
    }
    selected_target = selected.intersection(target)
    return {
        "selected_cluster_count": float(min(int(top_n), len(ranked_memberships))),
        "selected_record_count": float(len(selected)),
        "selected_labeled_record_count": float(len(selected_labeled)),
        "target_fact_recall": len(selected_target) / len(target) if target else 0.0,
        "target_fact_precision": (
            len(selected_target) / len(selected_labeled) if selected_labeled else 0.0
        ),
    }
