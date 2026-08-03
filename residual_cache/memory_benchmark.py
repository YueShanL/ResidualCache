"""Controlled, model-free benchmark for retrieval-memory policy research.

The benchmark keeps the write stream, query stream, read top-k, and payload
vectors identical across methods.  It is deliberately small enough to run in a
unit test and is not a replacement for the model-level evaluation prescribed
by ``probabilistic_hierarchical_retrieval_memory.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time as clock
from typing import Callable, Hashable, Iterable, Sequence

from residual_cache.probabilistic_hierarchical_memory import (
    AdaptiveQuantileMemory,
    CamelotHardMemory,
    CompressedSources,
    HierarchicalVMFMemory,
    MemoryConfig,
    RetrievalResult,
    TemporalVMFMemory,
    VMFPosteriorMemory,
)


@dataclass(frozen=True)
class MemoryWrite:
    name: str
    index_vector: tuple[float, ...]
    original_key: tuple[float, ...]
    original_value: tuple[float, ...]
    source: Hashable
    time: float
    conflict_group: Hashable | None = None
    supersedes_write: str | None = None
    source_authority: float = 0.0


@dataclass(frozen=True)
class MemoryQuery:
    name: str
    index_vector: tuple[float, ...]
    query_key: tuple[float, ...]
    expected_sources: frozenset[Hashable]
    time: float
    temporal_policy: str = "current"
    as_of: float | None = None
    stale_sources: frozenset[Hashable] = frozenset()
    absent: bool = False


@dataclass(frozen=True)
class ControlledScenario:
    name: str
    writes: tuple[MemoryWrite, ...]
    queries: tuple[MemoryQuery, ...]
    read_top_k: int


@dataclass(frozen=True)
class BenchmarkMetrics:
    method: str
    scenario: str
    queries: int
    top1_accuracy: float
    recall_at_k: float
    absent_abstention: float
    wrong_memory_injection_rate: float
    stale_fact_injection_rate: float
    slot_count: int
    record_count: int
    memory_bytes: int
    write_seconds: float
    read_seconds: float
    diagnostics: dict


def _result_sources(result: RetrievalResult) -> set[Hashable]:
    source = result.source_token_or_span
    if isinstance(source, CompressedSources):
        return set(source.sources)
    return {source}


def evaluate_scenario(
    memory: object,
    scenario: ControlledScenario,
    *,
    method_name: str | None = None,
) -> BenchmarkMetrics:
    """Run one immutable event stream against one fresh memory instance."""

    decisions: dict[str, str] = {}
    write_started = clock.perf_counter()
    for event in scenario.writes:
        supersedes = decisions.get(event.supersedes_write) if event.supersedes_write else None
        decision = memory.write(
            event.index_vector,
            event.original_key,
            event.original_value,
            source_token_or_span=event.source,
            time=event.time,
            conflict_group=event.conflict_group,
            supersedes=supersedes,
            source_authority=event.source_authority,
        )
        decisions[event.name] = decision.record_id
    write_seconds = clock.perf_counter() - write_started

    top1_hits = 0
    recall_hits = 0
    absent_queries = 0
    absent_abstentions = 0
    wrong_injections = 0
    stale_injections = 0
    read_started = clock.perf_counter()
    for query in scenario.queries:
        results = memory.retrieve(
            query.index_vector,
            query_key=query.query_key,
            time=query.time,
            temporal_policy=query.temporal_policy,
            as_of=query.as_of,
            top_k=scenario.read_top_k,
        )
        source_sets = [_result_sources(result) for result in results]
        if query.absent:
            absent_queries += 1
            absent_abstentions += int(not results)
            wrong_injections += int(bool(results))
            continue
        top1_hits += int(
            bool(source_sets)
            and bool(source_sets[0].intersection(query.expected_sources))
            and source_sets[0].issubset(query.expected_sources)
        )
        recall_hits += int(
            any(sources.intersection(query.expected_sources) for sources in source_sets)
        )
        wrong_injections += int(
            any(not sources.issubset(query.expected_sources) for sources in source_sets)
        )
        stale_injections += int(
            any(sources.intersection(query.stale_sources) for sources in source_sets)
        )
    read_seconds = clock.perf_counter() - read_started
    query_count = len(scenario.queries)
    non_absent = query_count - absent_queries
    snapshot = memory.snapshot()
    return BenchmarkMetrics(
        method=method_name or type(memory).__name__,
        scenario=scenario.name,
        queries=query_count,
        top1_accuracy=top1_hits / non_absent if non_absent else 0.0,
        recall_at_k=recall_hits / non_absent if non_absent else 0.0,
        absent_abstention=(
            absent_abstentions / absent_queries if absent_queries else 0.0
        ),
        wrong_memory_injection_rate=wrong_injections / query_count if query_count else 0.0,
        stale_fact_injection_rate=stale_injections / non_absent if non_absent else 0.0,
        slot_count=int(snapshot["slot_count"]),
        record_count=int(snapshot["record_count"]),
        memory_bytes=int(snapshot["memory_bytes"]),
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        diagnostics=snapshot,
    )


def sweep_camelot_hard(
    scenario: ControlledScenario,
    thresholds: Iterable[float],
) -> list[BenchmarkMetrics]:
    return [
        evaluate_scenario(
            CamelotHardMemory(threshold, record_top_k=scenario.read_top_k),
            scenario,
            method_name=f"CAMELoT-Hard(R={threshold:g})",
        )
        for threshold in thresholds
    ]


def sweep_adaptive_quantile(
    scenario: ControlledScenario,
    quantiles: Iterable[float],
    *,
    initial_threshold: float = 0.9,
    minimum_history: int = 4,
) -> list[BenchmarkMetrics]:
    return [
        evaluate_scenario(
            AdaptiveQuantileMemory(
                initial_threshold=initial_threshold,
                quantile=quantile,
                minimum_history=minimum_history,
                record_top_k=scenario.read_top_k,
            ),
            scenario,
            method_name=f"Adaptive-Quantile(q={quantile:g})",
        )
        for quantile in quantiles
    ]


def sweep_vmf(
    scenario: ControlledScenario,
    configs: Iterable[MemoryConfig],
    *,
    variant: str,
) -> list[BenchmarkMetrics]:
    factories: dict[str, Callable[[MemoryConfig], object]] = {
        "vmf": VMFPosteriorMemory,
        "temporal": TemporalVMFMemory,
        "hierarchical": HierarchicalVMFMemory,
    }
    if variant not in factories:
        raise ValueError(f"Unknown vMF variant {variant!r}.")
    rows = []
    for config in configs:
        memory = factories[variant](config)
        rows.append(
            evaluate_scenario(
                memory,
                scenario,
                method_name=(
                    f"{type(memory).__name__}"
                    f"(alpha={config.alpha:g},tau={config.tau_new:g},"
                    f"gamma={config.count_exponent:g})"
                ),
            )
        )
    return rows


def pareto_frontier(
    rows: Sequence[BenchmarkMetrics],
    *,
    quality_field: str = "recall_at_k",
) -> list[BenchmarkMetrics]:
    """Return points not dominated in quality, bytes, and total runtime."""

    frontier = []
    for candidate in rows:
        quality = float(getattr(candidate, quality_field))
        latency = candidate.write_seconds + candidate.read_seconds
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_quality = float(getattr(other, quality_field))
            other_latency = other.write_seconds + other.read_seconds
            no_worse = (
                other_quality >= quality
                and other.memory_bytes <= candidate.memory_bytes
                and other_latency <= latency
            )
            strictly_better = (
                other_quality > quality
                or other.memory_bytes < candidate.memory_bytes
                or other_latency < latency
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            row.memory_bytes,
            -(float(getattr(row, quality_field))),
            row.write_seconds + row.read_seconds,
        ),
    )


def best_under_memory_budget(
    rows: Sequence[BenchmarkMetrics],
    memory_bytes: int,
    *,
    quality_field: str = "recall_at_k",
) -> BenchmarkMetrics:
    eligible = [row for row in rows if row.memory_bytes <= memory_bytes]
    if not eligible:
        raise ValueError("No benchmark point satisfies the requested memory budget.")
    return max(
        eligible,
        key=lambda row: (
            float(getattr(row, quality_field)),
            row.top1_accuracy,
            -row.wrong_memory_injection_rate,
            -row.memory_bytes,
        ),
    )


def assert_effective_baseline(
    rows: Sequence[BenchmarkMetrics],
    *,
    minimum_recall: float = 0.8,
    minimum_top1: float = 0.8,
) -> BenchmarkMetrics:
    """Fail fast when a supposedly fair hard-threshold sweep is ineffective."""

    if not rows:
        raise AssertionError("The baseline sweep is empty.")
    best = max(rows, key=lambda row: (row.recall_at_k, row.top1_accuracy))
    if best.recall_at_k < minimum_recall:
        raise AssertionError(
            f"Best baseline recall {best.recall_at_k:.3f} is below "
            f"the required {minimum_recall:.3f}; the comparison is not informative."
        )
    if best.top1_accuracy < minimum_top1:
        raise AssertionError(
            f"Best baseline top-1 accuracy {best.top1_accuracy:.3f} is below "
            f"the required {minimum_top1:.3f}; the comparison is not informative."
        )
    return best


def make_static_cluster_scenario() -> ControlledScenario:
    """Two separable topics with within-topic paraphrase variation."""

    writes = (
        MemoryWrite("a1", (1.0, 0.02), (1.0, 0.02), (1.0, 0.0), "A", 1.0),
        MemoryWrite("a2", (1.0, -0.03), (1.0, -0.03), (1.0, 0.0), "A", 2.0),
        MemoryWrite("a3", (0.99, 0.05), (0.99, 0.05), (1.0, 0.0), "A", 3.0),
        MemoryWrite("b1", (0.02, 1.0), (0.02, 1.0), (0.0, 1.0), "B", 4.0),
        MemoryWrite("b2", (-0.03, 1.0), (-0.03, 1.0), (0.0, 1.0), "B", 5.0),
        MemoryWrite("b3", (0.05, 0.99), (0.05, 0.99), (0.0, 1.0), "B", 6.0),
    )
    queries = (
        MemoryQuery("qa", (1.0, 0.0), (1.0, 0.0), frozenset({"A"}), 7.0),
        MemoryQuery("qb", (0.0, 1.0), (0.0, 1.0), frozenset({"B"}), 8.0),
    )
    return ControlledScenario("static_two_cluster", writes, queries, read_top_k=1)


def make_temporal_conflict_scenario() -> ControlledScenario:
    """Same-key state update plus a time-specific historical query."""

    writes = (
        MemoryWrite(
            "old",
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            "Paris",
            1.0,
            conflict_group="alice-location",
        ),
        MemoryWrite(
            "new",
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            "London",
            10.0,
            conflict_group="alice-location",
            supersedes_write="old",
        ),
    )
    queries = (
        MemoryQuery(
            "current",
            (1.0, 0.0),
            (1.0, 0.0),
            frozenset({"London"}),
            11.0,
            stale_sources=frozenset({"Paris"}),
        ),
        MemoryQuery(
            "historical",
            (1.0, 0.0),
            (1.0, 0.0),
            frozenset({"Paris"}),
            12.0,
            temporal_policy="historical",
            as_of=5.0,
            stale_sources=frozenset({"London"}),
        ),
    )
    return ControlledScenario("temporal_conflict", writes, queries, read_top_k=1)


def default_five_way_benchmark(
    scenario: ControlledScenario,
) -> list[BenchmarkMetrics]:
    """Run one representative point for every mandatory comparison variant."""

    config = MemoryConfig(
        alpha=0.05,
        tau_new=0.5,
        count_exponent=0.5,
        route_top_k=4,
        record_top_k=scenario.read_top_k,
        enable_split_merge=True,
    )
    memories = (
        ("CAMELoT-Hard", CamelotHardMemory(0.9, record_top_k=scenario.read_top_k)),
        (
            "Adaptive-Quantile",
            AdaptiveQuantileMemory(
                initial_threshold=0.9,
                minimum_history=4,
                record_top_k=scenario.read_top_k,
            ),
        ),
        ("vMF-Posterior", VMFPosteriorMemory(config)),
        ("Temporal-vMF", TemporalVMFMemory(config)),
        ("Hierarchical-vMF", HierarchicalVMFMemory(config)),
    )
    return [
        evaluate_scenario(memory, scenario, method_name=name)
        for name, memory in memories
    ]


def write_json(path: Path, rows: Sequence[BenchmarkMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "BenchmarkMetrics",
    "ControlledScenario",
    "MemoryQuery",
    "MemoryWrite",
    "assert_effective_baseline",
    "best_under_memory_budget",
    "default_five_way_benchmark",
    "evaluate_scenario",
    "make_static_cluster_scenario",
    "make_temporal_conflict_scenario",
    "pareto_frontier",
    "sweep_adaptive_quantile",
    "sweep_camelot_hard",
    "sweep_vmf",
    "write_json",
]
