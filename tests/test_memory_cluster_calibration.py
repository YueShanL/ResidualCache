import pytest

from memory_cluster_calibration.metrics import (
    bcubed_metrics,
    cluster_size_metrics,
    permutation_baseline,
    retrieval_fact_metrics,
)
from memory_cluster_calibration.runner import expand_parameter_grid, _select_variant


def test_fact_metrics_reward_separation_and_are_deterministic():
    labels = {0: "a", 1: "a", 2: "b", 3: "b"}
    perfect = {0: "left", 1: "left", 2: "right", 3: "right"}
    mixed = {0: "one", 1: "one", 2: "one", 3: "one"}

    assert bcubed_metrics(perfect, labels)["bcubed_f1"] == pytest.approx(1.0)
    assert bcubed_metrics(mixed, labels)["bcubed_f1"] < 1.0
    first = permutation_baseline(perfect, labels, trials=32, seed=13)
    second = permutation_baseline(perfect, labels, trials=32, seed=13)
    assert first == second
    assert first["mean_bcubed_f1"] < 1.0


def test_structure_and_retrieval_metrics_expose_degenerate_clusters():
    structure = cluster_size_metrics(((0,), (1,), (2, 3, 4)))
    assert structure["cluster_count"] == 3.0
    assert structure["singleton_cluster_ratio"] == pytest.approx(2.0 / 3.0)
    retrieval = retrieval_fact_metrics(
        ((0, 1), (2, 3)),
        {0: "target", 1: "other", 2: "target", 3: "other"},
        target_fact_id="target",
        top_n=1,
    )
    assert retrieval["target_fact_recall"] == pytest.approx(0.5)
    assert retrieval["target_fact_precision"] == pytest.approx(0.5)


def test_grid_is_static_and_disables_eviction():
    variants = expand_parameter_grid(
        {
            "slot_capacity": 4,
            "initial_record_capacity": 8,
            "candidate_capacity": 2,
            "locality_bits": 2,
            "locality_probe_radius": 0,
            "write_chunk_size": 2,
        },
        {"alpha": [0.03, 0.1], "tau_new": [0.5, 0.9]},
    )
    assert len(variants) == 4
    assert len({row["variant_id"] for row in variants}) == 4
    assert all(row["memory_config"]["eviction_enabled"] is False for row in variants)
    assert all(row["memory_config"]["memory_budget_bytes"] is None for row in variants)


def _summary(*, sample_count, gain, consistency, singleton=0.2, mean_size=4.0):
    return {
        "sample_count": sample_count,
        "fact_separation": {
            "bcubed_f1": 0.7,
            "bcubed_f1_gain_over_permutation": gain,
            "consistent_condition_fraction": consistency,
        },
        "structure": {
            "singleton_cluster_ratio": singleton,
            "mean_cluster_size": mean_size,
            "p95_cluster_size": 16.0,
        },
    }


def test_selection_reports_possible_structural_limit_only_after_multiple_samples():
    weak = {
        "variant_id": "weak",
        "grid_values": {"tau_new": 0.9},
        "summary": _summary(sample_count=4, gain=0.01, consistency=0.25),
    }
    assert _select_variant([weak], {})["status"] == "possible_structural_limit"
    weak["summary"] = _summary(sample_count=1, gain=0.01, consistency=0.25)
    assert _select_variant([weak], {})["status"] == "insufficient_conditions"


def test_selection_rejects_fact_separation_with_bad_allocation_shape():
    fragmented = {
        "variant_id": "fragmented",
        "grid_values": {"tau_new": 0.5},
        "summary": _summary(
            sample_count=4,
            gain=0.2,
            consistency=1.0,
            singleton=0.9,
            mean_size=1.1,
        ),
    }
    result = _select_variant([fragmented], {})
    assert result["status"] == "grid_did_not_meet_allocation_constraints"
    assert result["selected_variant_id"] is None
