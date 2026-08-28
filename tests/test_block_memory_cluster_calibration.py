from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from block_memory_cluster_calibration.metrics import (
    build_block_fact_labels,
    target_block_retrieval_metrics,
)
from block_memory_cluster_calibration.runner import (
    FixedBlockCache,
    CachedBlock,
    _evaluate_variant,
    _select_variant,
    expand_parameter_grid,
    validate_config,
)


def _router(axis: int):
    value = torch.zeros(32, dtype=torch.float32)
    value[axis] = 1.0
    return value


def _block(block_id: str, start: int, axis: int) -> CachedBlock:
    key = torch.full((1, 1, 4, 32), float(start), dtype=torch.float32)
    return CachedBlock(
        block_id=block_id,
        logical_positions=tuple(range(start, start + 4)),
        router_key=_router(axis),
        key=key,
        value=key + 100.0,
    )


def test_block_fact_labels_preserve_ambiguity_and_target_membership():
    labels = build_block_fact_labels(
        (
            {"fact_id": "a", "start": 0, "end": 3},
            {"fact_id": "b", "start": 3, "end": 6},
            {"fact_id": "target", "start": 8, "end": 12},
        ),
        {
            "block-0": (0, 1, 2, 3),
            "block-1": (4, 5, 6, 7),
            "block-2": (8, 9, 10, 11),
        },
        target_fact_id="target",
    )

    assert labels.primary_by_block == {
        "block-0": "a",
        "block-1": "b",
        "block-2": "target",
    }
    assert labels.facts_by_block["block-0"] == ("a", "b")
    assert labels.target_block_ids == ("block-2",)
    assert labels.diagnostics["ambiguous_labeled_block_ratio"] == pytest.approx(
        1.0 / 3.0
    )


def test_target_retrieval_uses_any_target_overlap_and_reports_compression():
    metrics = target_block_retrieval_metrics(
        (("a", "target"), ("b", "c")),
        labeled_block_ids=("a", "target", "b"),
        target_block_ids=("target",),
        total_block_count=4,
        top_n=1,
    )
    assert metrics["target_fact_block_recall"] == 1.0
    assert metrics["target_fact_block_precision"] == pytest.approx(0.5)
    assert metrics["selected_block_ratio"] == pytest.approx(0.5)


def test_grid_uses_only_block_memory_parameters_and_disables_eviction():
    variants = expand_parameter_grid(
        {
            "slot_capacity": 4,
            "initial_record_capacity": 4,
            "candidate_capacity": 2,
            "locality_bits": 2,
            "locality_probe_radius": 0,
        },
        {"alpha": [0.01, 0.1], "tau_new": [0.5, 0.9]},
        block_size=4,
    )
    assert len(variants) == 4
    assert all(row["memory_config"]["block_size"] == 4 for row in variants)
    assert all(row["memory_config"]["eviction_enabled"] is False for row in variants)
    assert all(row["memory_config"]["memory_budget_bytes"] is None for row in variants)
    with pytest.raises(ValueError, match="unsupported block-memory"):
        expand_parameter_grid(
            {}, {"index_mode": ["mean_kv"]}, block_size=4
        )


def test_fixed_cache_variant_runs_the_real_block_memory():
    blocks = (
        _block("a-0", 0, 0),
        _block("a-1", 4, 0),
        _block("target", 8, 1),
    )
    cache = FixedBlockCache(
        sample_id="sample",
        target_fact_id="target",
        primary_label_by_block={"a-0": "a", "a-1": "a", "target": "target"},
        target_block_ids=("target",),
        label_diagnostics={"ambiguous_labeled_block_ratio": 0.0},
        query_router_key=_router(1),
        blocks=blocks,
        payload_layer=23,
        equivalent_memory_layers=(5, 11, 17, 23),
        forward_calls=3,
        forwarded_tokens=12,
        maximum_forward_context_length=8,
        final_local_context_length=4,
    )
    variant = expand_parameter_grid(
        {
            "slot_capacity": 2,
            "initial_record_capacity": 3,
            "candidate_capacity": 1,
            "locality_bits": 2,
            "locality_probe_radius": 0,
        },
        {},
        block_size=4,
    )[0]
    row = _evaluate_variant(
        cache,
        variant,
        top_ns=(1,),
        permutation_trials=8,
        seed=13,
    )
    assert row["memory_snapshot"]["record_unit"] == "layer_local_block"
    assert row["memory_snapshot"]["active_records"] == 3
    assert row["classification_equivalent_layers"] == [5, 11, 17, 23]


def _summary(
    *,
    samples: int,
    gain: float,
    consistency: float,
    recall: float,
    selected_ratio: float,
    singleton: float = 0.2,
):
    return {
        "sample_count": samples,
        "fact_separation": {
            "bcubed_f1": 0.7,
            "bcubed_f1_gain_over_permutation": gain,
            "consistent_condition_fraction": consistency,
        },
        "structure": {
            "singleton_cluster_ratio": singleton,
            "mean_cluster_size": 4.0,
            "p95_cluster_size": 16.0,
        },
        "retrieval": {
            "top_4": {
                "target_fact_block_recall": recall,
                "selected_block_ratio": selected_ratio,
            }
        },
    }


def test_selection_requires_fact_separation_retrieval_and_compression():
    good = {
        "variant_id": "good",
        "grid_values": {"tau_new": 0.5},
        "summary": _summary(
            samples=4, gain=0.2, consistency=1.0, recall=0.9, selected_ratio=0.5
        ),
    }
    assert _select_variant([good], {})["status"] == "calibrated"
    weak = {
        "variant_id": "weak",
        "grid_values": {"tau_new": 0.9},
        "summary": _summary(
            samples=4, gain=0.01, consistency=0.25, recall=0.4, selected_ratio=0.5
        ),
    }
    assert _select_variant([weak], {})["status"] == "possible_structural_limit"
    broad = {
        "variant_id": "broad",
        "grid_values": {"tau_new": 0.99},
        "summary": _summary(
            samples=4, gain=0.2, consistency=1.0, recall=1.0, selected_ratio=1.0
        ),
    }
    assert (
        _select_variant([broad], {})["status"]
        == "grid_did_not_meet_allocation_constraints"
    )


def test_validate_only_checks_checkpoint_and_grid_without_loading_model(tmp_path: Path):
    checkpoint = tmp_path / "router.pt"
    checkpoint.write_bytes(b"placeholder")
    config = {
        "output_dir": str(tmp_path / "output"),
        "data": {
            "sequence_count": 1,
            "sequence_length": 4096,
            "block_size": 64,
            "local_context_length": 512,
        },
        "model": {
            "checkpoint_path": str(checkpoint),
            "block_size": 64,
            "local_context_length": 512,
            "query_summary_length": 16,
        },
        "sweep": {"base_memory_config": {}, "grid": {}},
        "evaluation": {"top_ns": [1, 4], "permutation_trials": 8},
        "criteria": {"selection_top_n": 4},
    }
    result = validate_config(config)
    assert result["variant_count"] == 1
    assert result["block_size"] == 64
