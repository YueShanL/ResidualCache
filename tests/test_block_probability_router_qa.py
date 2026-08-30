from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from block_probability_router.qa import (
    QAEvaluationConfig,
    _aggregate,
    _evidence_only_prompt,
    _pack_selected_block_kv,
    answer_contains,
    exact_match,
    token_f1,
)


def test_answer_metrics_are_normalized_but_not_teacher_forced():
    assert exact_match("The Blue Bicycle", "blue bicycle") == 1.0
    assert token_f1("blue bicycle", "bicycle") == pytest.approx(2.0 / 3.0)
    assert answer_contains("blue bicycle", "It was the blue bicycle.") == 1.0
    assert exact_match("blue bicycle", "bicycle") == 0.0


def test_evidence_only_prompt_removes_both_distractor_sides():
    record = SimpleNamespace(
        token_ids=tuple(range(20)),
        metadata={
            "distractor_token_range": [3, 15],
            "target_memory_chunk_range": [7, 10],
            "answer_start_position": 18,
        },
    )

    assert _evidence_only_prompt(record) == tuple([0, 1, 2, 7, 8, 9, 15, 16, 17])


def test_selected_block_kv_is_packed_in_chronological_candidate_order():
    block_kv = {
        0: {2: (torch.tensor([[[[1.0]]]]), torch.tensor([[[[11.0]]]]))},
        2: {2: (torch.tensor([[[[3.0]]]]), torch.tensor([[[[13.0]]]]))},
    }

    packed = _pack_selected_block_kv(block_kv, (2, 0), (2,))

    assert packed[2][0].flatten().tolist() == [1.0, 3.0]
    assert packed[2][1].flatten().tolist() == [11.0, 13.0]


def test_qa_aggregate_reports_bootstrap_and_paired_quality_deltas():
    config = QAEvaluationConfig(
        missing_mass_tolerances=(0.2,),
        bootstrap_iterations=50,
    )
    rows = []
    for router_f1 in (1.0, 0.5):
        rows.append(
            {
                "conditions": {
                    "full_context": {
                        "answer_exact_match": 1.0,
                        "answer_token_f1": 1.0,
                        "answer_contains": 1.0,
                    },
                    "evidence_only": {
                        "answer_exact_match": 1.0,
                        "answer_token_f1": 1.0,
                        "answer_contains": 1.0,
                    },
                    "local_only": {
                        "answer_exact_match": 0.0,
                        "answer_token_f1": 0.0,
                        "answer_contains": 0.0,
                    },
                    "router_epsilon_0.2": {
                        "answer_exact_match": float(router_f1 == 1.0),
                        "answer_token_f1": router_f1,
                        "answer_contains": 1.0,
                        "selected_block_count": 3,
                    },
                }
            }
        )

    summary = _aggregate(rows, config)

    router = summary["conditions"]["router_epsilon_0.2"]
    assert router["answer_token_f1"] == pytest.approx(0.75)
    assert len(router["answer_token_f1_95ci"]) == 2
    paired = summary["paired_comparisons"]["router_epsilon_0.2"]
    assert paired["delta_vs_full_context/answer_token_f1"] == pytest.approx(-0.25)
    assert paired["delta_vs_local_only/answer_token_f1"] == pytest.approx(0.75)
