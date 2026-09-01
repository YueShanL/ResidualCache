from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from block_probability_router.oracle_replay_smoke import (
    select_teacher_oracle_blocks,
)
from block_probability_router.full_context_replay import (
    FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL,
    collect_full_context_replay_state,
)
from block_probability_router.qa import (
    QAEvaluationConfig,
    _aggregate,
    _evidence_only_prompt,
    _generate_sparse_replay,
    _pack_selected_block_kv,
    _router_selections,
    answer_contains,
    exact_match,
    token_f1,
)
from block_probability_router.streaming_collection import (
    collect_streaming_student_state,
)
from learnable_index.collectors import StudentCollectionConfig
from learnable_index.contracts import BlockRange
from learnable_index.model_adapter import cache_from_layer_kv, layer_kv_from_cache
from learnable_index.planning import RetrievalPlan, SequenceRecord


def test_answer_metrics_are_normalized_but_not_teacher_forced():
    assert exact_match("The Blue Bicycle", "blue bicycle") == 1.0
    assert token_f1("blue bicycle", "bicycle") == pytest.approx(2.0 / 3.0)
    assert answer_contains("blue bicycle", "It was the blue bicycle.") == 1.0
    assert exact_match("blue bicycle", "bicycle") == 0.0


def test_teacher_oracle_selects_minimum_global_mass_prefix():
    distribution = torch.tensor([0.10, 0.60, 0.05, 0.25])

    selected = select_teacher_oracle_blocks(distribution, 0.10)

    assert selected == (0, 1, 3)
    assert float(distribution[list(selected)].sum()) == pytest.approx(0.95)


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


def test_router_selection_casts_streaming_bfloat16_states_to_checkpoint_dtype():
    class FloatRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, query, blocks, candidate_mask):
            assert query.dtype == self.anchor.dtype
            assert blocks.dtype == self.anchor.dtype
            probabilities = torch.tensor(
                [[0.7, 0.2, 0.1]], dtype=self.anchor.dtype, device=query.device
            )
            return SimpleNamespace(probabilities=probabilities)

    probabilities, selections = _router_selections(
        FloatRouter(),
        torch.ones(4, dtype=torch.bfloat16),
        torch.ones(3, 4, dtype=torch.bfloat16),
        QAEvaluationConfig(
            missing_mass_tolerances=(0.2,),
            bootstrap_iterations=10,
        ),
        torch.device("cpu"),
    )

    assert probabilities.dtype == torch.float32
    assert selections[0.2] == (0, 1)


def test_sparse_generation_uses_block_aligned_growth_and_atomic_eviction(monkeypatch):
    observed_past_lengths = []
    observed_mask_lengths = []

    def fake_forward(
        _bundle,
        token_ids,
        logical_positions,
        *,
        past_key_values,
        attention_mask,
        use_cache,
        logical_cache_position,
        **_kwargs,
    ):
        assert use_cache and logical_cache_position
        pairs = layer_kv_from_cache(past_key_values)
        observed_past_lengths.append(int(pairs[0][0].shape[2]))
        observed_mask_lengths.append(
            int(attention_mask["full_attention"].shape[-1])
        )
        current = torch.tensor(
            [[[[float(tuple(logical_positions)[0])]]]], dtype=torch.float32
        )
        updated = tuple(
            (
                torch.cat((key, current), dim=2),
                torch.cat((value, current + 100.0), dim=2),
            )
            for key, value in pairs
        )
        logits = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
        return SimpleNamespace(
            past_key_values=cache_from_layer_kv(updated),
            logits=logits,
        )

    monkeypatch.setattr("block_probability_router.qa.forward_tokens", fake_forward)
    monkeypatch.setattr(
        "block_probability_router.qa._synchronize_bundle", lambda _bundle: None
    )
    bundle = SimpleNamespace(
        input_device=torch.device("cpu"),
        cache_layer_devices=(torch.device("cpu"),),
        text_model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))
        ),
        text_config=SimpleNamespace(layer_types=("full_attention",)),
        tokenizer=SimpleNamespace(
            eos_token_id=None,
            decode=lambda token_ids, skip_special_tokens: " ".join(
                map(str, token_ids)
            ),
        ),
    )
    initial = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)

    result = _generate_sparse_replay(
        bundle,
        local_layer_kv=((initial, initial + 100.0),),
        local_positions=(0, 1, 2, 3, 4),
        initial_token_id=9,
        initial_logical_position=5,
        historical_layer_kv={},
        local_context_length=4,
        block_size=2,
        maximum_new_tokens=2,
    )

    assert result.token_ids == (1, 1)
    assert observed_past_lengths == [5, 4]
    assert observed_mask_lengths == [6, 5]


def test_streaming_state_comes_from_one_block_aligned_collector(monkeypatch):
    calls = []

    class FakeCollector:
        def __init__(
            self,
            _bundle,
            *,
            local_context_length,
            block_size,
            residual_layer,
            query_summary_length,
        ):
            calls.append(
                (
                    "init",
                    local_context_length,
                    block_size,
                    residual_layer,
                    query_summary_length,
                )
            )

        def collect(self, _record, plan, *, on_block_ready, on_evict):
            calls.append(("collect", plan.sample_id))
            for block in plan.candidate_blocks:
                summary = torch.full((3,), block.start_position + 0.5)
                on_block_ready(block, summary)
                key = torch.tensor(
                    [float(block.start_position), float(block.start_position + 1)]
                ).view(1, 1, 2, 1)
                on_evict(
                    SimpleNamespace(
                        block=block,
                        logical_positions=tuple(
                            range(block.start_position, block.end_position)
                        ),
                        layer_kv=((key, key + 100.0),),
                    )
                )
            local = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
            return SimpleNamespace(
                query_summary=torch.tensor([9.5, 9.5, 9.5]),
                local_layer_kv=((local, local + 100.0),),
                local_positions=(6, 7, 8, 9),
                forward_calls=5,
                forwarded_tokens=10,
                evicted_blocks=3,
                evicted_tokens=6,
                maximum_forward_context_length=6,
            )

    monkeypatch.setattr(
        "block_probability_router.streaming_collection."
        "BlockAlignedRollingContextCollector",
        FakeCollector,
    )
    candidates = tuple(
        BlockRange(f"s:block:{start:09d}-{start + 2:09d}", start, start + 2)
        for start in (0, 2, 4)
    )
    plan = RetrievalPlan(
        sample_id="sample",
        sequence_id="s",
        retrieval_position=9,
        first_future_position_affected_by_retrieval=10,
        future_horizon_length=1,
        local_context_start=6,
        local_context_end=10,
        candidate_blocks=candidates,
    )

    state = collect_streaming_student_state(
        SimpleNamespace(),
        SequenceRecord("s", tuple(range(12)), {}),
        plan,
        StudentCollectionConfig(
            local_context_length=4,
            residual_layer=0,
            query_summary="mean",
            query_summary_length=2,
        ),
        block_size=2,
        capture_layers=(0,),
    )

    assert calls == [("init", 4, 2, 0, 2), ("collect", "sample")]
    assert state.query_summary.tolist() == [9.5, 9.5, 9.5]
    assert state.block_summaries[:, 0].tolist() == [0.5, 2.5, 4.5]
    assert state.local_positions == (6, 7, 8, 9)
    assert sorted(state.block_layer_kv) == [0, 1, 2]
    assert state.block_layer_kv[2][0][0].flatten().tolist() == [4.0, 5.0]


def test_full_context_replay_state_is_cut_from_one_shared_prefix(monkeypatch):
    observed = []

    def fake_forward(
        _bundle,
        token_ids,
        logical_positions,
        *,
        past_key_values,
        use_cache,
        output_hidden_states,
        logical_cache_position,
        **_kwargs,
    ):
        assert use_cache and output_hidden_states and logical_cache_position
        positions = tuple(logical_positions)
        past = past_key_values[0][0]
        observed.append((positions, int(past.shape[2])))
        current = torch.tensor(positions, dtype=torch.float32).view(1, 1, -1, 1)
        key = torch.cat((past, current), dim=2)
        return SimpleNamespace(
            past_key_values=((key, key + 100.0),),
            selected_hidden=current[:, 0, :, :],
        )

    empty = torch.empty((1, 1, 0, 1), dtype=torch.float32)
    monkeypatch.setattr(
        "block_probability_router.full_context_replay.new_full_dynamic_cache",
        lambda: ((empty, empty),),
    )
    monkeypatch.setattr(
        "block_probability_router.full_context_replay.forward_tokens", fake_forward
    )
    monkeypatch.setattr(
        "block_probability_router.full_context_replay.cache_from_layer_kv", tuple
    )
    monkeypatch.setattr(
        "block_probability_router.full_context_replay.layer_kv_from_cache", tuple
    )
    monkeypatch.setattr(
        "block_probability_router.full_context_replay.hidden_state_at_layer",
        lambda output, _layer: output.selected_hidden,
    )
    candidates = tuple(
        BlockRange(f"s:block:{start:09d}-{start + 2:09d}", start, start + 2)
        for start in (0, 2, 4)
    )
    plan = RetrievalPlan(
        sample_id="sample",
        sequence_id="s",
        retrieval_position=9,
        first_future_position_affected_by_retrieval=10,
        future_horizon_length=1,
        local_context_start=6,
        local_context_end=10,
        candidate_blocks=candidates,
    )

    state = collect_full_context_replay_state(
        SimpleNamespace(),
        SequenceRecord("s", tuple(range(12)), {}),
        plan,
        StudentCollectionConfig(
            local_context_length=4,
            residual_layer=0,
            query_summary="mean",
            query_summary_length=2,
        ),
        block_size=2,
        prefill_chunk_size=3,
        capture_layers=(0,),
    )

    assert FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL.endswith("posthoc_block_cut_v2")
    assert observed == [
        ((0, 1, 2), 0),
        ((3, 4, 5), 3),
        ((6, 7, 8), 6),
        ((9,), 9),
    ]
    assert state.query_summary.tolist() == [8.5]
    assert state.block_summaries[:, 0].tolist() == [0.5, 2.5, 4.5]
    assert state.local_positions == (6, 7, 8, 9)
    assert state.local_layer_kv[0][0].flatten().tolist() == [6.0, 7.0, 8.0, 9.0]
    assert state.block_layer_kv[2][0][0].flatten().tolist() == [4.0, 5.0]
    assert state.forward_calls == 4
    assert state.forwarded_tokens == 10


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
