from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

import residual_cache.answer_block_cache as answer_block_cache
from residual_cache.analysis import AnalysisConfig, analyze
from residual_cache.answer_block_cache import (
    _answer_block_end,
    _cat_block_caches,
    _encode_chat_prompt,
    _generate,
    _history_quality,
    _history_prompt,
    _make_prefix_causal_mask,
    _slice_cache,
)
from residual_cache.data_process import build_convomem_rows, build_lufy_rows, build_rows
from residual_cache.residual_collect import (
    _encode_pair,
    _generated_prefix_anchor_index,
    _generation_stop_token_ids,
    _positions,
)
from residual_cache.prompt_free_index import (
    PromptFreeAnalysisConfig,
    analyze_prompt_free_indices,
    _batched_dtw_distance_matrix,
    _capture_user_states,
    _encode_user_span,
    _evaluate_distances,
    _momentum_index,
    _parse_layer_spec,
)


def test_build_rows_covers_pre_research_conditions():
    rows = build_rows(max_facts=2, seed=1)
    conditions = {row["condition_id"] for row in rows}
    modes = {row["mode"] for row in rows}

    assert {"noise_0", "absent", "same_entity_different_value", "different_task_same_target", "conflict_latest"} <= conditions
    assert modes == {"natural", "canonical"}
    assert all({"prompt_id", "task_id", "target_fact_id", "condition_id", "prompt", "answer"} <= set(row) for row in rows)


def test_prompt_free_chat_span_excludes_template_tokens():
    class Encoding:
        def __init__(self, text):
            self.input_ids = [ord(char) for char in text]
            self.offset_mapping = [(index, index + 1) for index in range(len(text))]

    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            assert add_generation_prompt
            return f"<user>{messages[0]['content']}</user><assistant>"

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            assert not add_special_tokens
            assert return_offsets_mapping
            return Encoding(text)

    formatted, input_ids, positions = _encode_user_span(
        FakeTokenizer(),
        "hello",
        raw_prompt=False,
    )

    assert "".join(chr(input_ids[index]) for index in positions) == "hello"
    assert len(input_ids) == len(formatted)
    assert len(input_ids) - len(positions) == len("<user></user><assistant>")


def test_prompt_free_momentum_is_first_difference_and_resampled():
    states = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [3.0, 6.0],
            ]
        ]
    )

    momentum = _momentum_index(torch, states, points=2, projection=None)

    assert momentum.shape == (1, 2, 2)
    assert momentum[0].tolist() == [[1.0, 2.0], [2.0, 4.0]]


def test_prompt_free_capture_pads_layer_specific_q_widths():
    class FakeQ(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width

        def forward(self, hidden):
            return hidden[..., : self.width]

    class FakeAttention(torch.nn.Module):
        def __init__(self, q_width):
            super().__init__()
            self.q_proj = FakeQ(q_width)

        def forward(self, hidden):
            self.q_proj(hidden)
            return hidden

    class FakeLayer(torch.nn.Module):
        def __init__(self, q_width):
            super().__init__()
            self.self_attn = FakeAttention(q_width)

        def forward(self, hidden):
            return self.self_attn(hidden)

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer(2), FakeLayer(4)])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBackbone()

        def get_base_model(self):
            return self

        def forward(self, input_ids, use_cache=False):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            for layer in self.model.layers:
                hidden = layer(hidden)
            return hidden

    states, queries = _capture_user_states(
        torch,
        FakeModel(),
        torch.tensor([[2, 3, 0]]),
        positions=[0, 1],
        layer_indices=[0, 1],
        state_target="block_output",
    )

    assert states.shape == (2, 2, 4)
    assert queries.shape == (2, 2, 4)
    assert torch.all(queries[0, :, 2:] == 0)
    assert torch.any(queries[1, :, 2:] != 0)


def test_batched_dtw_prefers_matching_momentum_trajectory():
    query = torch.tensor(
        [[
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]]
    )
    refs = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ]
    )

    distances = _batched_dtw_distance_matrix(torch, query, refs)

    assert distances.shape == (1, 2)
    assert distances[0, 0] < distances[0, 1]


def test_prompt_free_knn_metrics_use_lower_distance():
    query_rows = [
        {"prompt_id": "q-a", "target_fact_id": "a"},
        {"prompt_id": "q-b", "target_fact_id": "b"},
    ]
    ref_rows = [
        {"prompt_id": "r-a", "target_fact_id": "a"},
        {"prompt_id": "r-b", "target_fact_id": "b"},
    ]
    distances = torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    report, neighbors = _evaluate_distances(distances, query_rows, ref_rows, top_k=1)

    assert report["top1_accuracy"] == 1.0
    assert report["top_k_recall"] == 1.0
    assert report["distance_margin"] > 0
    assert all(row["correct_rank"] == 1 for row in neighbors)


def test_prompt_free_layer_spec_supports_ranges():
    assert _parse_layer_spec("1,3-5", 7) == [1, 3, 4, 5]
    assert _parse_layer_spec("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="out of range"):
        _parse_layer_spec("3", 3)


def test_prompt_free_analysis_writes_three_method_comparison(tmp_path):
    collect_dir = tmp_path / "collect"
    tensor_dir = collect_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    metadata = []
    rows = [
        ("a", "fact_reference", [1.0, 0.0]),
        ("a", "question_query", [1.0, 0.0]),
        ("b", "fact_reference", [0.0, 1.0]),
        ("b", "question_query", [0.0, 1.0]),
    ]
    for index, (fact_id, condition, vector) in enumerate(rows):
        base = torch.tensor(vector, dtype=torch.float32)
        trajectory = torch.stack([base, base.roll(1), -base])
        tensor_path = tensor_dir / f"{index}.pt"
        torch.save(
            {
                "layer_indices": [7],
                "state_target": "block_output",
                "state_mean": base.view(1, -1),
                "q_mean": base.view(1, -1),
                "momentum": trajectory.view(1, 3, -1),
                "input_ids": torch.tensor([1, 2, 3]),
                "user_positions": torch.tensor([0, 1, 2]),
            },
            tensor_path,
        )
        metadata.append(
            {
                "prompt_id": f"{condition}-{fact_id}",
                "suite": "convomem_query_to_fact",
                "condition_id": condition,
                "target_fact_id": fact_id,
                "knowledge_prompt": False,
                "tensor_path": str(tensor_path.relative_to(collect_dir)),
            }
        )
    with (collect_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(__import__("json").dumps(row) + "\n")

    output_dir = analyze_prompt_free_indices(
        PromptFreeAnalysisConfig(
            collect_dir=collect_dir,
            output_dir=tmp_path / "analysis",
            top_k=1,
            device="cpu",
        )
    )

    reports = [
        __import__("json").loads(line)
        for line in (output_dir / "layer_report.jsonl").read_text().splitlines()
    ]
    summary = __import__("json").loads((output_dir / "summary.json").read_text())
    assert {row["method"] for row in reports} == {
        "momentum_dtw",
        "mean_state_cosine",
        "mean_q_cosine",
    }
    assert all(row["layer"] == 7 for row in reports)
    assert all(row["top1_accuracy"] == 1.0 for row in reports)
    assert set(summary["best_by_method"]) == {
        "momentum_dtw",
        "mean_state_cosine",
        "mean_q_cosine",
    }


def test_build_lufy_rows_creates_shift_suite():
    qa = [
        {
            "user_name": "Akane",
            "conversation_id": 1,
            "question": "What is Akane studying?",
            "answer": "Meiosis.",
            "evidence_turn_ids": ["D1:2"],
        }
    ]
    turns = [
        {"user_name": "Akane", "turn_id": "D1:1", "role": "assistant", "content": "What do you study?"},
        {"user_name": "Akane", "turn_id": "D1:2", "role": "user", "content": "I study meiosis."},
    ]

    rows = build_lufy_rows(qa, turns, max_qa=1, seed=1, max_context_turns=2)

    assert {row["suite"] for row in rows} == {"lufy_shift"}
    assert {row["condition_id"] for row in rows} == {"evidence_only", "recent_context", "noisy_context"}


def test_build_convomem_rows_creates_query_to_fact_pairs(tmp_path):
    root = tmp_path / "evidence_questions" / "changing_evidence" / "2_evidence"
    root.mkdir(parents=True)
    payload = {
        "evidence_items": [
            {
                "question": "What book is my book club reading?",
                "answer": "Dune.",
                "message_evidences": [{"speaker": "User", "text": "My book club is reading Dune."}],
                "conversations": [
                    {
                        "containsEvidence": True,
                        "messages": [
                            {"speaker": "User", "text": "A filler turn."},
                            {"speaker": "User", "text": "My book club is reading Dune."},
                        ],
                    }
                ],
            },
            {
                "question": "What book is my book club reading?",
                "answer": "Project Hail Mary.",
                "message_evidences": [{"speaker": "User", "text": "My book club switched to Project Hail Mary."}],
                "conversations": [
                    {
                        "containsEvidence": True,
                        "messages": [
                            {"speaker": "User", "text": "My book club switched to Project Hail Mary."}
                        ],
                    }
                ],
            },
            {
                "question": "What is my dentist appointment?",
                "answer": "Tuesday.",
                "message_evidences": [{"speaker": "User", "text": "The dentist is Tuesday."}],
            },
        ]
    }
    (root / "sample.json").write_text(__import__("json").dumps(payload), encoding="utf-8")

    rows = build_convomem_rows(tmp_path, max_facts=10, seed=1, max_context_turns=8)

    assert len(rows) == 6
    assert {row["suite"] for row in rows} == {"convomem_query_to_fact"}
    assert {row["condition_id"] for row in rows} == {"fact_reference", "question_query"}
    assert all(row["raw_prompt"] for row in rows)
    assert len({row["target_fact_id"] for row in rows}) == 3
    for fact_id in {row["target_fact_id"] for row in rows}:
        assert {row["condition_id"] for row in rows if row["target_fact_id"] == fact_id} == {
            "fact_reference",
            "question_query",
        }
    assert any(row["specific_question"].startswith("For the sample user,") for row in rows)
    fact_rows = [row for row in rows if row["condition_id"] == "fact_reference"]
    query_rows = [row for row in rows if row["condition_id"] == "question_query"]
    assert all("Conversation evidence:" in row["prompt"] for row in fact_rows)
    assert all("Question:" not in row["prompt"] for row in fact_rows)
    assert all("Conversation evidence:" not in row["prompt"] for row in query_rows)
    assert all("Question:" in row["prompt"] for row in query_rows)
    assert all("Conversation evidence:" in row["history_prompt"] for row in query_rows)
    assert all(row["specific_question"] in row["history_prompt"] for row in query_rows)
    assert all("Conversation evidence:" in _history_prompt(row) for row in query_rows)

    templated_rows = build_convomem_rows(
        tmp_path,
        max_facts=10,
        seed=1,
        max_context_turns=8,
        use_chat_template=True,
    )
    assert all(not row["raw_prompt"] for row in templated_rows)

    knowledge_rows = build_convomem_rows(
        tmp_path,
        max_facts=10,
        seed=1,
        max_context_turns=8,
        use_chat_template=True,
        use_knowledge_prompt=True,
    )
    assert all(row["prompt"].startswith("Extract only the entity and the relevant fact") for row in knowledge_rows)
    assert all("FACT: <entity>, <fact>" in row["prompt"] for row in knowledge_rows)
    assert all("RELEVANT_KNOWLEDGE:" not in row["prompt"] for row in knowledge_rows)
    assert all("<value-or-?>" not in row["prompt"] for row in knowledge_rows)
    assert all("at most 18 words" in row["prompt"] for row in knowledge_rows)
    assert all("Do not stop after the `FACT:` line" in row["prompt"] for row in knowledge_rows)
    assert all(row["knowledge_prompt"] for row in knowledge_rows)
    knowledge_fact_rows = [row for row in knowledge_rows if row["condition_id"] == "fact_reference"]
    knowledge_query_rows = [row for row in knowledge_rows if row["condition_id"] == "question_query"]
    assert all("Conversation evidence:" in row["prompt"] for row in knowledge_fact_rows)
    assert all("Conversation evidence:" not in row["prompt"].split("INPUT:\n", 1)[1] for row in knowledge_query_rows)
    assert all(row["history_prompt"].startswith("Extract only the entity") for row in knowledge_query_rows)
    assert all(answer_block_cache._query_prompt(row) == row["query_prompt"] for row in knowledge_query_rows)
    assert all("Conversation evidence:" in _history_prompt(row) for row in knowledge_query_rows)
    assert all("Return only the answer" not in _history_prompt(row) for row in knowledge_query_rows)


def test_chat_templated_final_prompt_uses_last_processed_prompt_token():
    class Tokenized:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            assert add_generation_prompt
            return f"<bos><user>{messages[0]['content']}</user><model>"

        def __call__(self, text, add_special_tokens=False):
            return Tokenized([ord(char) for char in text])

    row = {
        "prompt_id": "templated",
        "prompt": "Question?",
        "answer": "Answer.",
        "raw_prompt": False,
    }
    prompt_text, prompt_ids, _answer_ids, full_ids = _encode_pair(FakeTokenizer(), row)
    positions, answer_indices = _positions(len(prompt_ids), len(full_ids), "final_prompt")

    assert prompt_text.endswith("<model>")
    assert positions == [len(prompt_ids) - 1]
    assert prompt_ids[positions[0]] == ord(">")
    assert answer_indices == []


def test_generation_stops_on_eos_and_chat_end_of_turn():
    class FakeTokenizer:
        eos_token_id = 1
        eot_token_id = 106

    assert _generation_stop_token_ids(FakeTokenizer()) == {1, 106}


def test_generated_prefix_anchor_is_token_that_completes_prefix():
    class FakeTokenizer:
        pieces = {10: "FA", 11: "CT", 12: ":", 13: " Alice"}

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(self.pieces[token_id] for token_id in token_ids)

    generated_ids = [10, 11, 12, 13]
    assert _generated_prefix_anchor_index(FakeTokenizer(), generated_ids) == 2
    with pytest.raises(RuntimeError, match="did not generate required prefix"):
        _generated_prefix_anchor_index(FakeTokenizer(), [10, 11])


def test_analysis_reports_retrieval(tmp_path):
    collect_dir = tmp_path / "collect"
    tensor_dir = collect_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    metadata = []
    for index, fact_id in enumerate(["a_blue", "a_blue", "b_green", "b_green"]):
        state = torch.zeros(2, 2, 4)
        state[:, :, 0 if fact_id == "a_blue" else 1] = 1.0
        path = tensor_dir / f"{index}.pt"
        torch.save(
            {
                "states": {"pre_attn": state, "q": state, "block_output": state},
                "final_prompt_index": 0,
                "answer_indices": [1],
            },
            path,
        )
        metadata.append(
            {
                "prompt_id": str(index),
                "task_id": "recall_color",
                "target_fact_id": fact_id,
                "condition_id": "noise_0",
                "mode": "canonical",
                "entity": fact_id[0],
                "conflict": False,
                "tensor_path": str(path.relative_to(collect_dir)),
            }
        )
    with (collect_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(__import__("json").dumps(row) + "\n")

    out = analyze(AnalysisConfig(collect_dir=collect_dir, top_k=1))

    assert (out / "layer_similarity_report.jsonl").exists()
    assert (out / "query_to_fact_report.json").exists()
    recommendation = __import__("json").loads((out / "recommendation.json").read_text())
    assert recommendation["best_top1_same_target_recall"] == 1.0


def test_analysis_reports_query_to_fact(tmp_path):
    collect_dir = tmp_path / "collect"
    tensor_dir = collect_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    metadata = []
    rows = [
        ("a_fact", "fact_reference", [1.0, 0.0]),
        ("a_fact", "question_query", [1.0, 0.0]),
        ("b_fact", "fact_reference", [0.0, 1.0]),
        ("b_fact", "question_query", [0.0, 1.0]),
    ]
    for index, (fact_id, condition, values) in enumerate(rows):
        state = torch.tensor(values, dtype=torch.float32).view(1, 1, 2).repeat(2, 1, 1)
        path = tensor_dir / f"{index}.pt"
        torch.save(
            {
                "states": {"pre_attn": state, "q": state, "block_output": state},
                "final_prompt_index": 0,
                "answer_indices": [],
            },
            path,
        )
        metadata.append(
            {
                "prompt_id": str(index),
                "task_id": "convomem_memory_qa",
                "suite": "convomem_query_to_fact",
                "target_fact_id": fact_id,
                "condition_id": condition,
                "question_key": fact_id,
                "mode": "natural",
                "entity": fact_id,
                "conflict": False,
                "tensor_path": str(path.relative_to(collect_dir)),
            }
        )
    with (collect_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(__import__("json").dumps(row) + "\n")

    out = analyze(AnalysisConfig(collect_dir=collect_dir, top_k=1))

    query_report = __import__("json").loads((out / "query_to_fact_report.json").read_text())
    assert query_report
    assert max(row["top1_query_to_fact_accuracy"] for row in query_report) == 1.0


def test_answer_block_end_caps_at_block_size_and_eos():
    assert _answer_block_end([10, 11, 12, 13], block_size=3, eos_token_id=None) == 3
    assert _answer_block_end([10, 99, 12, 13], block_size=4, eos_token_id=99) == 2
    assert _answer_block_end([10, 11, 12, 99], block_size=3, eos_token_id=99) == 3
    assert _answer_block_end([], block_size=3, eos_token_id=99) == 0


def test_history_quality_rejects_prompt_echo():
    bad = _history_quality("Question: What is it?\nAnswer: Question: What is it?", "It is Tuesday.")
    good = _history_quality("It is Tuesday.", "It is Tuesday.")

    assert not bad["history_valid_payload"]
    assert bad["history_has_prompt_marker"]
    assert good["history_valid_payload"]


def test_chat_prompt_anchor_uses_last_text_token():
    class Tokenized:
        def __init__(self, text, offsets=False):
            self.input_ids = [ord(char) for char in text]
            if offsets:
                self.offset_mapping = [(index, index + 1) for index, _char in enumerate(text)]

    class FakeTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            return f"<user>{messages[0]['content']}</user><assistant>"

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            return Tokenized(text, return_offsets_mapping)

    tokens = _encode_chat_prompt(FakeTokenizer(), "abc")

    assert tokens.ids[tokens.anchor_index] == ord("c")
    assert tokens.suffix_start == tokens.anchor_index + 1
    assert tokens.ids[tokens.suffix_start] == ord("<")


def test_prefix_causal_mask_allows_sparse_prefix_and_causal_current_tokens():
    class FakeConfig:
        layer_types = ["full_attention"]

    class FakeModel(torch.nn.Module):
        config = FakeConfig()

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    mask_map = _make_prefix_causal_mask(
        torch,
        model=FakeModel(),
        q_len=3,
        prefix_len=2,
        current_kv_len=3,
        current_query_start=0,
    )
    mask = mask_map["full_attention"][0, 0]

    assert mask.shape == (3, 5)
    assert torch.all(mask[:, :2] == 0)
    assert mask[0, 2] == 0
    assert mask[0, 3] < -1e20
    assert mask[0, 4] < -1e20
    assert mask[1, 2] == 0
    assert mask[1, 3] == 0
    assert mask[1, 4] < -1e20
    assert torch.all(mask[2, 2:] == 0)


def test_slice_cache_maps_logical_positions_into_sliding_storage():
    class FakeLayer:
        def __init__(self, logical_values, logical_length):
            values = torch.tensor(logical_values, dtype=torch.float32).view(1, 1, -1, 1)
            self.keys = values
            self.values = values + 100
            self.logical_length = logical_length

        def get_seq_length(self):
            return self.logical_length

    class FakeCache:
        def __init__(self):
            self.layers = [
                FakeLayer([3, 4, 5, 6, 7], logical_length=8),
                FakeLayer(list(range(8)), logical_length=8),
            ]

        def __iter__(self):
            for layer in self.layers:
                yield layer.keys, layer.values, None

    sliced = _slice_cache(FakeCache(), 5, 8)

    assert len(sliced) == 2
    assert sliced[0][0].flatten().tolist() == [5, 6, 7]
    assert sliced[1][0].flatten().tolist() == [5, 6, 7]
    assert sliced[0][1].flatten().tolist() == [105, 106, 107]


def test_slice_cache_rejects_positions_rolled_out_of_sliding_storage():
    class FakeLayer:
        keys = torch.zeros(1, 1, 5, 1)
        values = torch.zeros(1, 1, 5, 1)

        def get_seq_length(self):
            return 8

    class FakeCache:
        layers = [FakeLayer()]

        def __iter__(self):
            yield self.layers[0].keys, self.layers[0].values, None

    with pytest.raises(RuntimeError, match="unavailable"):
        _slice_cache(FakeCache(), 1, 3)


def test_cat_block_caches_validates_and_merges_layer_lengths(tmp_path):
    def block(length):
        tensor = torch.zeros(1, 1, length, 2)
        return ((tensor, tensor.clone()), (tensor.clone(), tensor.clone()))

    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save({"past_key_values": block(2)}, first_path)
    torch.save({"past_key_values": block(3)}, second_path)

    merged = _cat_block_caches(torch, [first_path, second_path], torch.device("cpu"))

    assert merged.get_seq_length() == 5
    assert {layer.keys.shape[2] for layer in merged.layers} == {5}

    empty_path = tmp_path / "empty.pt"
    torch.save({"past_key_values": block(0)}, empty_path)
    with pytest.raises(RuntimeError, match="no cached tokens"):
        _cat_block_caches(torch, [empty_path], torch.device("cpu"))


def test_cat_block_caches_can_filter_generated_fact_line(tmp_path):
    class FakeTokenizer:
        pieces = {10: "FACT", 11: ":", 12: " entity", 13: "\n", 14: "answer", 15: "<eot>"}

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(
                "" if skip_special_tokens and token_id == 15 else self.pieces[token_id]
                for token_id in token_ids
            )

    token_ids = [10, 11, 12, 13, 14, 15]
    keys = torch.tensor(token_ids, dtype=torch.float32).view(1, 1, -1, 1)
    block_path = tmp_path / "fact_block.pt"
    torch.save({"past_key_values": ((keys, keys + 100),), "token_ids": token_ids}, block_path)

    merged = _cat_block_caches(
        torch,
        [block_path],
        torch.device("cpu"),
        tokenizer=FakeTokenizer(),
        block_metadata=[{"suffix_tokens": 0}],
        filter_fact_line=True,
    )

    assert merged.get_seq_length() == 2
    assert merged.layers[0].keys.flatten().tolist() == [14.0, 15.0]
    assert merged.layers[0].values.flatten().tolist() == [114.0, 115.0]


def test_generate_captures_first_block_then_continues_history(monkeypatch):
    class FakeLayer:
        def __init__(self, values):
            self.keys = values
            self.values = values + 100

        def get_seq_length(self):
            return int(self.keys.shape[2])

    class FakeCache:
        def __init__(self, values):
            self.layers = [FakeLayer(values)]

        def get_seq_length(self):
            return self.layers[0].get_seq_length()

        def append(self, value):
            token = torch.tensor([[[[value]]]], dtype=torch.float32)
            layer = self.layers[0]
            layer.keys = torch.cat([layer.keys, token], dim=2)
            layer.values = torch.cat([layer.values, token + 100], dim=2)

        def __iter__(self):
            layer = self.layers[0]
            yield layer.keys, layer.values, None

    class FakeOutput:
        def __init__(self, cache, next_token):
            self.past_key_values = cache
            self.logits = torch.full((1, 1, 128), -1000.0)
            self.logits[0, 0, next_token] = 1.0

    class FakeModel:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.weight

    class FakeTokenizer:
        eos_token_id = 99

        def decode(self, token_ids, skip_special_tokens=True):
            return " ".join(str(token_id) for token_id in token_ids)

    next_generated_token = 10

    def fake_forward(_torch, _model, input_ids, _start_position, past_key_values=None, **_kwargs):
        nonlocal next_generated_token
        if past_key_values is None:
            prompt_values = input_ids.float().view(1, 1, -1, 1)
            cache = FakeCache(prompt_values)
        else:
            cache = past_key_values
            cache.append(int(input_ids[0, 0]))
        output = FakeOutput(cache, next_generated_token)
        next_generated_token += 1
        return output, torch.zeros(1)

    monkeypatch.setattr(answer_block_cache, "_forward", fake_forward)

    result = _generate(
        torch,
        FakeTokenizer(),
        FakeModel(),
        [0, 1, 2],
        start_position=0,
        max_new_tokens=5,
        index_layer=0,
        capture_cache_prompt_start=3,
        capture_cache_generated_tokens=2,
    )

    assert result["generated_ids"] == [10, 11, 12, 13, 14]
    assert result["past_key_values"].get_seq_length() == 8
    assert result["captured_generated_tokens"] == 2
    assert result["captured_cache"][0][0].flatten().tolist() == [10, 11]


def test_dual_pipeline_appends_instruction_only_to_index_branch():
    from residual_cache.answer_block_cache import (
        _dual_branch_prompts,
        _dual_query_prompts,
    )

    row = {
        "history_prompt": "HISTORY CONTENT",
        "query_prompt": "QUERY CONTENT",
    }
    main_history, index_history = _dual_branch_prompts(row)
    main_query, index_query = _dual_query_prompts(row)

    assert main_history == "HISTORY CONTENT"
    assert main_query == "QUERY CONTENT"
    assert index_history.startswith("HISTORY CONTENT\n\n")
    assert index_query.startswith("QUERY CONTENT\n\n")
    assert "FACT: <entity>, <fact>" in index_history
    assert "FACT: <entity>, <fact>" in index_query
    assert "FACT:" not in main_history
    assert "FACT:" not in main_query


def test_clone_cache_creates_independent_branch():
    from residual_cache.answer_block_cache import _clone_cache

    class FakeCache:
        def __init__(self):
            self.tensor = torch.tensor([1.0])

    original = FakeCache()
    cloned = _clone_cache(original)
    cloned.tensor[0] = 9.0

    assert original.tensor.tolist() == [1.0]
    assert cloned.tensor.tolist() == [9.0]


def test_generate_captures_index_after_actual_generated_prefix(monkeypatch):
    class FakeCache:
        def __init__(self, length):
            self.length = length

        def get_seq_length(self):
            return self.length

    class FakeOutput:
        def __init__(self, cache, next_token):
            self.past_key_values = cache
            self.logits = torch.full((1, 1, 128), -1000.0)
            self.logits[0, 0, next_token] = 1.0

    class FakeModel:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.weight

    class FakeTokenizer:
        eos_token_id = 99
        pieces = {10: "FACT", 11: ":", 12: " Alice", 13: " fact"}

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(self.pieces.get(token_id, "") for token_id in token_ids)

    next_tokens = iter([10, 11, 12, 13, 13])
    captured_inputs = []

    def fake_forward(_torch, _model, input_ids, _start_position, past_key_values=None, capture_layer=None, **_kwargs):
        length = input_ids.shape[1] if past_key_values is None else past_key_values.length + input_ids.shape[1]
        if capture_layer is not None:
            captured_inputs.append(int(input_ids[0, -1]))
            index = torch.tensor([float(input_ids[0, -1])])
        else:
            index = None
        return FakeOutput(FakeCache(length), next(next_tokens)), index

    monkeypatch.setattr(answer_block_cache, "_forward", fake_forward)

    result = _generate(
        torch,
        FakeTokenizer(),
        FakeModel(),
        [1, 2],
        start_position=0,
        max_new_tokens=4,
        index_layer=0,
        capture_generated_prefix="FACT:",
    )

    assert result["generated_ids"][:3] == [10, 11, 12]
    assert result["index_generated_token_index"] == 1
    assert result["index_generated_prefix_text"] == "FACT:"
    assert result["index_vector"].tolist() == [11.0]
    assert captured_inputs == [11]


def test_attention_prefix_layout_tracks_filtered_gold_and_distractor_regions():
    from residual_cache.attention_analysis import _prefix_region_layout

    class FakeTokenizer:
        pieces = {
            1: "model",
            2: "\n",
            3: "FACT:",
            4: " gold",
            5: "\n",
            6: "answer",
            7: " other",
        }

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(self.pieces[token_id] for token_id in token_ids)

    metadata = [
        {
            "block_id": "gold",
            "target_fact_id": "fact-1",
            "suffix_tokens": 2,
            "token_ids": [1, 2, 3, 4, 5, 6],
        },
        {
            "block_id": "other",
            "target_fact_id": "fact-2",
            "suffix_tokens": 2,
            "token_ids": [1, 2, 3, 7, 5, 6],
        },
    ]

    regions, prefix_len = _prefix_region_layout(
        FakeTokenizer(),
        metadata,
        filter_fact_line=True,
        target_fact_id="fact-1",
    )

    assert prefix_len == 6
    assert regions["history"] == [(0, 3), (3, 6)]
    assert regions["history_suffix"] == [(0, 2), (3, 5)]
    assert regions["history_body"] == [(2, 3), (5, 6)]
    assert regions["gold_body"] == [(2, 3)]
    assert regions["distractor_body"] == [(5, 6)]
    assert "history_fact" not in regions


def test_attention_event_summary_reports_mass_and_uniform_enrichment():
    from residual_cache.attention_analysis import _summarize_events

    events = [
        {
            "key_len": 10,
            "regions": {
                "gold_body": {
                    "length": 2,
                    "head_mass": [0.4, 0.2],
                }
            },
        },
        {
            "key_len": 10,
            "regions": {
                "gold_body": {
                    "length": 2,
                    "head_mass": [0.2, 0.2],
                }
            },
        },
    ]

    summary = _summarize_events(events, [0, 1])
    gold = summary["regions"]["gold_body"]
    assert summary["events"] == 2
    assert gold["mean_mass"] == pytest.approx(0.25)
    assert gold["max_head_mass"] == pytest.approx(0.3)
    assert gold["mean_enrichment"] == pytest.approx(1.25)
    assert gold["heads_over_2x_uniform"] == 0


def test_distance_sweep_condition_plan_separates_gap_and_joint_shift():
    from residual_cache.distance_sweep import _condition_plan

    conditions = _condition_plan(
        gaps=(0, 512),
        joint_shifts=(0, 8192),
        control_gap=256,
    )

    assert conditions == [
        {
            "condition": "distance_sweep",
            "history_start_position": 0,
            "requested_gap": 0,
        },
        {
            "condition": "distance_sweep",
            "history_start_position": 0,
            "requested_gap": 512,
        },
        {
            "condition": "joint_shift_control",
            "history_start_position": 0,
            "requested_gap": 256,
        },
        {
            "condition": "joint_shift_control",
            "history_start_position": 8192,
            "requested_gap": 256,
        },
    ]


def test_generated_token_comparison_reports_first_divergence():
    from residual_cache.distance_sweep import (
        _generated_token_comparison,
    )

    comparison = _generated_token_comparison(
        [10, 11, 12],
        [10, 11, 99, 13],
    )

    assert not comparison["exact"]
    assert comparison["common_prefix_tokens"] == 2
    assert comparison["first_mismatch"] == 2
    assert comparison["continuous_token_at_mismatch"] == 12
    assert comparison["cache_split_token_at_mismatch"] == 99
    assert _generated_token_comparison(
        [1, 2],
        [1, 2],
    )["exact"]


def test_answer_body_keeps_full_output_without_fact_prefix():
    from residual_cache.distance_sweep import _answer_body

    text = "The sync is Wednesday at 11 AM."
    assert _answer_body(
        text,
        fact_prefixed=False,
    ) == text
    assert _answer_body(
        "FACT: sync time\nWednesday at 11 AM.",
        fact_prefixed=True,
    ) == "Wednesday at 11 AM."


def test_position_extrapolation_requires_explicit_opt_in():
    from residual_cache.distance_sweep import (
        _validate_position_plan,
    )

    assert not _validate_position_plan(
        planned_end=1024,
        position_limit=2048,
        allow_extrapolation=False,
    )
    with pytest.raises(
        ValueError,
        match="allow-position-extrapolation",
    ):
        _validate_position_plan(
            planned_end=4096,
            position_limit=2048,
            allow_extrapolation=False,
        )
    assert _validate_position_plan(
        planned_end=4096,
        position_limit=2048,
        allow_extrapolation=True,
    )


def test_stored_knowledge_prompts_use_current_instruction():
    from residual_cache.answer_block_cache import (
        _history_prompt,
        _query_prompt,
    )

    stale = (
        "OLD KNOWLEDGE INSTRUCTION\n\nINPUT:\n"
        "User profile: Alice\nQuestion: What changed?"
    )
    row = {
        "knowledge_prompt": True,
        "history_prompt": stale,
        "query_prompt": stale,
    }

    expected_start = (
        "Extract only the entity and the relevant fact.\n"
    )
    assert _history_prompt(row).startswith(expected_start)
    assert _query_prompt(row).startswith(expected_start)
    assert "OLD KNOWLEDGE INSTRUCTION" not in _history_prompt(
        row
    )
    assert _history_prompt(row).endswith(
        "User profile: Alice\nQuestion: What changed?"
    )


def test_continuous_turn_prefix_split_rejects_bos_in_continuation():
    from residual_cache.continuation_equivalence import (
        build_continuous_turn_tokens,
    )

    class FakeEncoding:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class FakeTokenizer:
        chat_template = "fake"
        bos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=True):
            return "answer"

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            return "multi"

        def __call__(self, text, add_special_tokens=False):
            return FakeEncoding([2, 10, 11, 2, 20])

    with pytest.raises(
        RuntimeError,
        match="starts with BOS",
    ):
        build_continuous_turn_tokens(
            FakeTokenizer(),
            history_user_content="history",
            history_prompt_ids=[2, 10],
            history_generated_ids=[11],
            query_user_content="query",
        )


def test_continuous_turn_prefix_split_returns_exact_non_bos_suffix():
    from residual_cache.continuation_equivalence import (
        build_continuous_turn_tokens,
    )

    class FakeEncoding:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class FakeTokenizer:
        chat_template = "fake"
        bos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=True):
            return "answer"

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            return "multi"

        def __call__(self, text, add_special_tokens=False):
            return FakeEncoding([2, 10, 11, 107, 20])

    tokens = build_continuous_turn_tokens(
        FakeTokenizer(),
        history_user_content="history",
        history_prompt_ids=[2, 10],
        history_generated_ids=[11],
        query_user_content="query",
    )

    assert tokens.history_ids == [2, 10, 11]
    assert tokens.continuation_ids == [107, 20]
    assert tokens.full_ids == [2, 10, 11, 107, 20]
