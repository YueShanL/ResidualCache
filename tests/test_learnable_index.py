from __future__ import annotations

import json

import pytest
import torch
import learnable_index.trainer as trainer_module

from learnable_index.config import (
    AttentionAggregationConfig,
    LossConfig,
    RouterConfig,
    TrainConfig,
)
from learnable_index.contracts import BlockRange, RetrievalSample
from learnable_index.data import (
    RetrievalDataset,
    collate_retrieval_samples,
    load_dataset,
    save_dataset,
    split_dataset,
)
from learnable_index.losses import router_loss
from learnable_index.metrics import MetricAccumulator, update_router_metrics
from learnable_index.model import LearnableBlockIndex
from learnable_index.prepare_wikitext import (
    _iter_arrow_texts,
    _iter_hf_texts,
    build_wikitext_sequences,
    iter_wikitext_articles,
)
from learnable_index.prepare_convomem import (
    ConvoMemExample,
    _split_for_source,
    build_convomem_long_sequences,
)
from learnable_index.prepare_wildchat import (
    _split_for_row,
    build_wildchat_sequences,
)
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from learnable_index.synthetic import make_synthetic_samples
from learnable_index.targets import aggregate_teacher_attention
from learnable_index.trainer import fit_router, load_checkpoint


class _WhitespaceTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        tokens = [101]
        tokens.extend(range(200, 200 + len(text.split())))
        return type("Encoding", (), {"input_ids": tokens})()


class _CharacterChatTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"<bos><user>{messages[0]['content']}</user><assistant>"

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        token_ids = [10 + ord(character) for character in text]
        payload = {"input_ids": token_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return type("Encoding", (), payload)()


class _WildChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is False
        rendered = "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if tokenize:
            return [10 + ord(character) for character in rendered]
        return rendered


def test_wikitext_arrow_reader_projects_text_column(tmp_path):
    import pyarrow as pa

    path = tmp_path / "wikitext-train.arrow"
    schema = pa.schema([("text", pa.string()), ("unused", pa.int64())])
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(
                pa.record_batch(
                    [pa.array(["first", "second"]), pa.array([1, 2])],
                    schema=schema,
                )
            )

    assert list(_iter_arrow_texts([path])) == ["first", "second"]


def test_wikitext_hf_reader_iterates_text_batches_without_materializing_split():
    class FakeDataset:
        column_names = ["text", "unused"]

        def iter(self, *, batch_size):
            assert batch_size == 2
            yield {"text": ["first", "second"], "unused": [1, 2]}
            yield {"text": ["third"], "unused": [3]}

    assert list(_iter_hf_texts(FakeDataset(), batch_size=2)) == [
        "first",
        "second",
        "third",
    ]


def test_wikitext_preparation_preserves_article_boundaries_and_exact_length():
    rows = [
        "orphan preamble\n",
        " = First article = \n",
        "one two three four five six\n",
        " = = Subheading = = \n",
        "seven eight\n",
        " = Second article = \n",
        "alpha beta gamma delta epsilon zeta eta theta\n",
    ]
    articles = list(iter_wikitext_articles(rows))
    assert [(article.article_index, article.title) for article in articles] == [
        (0, "First article"),
        (1, "Second article"),
    ]
    assert "Subheading" in articles[0].text
    assert "Second article" not in articles[0].text

    records = build_wikitext_sequences(
        articles,
        _WhitespaceTokenizer(),
        sequence_length=6,
        sequence_count=2,
        article_stride=1,
        seed=13,
        split="validation",
    )
    assert len(records) == 2
    assert all(len(record["token_ids"]) == 6 for record in records)
    assert len({record["article_index"] for record in records}) == 2
    assert all("-validation-" in record["sequence_id"] for record in records)


def test_wildchat_preparation_keeps_native_long_conversation_prefix():
    row = {
        "conversation_hash": "wildchat-test",
        "turn": 10,
        "conversation": [
            {"role": role, "content": f"turn {index} " + "context " * 8}
            for index in range(20)
            for role in ("user", "assistant")
        ],
    }
    split = _split_for_row(0, 13)
    records = build_wildchat_sequences(
        [row],
        _WildChatTokenizer(),
        split=split,
        sequence_length=128,
        sequence_count=1,
        seed=13,
        minimum_turns=10,
    )
    record = records[0]
    assert len(record["token_ids"]) == 128
    assert record["source"] == "allenai/WildChat-1M"
    assert record["turn_count"] == 10
    assert record["message_count"] == 40
    assert record["original_token_count"] > 128
    assert record["sequence_id"] == f"wildchat-{split}-000000000"


def test_convomem_split_groups_same_profile_across_categories():
    assert _split_for_source("stable_evidence/2_evidence/profile.json", 13) == (
        _split_for_source("changing_evidence/5_evidence/profile.json", 13)
    )


def test_convomem_synthesis_inserts_exact_distractor_distance_and_answer_point():
    source_file = "stable_evidence/2_evidence/profile.json"
    split = _split_for_source(source_file, 13)
    examples = [
        ConvoMemExample(
            example_id=f"example-{index}",
            source_file=source_file,
            source_item_index=index,
            category="stable_evidence",
            question=f"What is fact {index}?",
            answer=f"answer-{index}",
            evidence_context=f"User: evidence-{index}",
            conversation_context=f"User: unrelated-conversation-{index}",
        )
        for index in range(4)
    ]
    tokenizer = _CharacterChatTokenizer()
    rows = build_convomem_long_sequences(
        examples,
        tokenizer,
        split=split,
        sequence_length=512,
        sequence_count=2,
        seed=13,
        maximum_answer_tokens=32,
        maximum_future_horizon=16,
    )
    assert all(len(row["token_ids"]) == 512 for row in rows)
    assert all(row["distractor_token_count"] > 0 for row in rows)
    assert all(row["evidence_to_answer_distance_tokens"] > 256 for row in rows)
    assert all(row["split_group_id"] == "convomem-profile:profile" for row in rows)
    for row in rows:
        answer_start = row["answer_start_position"]
        assert row["retrieval_points"][0]["retrieval_position"] == answer_start - 2
        record = SequenceRecord(row["sequence_id"], tuple(row["token_ids"]), row)
        plans = build_retrieval_plans(
            record,
            PlanConfig(
                local_context_length=64,
                block_size=16,
                future_horizon_length=16,
                retrieval_interval=32,
                minimum_candidate_blocks=2,
                retrieval_point_policy="metadata",
            ),
        )
        assert len(plans) == 1
        assert plans[0].future_horizon_length == len(row["answer_token_ids"])
        assert len(plans[0].candidate_blocks) > 16


def test_convomem_stratified_random_placement_is_neutral_and_retrievable():
    source_file = "stable_evidence/2_evidence/profile.json"
    split = _split_for_source(source_file, 13)
    examples = [
        ConvoMemExample(
            example_id=f"example-{index}",
            source_file=source_file,
            source_item_index=index,
            category="stable_evidence",
            question=f"What is fact {index}?",
            answer=f"answer-{index}",
            evidence_context=f"User: evidence-{index}",
            conversation_context=f"User: unrelated-conversation-{index}",
        )
        for index in range(12)
    ]
    tokenizer = _CharacterChatTokenizer()
    rows = build_convomem_long_sequences(
        examples,
        tokenizer,
        split=split,
        sequence_length=768,
        sequence_count=8,
        seed=13,
        sampling_seed=97,
        maximum_answer_tokens=32,
        maximum_future_horizon=16,
        evidence_placement="stratified_random",
        evidence_placement_bins=4,
        placement_block_size=16,
        retrieval_local_context_length=64,
    )
    assert [row["evidence_placement_bin"] for row in rows] == [0, 1, 2, 3] * 2
    assert len({tuple(row["target_memory_chunk_range"]) for row in rows}) >= 4
    mean_start_by_bin = [
        sum(
            row["target_memory_chunk_range"][0]
            for row in rows
            if row["evidence_placement_bin"] == bin_index
        )
        / 2
        for bin_index in range(4)
    ]
    assert mean_start_by_bin == sorted(mean_start_by_bin)
    for row in rows:
        target_start, target_end = row["target_memory_chunk_range"]
        assert target_start % 16 == 0
        candidate_history_end = (
            row["answer_start_position"] - 1 - 64
        )
        containing_block_end = ((target_end + 15) // 16) * 16
        assert containing_block_end <= candidate_history_end
        assert all(
            target_end <= start or target_start >= end
            for start, end in row["distractor_token_ranges"]
            if start < end
        )
        rendered = "".join(chr(token - 10) for token in row["token_ids"][:-1])
        assert "Relevant earlier conversation" not in rendered
        assert "Intervening conversation" not in rendered
        assert rendered.count("Memory conversation:") >= 2
        record = SequenceRecord(row["sequence_id"], tuple(row["token_ids"]), row)
        plan = build_retrieval_plans(
            record,
            PlanConfig(
                local_context_length=64,
                block_size=16,
                future_horizon_length=16,
                retrieval_interval=32,
                minimum_candidate_blocks=2,
                retrieval_point_policy="metadata",
            ),
        )[0]
        candidate_indices = {
                block.start_position // 16 for block in plan.candidate_blocks
        }
        assert set(row["evidence_block_indices"]) <= candidate_indices


def test_teacher_attention_aggregation_preserves_absolute_and_conditional_mass():
    blocks = (
        BlockRange("a", 0, 2),
        BlockRange("b", 2, 5),
    )
    # Two future queries. Local keys at logical positions 5 and 6 are not
    # candidates and therefore remain outside the historical mass.
    attention = torch.tensor(
        [[[[0.10, 0.20, 0.05, 0.05, 0.10, 0.20, 0.30],
           [0.20, 0.10, 0.10, 0.10, 0.10, 0.20, 0.20]]]],
        dtype=torch.float32,
    )
    target = aggregate_teacher_attention(
        attention,
        torch.arange(7),
        blocks,
        AttentionAggregationConfig(future_reduction="mean"),
    )

    assert target.absolute_block_mass.tolist() == pytest.approx([0.30, 0.25])
    assert float(target.total_historical_mass) == pytest.approx(0.55)
    assert target.conditional_block_distribution.sum().item() == pytest.approx(1.0)
    assert target.conditional_block_distribution.tolist() == pytest.approx(
        [0.30 / 0.55, 0.25 / 0.55]
    )
    torch.testing.assert_close(
        target.per_future_absolute_block_mass,
        torch.tensor([[0.30, 0.20], [0.30, 0.30]]),
    )
    assert target.per_layer_head_future_block_mass.shape == (1, 1, 2, 2)
    assert target.metadata["absolute_mass_preserved"] is True


def test_length_normalization_changes_only_distribution_basis():
    blocks = (BlockRange("short", 0, 1), BlockRange("long", 1, 3))
    attention = torch.tensor([[[[0.2, 0.2, 0.2, 0.4]]]])
    target = aggregate_teacher_attention(
        attention,
        torch.arange(4),
        blocks,
        AttentionAggregationConfig(length_normalize_blocks=True),
    )

    assert target.absolute_block_mass.tolist() == pytest.approx([0.2, 0.4])
    assert float(target.total_historical_mass) == pytest.approx(0.6)
    assert target.distribution_basis_mass.tolist() == pytest.approx([0.2, 0.2])
    assert target.conditional_block_distribution.tolist() == pytest.approx([0.5, 0.5])


def test_contract_rejects_teacher_provenance_in_router_inputs():
    sample = make_synthetic_samples(sample_count=1)[0]
    sample.query_state_source = "teacher_full_context"

    with pytest.raises(ValueError, match="restricted student"):
        sample.validate()


def test_contract_rejects_candidate_overlapping_local_window():
    sample = make_synthetic_samples(sample_count=1)[0]
    sample.candidate_blocks = (
        BlockRange("leak", sample.local_context_start - 1, sample.local_context_start + 1),
    )
    sample.block_summaries = sample.block_summaries[:1]
    sample.absolute_teacher_block_mass = torch.tensor([float(sample.total_teacher_historical_mass)])
    sample.conditional_teacher_distribution = torch.ones(1)

    with pytest.raises(ValueError, match="outside the current local window"):
        sample.validate()


def test_masked_batch_and_router_ignore_padding_candidates():
    samples = make_synthetic_samples(sample_count=2, min_blocks=2, max_blocks=3)
    batch = collate_retrieval_samples(samples)
    router = LearnableBlockIndex(
        RouterConfig(residual_dim=samples[0].residual_dim, projection_dim=8, hidden_dim=16)
    )
    output = router(batch.query_summaries, batch.block_summaries, batch.candidate_mask)

    assert output.scores.shape == batch.candidate_mask.shape
    assert torch.all(output.scores[~batch.candidate_mask] < -1e20)


def test_variable_future_horizons_are_padded_and_masked_in_batch_metrics():
    samples = make_synthetic_samples(
        sample_count=2,
        min_blocks=2,
        max_blocks=2,
        future_horizon_length=3,
    )
    samples[0].future_horizon_length = 2
    samples[0].per_future_teacher_block_mass = (
        samples[0].absolute_teacher_block_mass.repeat(2, 1)
    )
    samples[1].per_future_teacher_block_mass = (
        samples[1].absolute_teacher_block_mass.repeat(3, 1)
    )
    batch = collate_retrieval_samples(samples)

    assert batch.per_future_teacher_block_mass.shape == (2, 3, 2)
    assert batch.per_future_mask.tolist() == [[True, True, False], [True, True, True]]
    router = LearnableBlockIndex(RouterConfig(residual_dim=samples[0].residual_dim))
    output = router(batch.query_summaries, batch.block_summaries, batch.candidate_mask)
    metrics = MetricAccumulator()
    update_router_metrics(metrics, output, batch, top_n=1)

    computed = metrics.compute()
    assert computed["historical_mass/distance_3"] == pytest.approx(
        float(samples[1].total_teacher_historical_mass)
    )


def test_zero_historical_mass_skips_conditional_ranking_loss():
    sample = make_synthetic_samples(sample_count=1, min_blocks=2, max_blocks=2)[0]
    sample.absolute_teacher_block_mass.zero_()
    sample.total_teacher_historical_mass.zero_()
    sample.conditional_teacher_distribution.zero_()
    batch = collate_retrieval_samples([sample])
    router = LearnableBlockIndex(RouterConfig(residual_dim=sample.residual_dim))
    output = router(batch.query_summaries, batch.block_summaries, batch.candidate_mask)
    loss = router_loss(output, batch, LossConfig())

    assert loss.conditional_sample_count == 0
    assert float(loss.conditional.detach()) == pytest.approx(0.0)
    assert torch.isfinite(loss.total)


def test_ranking_loss_accepts_summed_attention_mass():
    sample = make_synthetic_samples(sample_count=1, min_blocks=2, max_blocks=2)[0]
    sample.absolute_teacher_block_mass *= 3.0
    sample.total_teacher_historical_mass *= 3.0
    batch = collate_retrieval_samples([sample])
    router = LearnableBlockIndex(RouterConfig(residual_dim=sample.residual_dim))
    output = router(batch.query_summaries, batch.block_summaries, batch.candidate_mask)

    loss = router_loss(output, batch, LossConfig())

    assert torch.isfinite(loss.total)


def test_dataset_round_trip_preserves_contract(tmp_path):
    samples = make_synthetic_samples(sample_count=4, residual_dim=7)
    save_dataset(tmp_path / "dataset", samples, metadata={"kind": "test"})
    loaded, manifest = load_dataset(tmp_path / "dataset")

    assert len(loaded) == 4
    assert loaded.residual_dim == 7
    assert manifest["metadata"]["kind"] == "test"
    assert loaded[0].candidate_blocks == samples[0].candidate_blocks
    assert torch.equal(loaded[0].query_summary, samples[0].query_summary)


def test_validation_split_is_grouped_by_sequence_id():
    samples = make_synthetic_samples(sample_count=6, residual_dim=4)
    for index, sample in enumerate(samples):
        sample.sequence_id = f"sequence-group-{index // 2}"
    train, validation = split_dataset(RetrievalDataset(samples), 0.34, seed=13)

    assert validation is not None
    train_sequences = {sample.sequence_id for sample in train.samples}
    validation_sequences = {sample.sequence_id for sample in validation.samples}
    assert train_sequences.isdisjoint(validation_sequences)
    assert len(validation.samples) == 2


def test_validation_split_prefers_explicit_profile_group_id():
    samples = make_synthetic_samples(sample_count=8, residual_dim=4)
    for index, sample in enumerate(samples):
        sample.sequence_id = f"unique-sequence-{index}"
        sample.logical_position_metadata["split_group_id"] = f"profile-{index // 2}"
    train, validation = split_dataset(RetrievalDataset(samples), 0.25, seed=13)

    assert validation is not None
    train_profiles = {
        sample.logical_position_metadata["split_group_id"] for sample in train.samples
    }
    validation_profiles = {
        sample.logical_position_metadata["split_group_id"]
        for sample in validation.samples
    }
    assert train_profiles.isdisjoint(validation_profiles)
    assert len(validation.samples) == 2


def test_small_training_run_writes_reproducible_artifacts(tmp_path):
    dataset = RetrievalDataset(
        make_synthetic_samples(sample_count=24, residual_dim=6, min_blocks=2, max_blocks=4)
    )
    output_dir = tmp_path / "run"
    history = fit_router(
        dataset,
        output_dir,
        RouterConfig(residual_dim=6, projection_dim=6, hidden_dim=12, depth=1),
        LossConfig(),
        TrainConfig(
            epochs=2,
            batch_size=8,
            validation_fraction=0.25,
            learning_rate=1e-3,
            device="cpu",
            top_n=2,
        ),
    )

    assert len(history) == 2
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "final.pt").is_file()
    with (output_dir / "run_config.json").open(encoding="utf-8") as handle:
        run_config = json.load(handle)
    assert run_config["information_boundary"]["teacher_use"] == "labels_only"
    model, router_config, _, _, checkpoint = load_checkpoint(output_dir / "best.pt")
    assert isinstance(model, LearnableBlockIndex)
    assert router_config.residual_dim == 6
    assert checkpoint["epoch"] in {1, 2}
    assert checkpoint["format_version"] == 2
    assert checkpoint["model_kind"] == "query_key_only"


def test_training_stops_after_configured_non_improving_epochs(tmp_path, monkeypatch):
    dataset = RetrievalDataset(
        make_synthetic_samples(sample_count=12, residual_dim=4, min_blocks=2, max_blocks=3)
    )

    def constant_epoch(*args, **kwargs):
        return {"loss": 1.0, "conditional_loss": 1.0}

    monkeypatch.setattr(trainer_module, "_run_epoch", constant_epoch)
    output_dir = tmp_path / "early_stop"
    history = fit_router(
        dataset,
        output_dir,
        RouterConfig(residual_dim=4, projection_dim=4, hidden_dim=8, depth=1),
        LossConfig(),
        TrainConfig(
            epochs=10,
            early_stopping_patience=2,
            batch_size=4,
            validation_fraction=0.25,
            device="cpu",
        ),
    )

    assert [row["epoch"] for row in history] == [1, 2, 3]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == 3
    assert summary["maximum_epochs"] == 10
    assert summary["best_epoch"] == 1
    assert summary["early_stopping_patience"] == 2
    assert summary["stopped_early"] is True
    _, _, _, stored_train_config, checkpoint = load_checkpoint(output_dir / "final.pt")
    assert stored_train_config.early_stopping_patience == 2
    assert checkpoint["epoch"] == 3


def test_training_config_rejects_non_positive_early_stopping_patience():
    with pytest.raises(ValueError, match="early_stopping_patience"):
        TrainConfig(early_stopping_patience=0)


def test_legacy_demand_checkpoint_loads_as_query_key_only(tmp_path):
    config = RouterConfig(residual_dim=4, projection_dim=3, hidden_dim=5, depth=1)
    model = LearnableBlockIndex(config)
    legacy_path = tmp_path / "legacy.pt"
    state_dict = dict(model.state_dict())
    state_dict["demand_head.weight"] = torch.zeros(1, config.projection_dim)
    state_dict["demand_head.bias"] = torch.zeros(1)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": {},
            "router_config": config.__dict__,
            "loss_config": {
                "conditional_weight": 1.0,
                "demand_weight": 1.0,
                "demand_loss": "bce",
                "minimum_historical_mass": 1e-8,
            },
            "train_config": TrainConfig(epochs=1).__dict__,
            "epoch": 1,
            "metrics": {},
        },
        legacy_path,
    )

    loaded, _, loss_config, _, payload = load_checkpoint(legacy_path)

    assert isinstance(loaded, LearnableBlockIndex)
    assert not hasattr(loaded, "demand_head")
    assert loss_config.minimum_historical_mass == pytest.approx(1e-8)
    assert payload["format_version"] == 1
