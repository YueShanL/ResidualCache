from __future__ import annotations

import json

import pytest
import torch

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
from learnable_index.model import LearnableBlockIndex
from learnable_index.prepare_wikitext import (
    _iter_arrow_texts,
    _iter_hf_texts,
    build_wikitext_sequences,
    iter_wikitext_articles,
)
from learnable_index.synthetic import make_synthetic_samples
from learnable_index.targets import aggregate_teacher_attention
from learnable_index.trainer import fit_router, load_checkpoint


class _WhitespaceTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        tokens = [101]
        tokens.extend(range(200, 200 + len(text.split())))
        return type("Encoding", (), {"input_ids": tokens})()


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
    assert output.demand_logits.shape == (2,)
    assert torch.all(output.scores[~batch.candidate_mask] < -1e20)


def test_zero_demand_skips_conditional_loss_but_keeps_demand_loss():
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
    assert torch.isfinite(loss.demand)


def test_mse_demand_loss_accepts_summed_attention_mass():
    sample = make_synthetic_samples(sample_count=1, min_blocks=2, max_blocks=2)[0]
    sample.absolute_teacher_block_mass *= 3.0
    sample.total_teacher_historical_mass *= 3.0
    batch = collate_retrieval_samples([sample])
    router = LearnableBlockIndex(RouterConfig(residual_dim=sample.residual_dim))
    output = router(batch.query_summaries, batch.block_summaries, batch.candidate_mask)

    loss = router_loss(output, batch, LossConfig(demand_loss="mse"))

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
