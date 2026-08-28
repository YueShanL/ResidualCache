from __future__ import annotations

import json

import pytest
import torch

from block_probability_router.config import (
    ProbabilityLossConfig,
    ProbabilityRouterConfig,
    ProbabilityTrainConfig,
)
from block_probability_router.losses import probability_router_loss
from block_probability_router.model import BlockProbabilityRouter
from block_probability_router.trainer import fit_router, load_checkpoint
from learnable_index.data import RetrievalDataset, collate_retrieval_samples
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from learnable_index.synthetic import make_synthetic_samples


def _model(residual_dim: int = 6) -> BlockProbabilityRouter:
    torch.manual_seed(13)
    return BlockProbabilityRouter(
        ProbabilityRouterConfig(
            residual_dim=residual_dim,
            feature_dim=5,
            hidden_dim=10,
            depth=1,
            dropout=0.0,
        )
    )


def test_explicit_memory_normalizer_matches_sum_of_positive_block_weights():
    model = _model()
    query = torch.randn(2, 6)
    keys = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    output = model(query, keys, mask)

    assert torch.all(output.weights[mask] > 0)
    assert torch.all(output.weights[~mask] == 0)
    assert torch.allclose(output.normalizer, output.weights.sum(dim=-1), atol=1e-5)
    assert torch.allclose(
        output.normalizer,
        torch.einsum("bf,bf->b", output.query_features, output.key_sum),
        atol=1e-5,
    )
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(output.probabilities[~mask] == 0)


@pytest.mark.parametrize("threshold", [0.05, 0.2, 0.4])
def test_weight_space_range_condition_is_exactly_probability_threshold(threshold: float):
    model = _model()
    mask = torch.tensor([[True, True, True, False]])
    output = model(torch.randn(1, 6), torch.randn(1, 4, 6), mask)

    selected = model.threshold_mask(output, mask, threshold)

    assert torch.equal(selected, mask & (output.probabilities > threshold))


def test_loss_is_direct_cross_entropy_of_normalized_positive_weights():
    samples = make_synthetic_samples(
        sample_count=3,
        residual_dim=6,
        min_blocks=2,
        max_blocks=4,
        seed=17,
    )
    batch = collate_retrieval_samples(samples)
    model = _model()
    output = model(batch.query_summaries, batch.block_summaries, batch.candidate_mask)

    loss = probability_router_loss(output, batch, ProbabilityLossConfig())
    manual = -(
        batch.conditional_teacher_distribution
        * output.log_probabilities.masked_fill(~batch.candidate_mask, 0.0)
    ).sum(dim=-1).mean()

    assert torch.allclose(loss.total, manual)
    loss.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_shared_plan_contract_excludes_current_visible_window_from_memory_candidates():
    record = SequenceRecord("sequence", tuple(range(800)), {})
    plans = build_retrieval_plans(
        record,
        PlanConfig(
            local_context_length=256,
            block_size=64,
            future_horizon_length=16,
            retrieval_interval=128,
            minimum_candidate_blocks=2,
        ),
    )

    assert plans
    for plan in plans:
        assert plan.future_start == plan.retrieval_position + 1
        assert plan.local_context_end == plan.retrieval_position + 1
        assert all(block.end_position <= plan.local_context_start for block in plan.candidate_blocks)


def test_small_training_run_writes_distinct_probability_router_checkpoint(tmp_path):
    dataset = RetrievalDataset(
        make_synthetic_samples(
            sample_count=20,
            residual_dim=6,
            min_blocks=2,
            max_blocks=4,
            seed=23,
        )
    )
    output_dir = tmp_path / "training"
    history = fit_router(
        dataset,
        output_dir,
        ProbabilityRouterConfig(
            residual_dim=6,
            feature_dim=6,
            hidden_dim=12,
            depth=1,
        ),
        ProbabilityLossConfig(),
        ProbabilityTrainConfig(
            epochs=2,
            early_stopping_patience=None,
            batch_size=5,
            validation_fraction=0.2,
            learning_rate=1e-3,
            top_n=2,
            probability_thresholds=(0.05, 0.1),
            device="cpu",
        ),
    )

    assert len(history) == 2
    assert (output_dir / "best.pt").is_file()
    model, router_config, _, train_config, payload = load_checkpoint(output_dir / "best.pt")
    assert isinstance(model, BlockProbabilityRouter)
    assert router_config.feature_dim == 6
    assert train_config.probability_thresholds == (0.05, 0.1)
    assert payload["model_kind"] == "positive_block_probability_router"
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["information_boundary"]["current_live_block_is_candidate"] is False
    assert run_config["normalization"] == "q_dot_sum_historical_key_features"
    assert history[-1]["train"]["probability_normalization_error"] < 1e-5
