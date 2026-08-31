from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from block_probability_router.cli import _evaluate
from block_probability_router.config import (
    ProbabilityLossConfig,
    ProbabilityRouterConfig,
    ProbabilityTrainConfig,
)
from block_probability_router.losses import probability_router_loss
from block_probability_router.model import (
    BlockProbabilityRouter,
    minimum_cumulative_mass_mask,
)
from block_probability_router.streaming_collection import STUDENT_STATE_PROTOCOL
from block_probability_router.trainer import evaluate_model, fit_router, load_checkpoint
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


def test_global_two_percent_missing_mass_retains_flat_attention_tail():
    probabilities = torch.full((1, 59), 1.0 / 59.0)
    mask = torch.ones_like(probabilities, dtype=torch.bool)

    selected = minimum_cumulative_mass_mask(probabilities, mask, 0.02)

    assert int(selected.sum()) == 58
    assert float(probabilities[selected].sum()) >= 0.98 - 1e-6
    assert float(probabilities[selected].sum() - 1.0 / 59.0) < 0.98


def test_global_mass_policy_selects_smallest_concentrated_prefix_and_ignores_padding():
    probabilities = torch.tensor([[0.70, 0.20, 0.081, 0.019, 0.0]])
    mask = torch.tensor([[True, True, True, True, False]])

    selected = minimum_cumulative_mass_mask(probabilities, mask, 0.02)

    assert torch.equal(selected, torch.tensor([[True, True, True, False, False]]))
    assert float(probabilities[selected].sum()) >= 0.98


def test_global_mass_policy_applies_hard_block_limit_after_mass_selection():
    probabilities = torch.tensor([[0.40, 0.30, 0.20, 0.10]])
    mask = torch.ones_like(probabilities, dtype=torch.bool)

    uncapped = minimum_cumulative_mass_mask(probabilities, mask, 0.05)
    capped = minimum_cumulative_mass_mask(
        probabilities,
        mask,
        0.05,
        maximum_retrieval_blocks=2,
    )

    assert torch.equal(uncapped, torch.tensor([[True, True, True, True]]))
    assert torch.equal(capped, torch.tensor([[True, True, False, False]]))
    assert float(probabilities[capped].sum()) == pytest.approx(0.70)


@pytest.mark.parametrize("maximum", [0, -2])
def test_global_mass_policy_rejects_invalid_hard_block_limit(maximum):
    probabilities = torch.tensor([[0.60, 0.40]])
    mask = torch.ones_like(probabilities, dtype=torch.bool)

    with pytest.raises(ValueError, match="maximum_retrieval_blocks"):
        minimum_cumulative_mass_mask(
            probabilities,
            mask,
            0.05,
            maximum_retrieval_blocks=maximum,
        )


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


def test_evaluation_reports_retrieval_cap_effects():
    dataset = RetrievalDataset(
        make_synthetic_samples(
            sample_count=6,
            residual_dim=6,
            min_blocks=3,
            max_blocks=4,
            seed=31,
        )
    )
    metrics = evaluate_model(
        _model(),
        dataset,
        ProbabilityLossConfig(),
        ProbabilityTrainConfig(
            epochs=1,
            batch_size=3,
            validation_fraction=0.0,
            missing_mass_tolerances=(0.02,),
            device="cpu",
        ),
        device=torch.device("cpu"),
        maximum_retrieval_blocks=1,
    )

    assert metrics["missing_mass/0.02/selected_blocks"] == 1.0
    assert metrics["missing_mass/0.02/maximum_retrieval_blocks"] == 1.0
    assert metrics["missing_mass/0.02/cap_applied_rate"] == 1.0
    assert metrics["missing_mass/0.02/predicted_target_success_rate"] == 0.0
    assert metrics["missing_mass/0.02/oracle_target_success_rate"] < 1.0
    assert metrics["missing_mass/0.02/oracle_mass_shortfall"] > 0.0
    assert metrics["missing_mass/0.02/oracle_uncapped_selected_blocks"] > 1.0
    assert metrics["missing_mass/0.02/oracle_cap_applied_rate"] > 0.0


def test_evaluation_allows_legacy_checkpoint_and_reports_protocol_mismatch(
    monkeypatch,
):
    dataset = RetrievalDataset(
        make_synthetic_samples(
            sample_count=2,
            residual_dim=6,
            min_blocks=2,
            max_blocks=2,
            seed=37,
        )
    )
    train_config = ProbabilityTrainConfig(
        epochs=1,
        batch_size=2,
        validation_fraction=0.0,
        missing_mass_tolerances=(0.02,),
        device="cpu",
    )
    monkeypatch.setattr(
        "block_probability_router.cli.load_dataset",
        lambda _path: (
            dataset,
            {"metadata": {"student_state_protocol": STUDENT_STATE_PROTOCOL}},
        ),
    )
    monkeypatch.setattr(
        "block_probability_router.cli.load_checkpoint",
        lambda _path: (
            _model(),
            None,
            ProbabilityLossConfig(),
            train_config,
            {"epoch": 3},
        ),
    )
    monkeypatch.setattr(
        "block_probability_router.cli.evaluate_model",
        lambda *_args, **_kwargs: {"loss": 1.0},
    )

    result = _evaluate(
        SimpleNamespace(
            max_block=-1,
            dataset_dir="dataset",
            checkpoint="legacy.pt",
            output=None,
            device="cpu",
            top_n=None,
            missing_mass_tolerances=None,
        )
    )

    assert result["student_state_protocol"] == {
        "dataset": STUDENT_STATE_PROTOCOL,
        "checkpoint": None,
        "matched": False,
    }


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
            missing_mass_tolerances=(0.02, 0.05),
            device="cpu",
        ),
        student_state_protocol=STUDENT_STATE_PROTOCOL,
    )

    assert len(history) == 2
    assert (output_dir / "best.pt").is_file()
    model, router_config, _, train_config, payload = load_checkpoint(output_dir / "best.pt")
    assert isinstance(model, BlockProbabilityRouter)
    assert router_config.feature_dim == 6
    assert train_config.missing_mass_tolerances == (0.02, 0.05)
    assert payload["format_version"] == 2
    assert payload["model_kind"] == "positive_block_probability_router"
    assert payload["student_state_protocol"] == STUDENT_STATE_PROTOCOL
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["information_boundary"]["current_live_block_is_candidate"] is False
    assert run_config["normalization"] == "q_dot_sum_historical_key_features"
    assert history[-1]["train"]["probability_normalization_error"] < 1e-5
    assert "missing_mass/0.02/teacher_mass_recall" in history[-1]["train"]
    assert history[-1]["train"]["missing_mass/0.02/predicted_target_success_rate"] == 1.0


def test_v1_per_block_threshold_checkpoint_loads_with_global_mass_budgets(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    router_config = model.config
    loss_config = ProbabilityLossConfig()
    legacy_train_config = ProbabilityTrainConfig(
        epochs=1,
        batch_size=2,
        validation_fraction=0.0,
        missing_mass_tolerances=(0.02, 0.05),
        device="cpu",
    )
    legacy_payload = {
        "format_version": 1,
        "model_kind": "positive_block_probability_router",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "router_config": asdict(router_config),
        "loss_config": asdict(loss_config),
        "train_config": {
            **asdict(legacy_train_config),
            "probability_thresholds": [0.02, 0.05],
        },
        "epoch": 1,
        "metrics": {},
    }
    legacy_payload["train_config"].pop("missing_mass_tolerances")
    checkpoint = tmp_path / "legacy-v1.pt"
    torch.save(legacy_payload, checkpoint)

    loaded_model, _, _, loaded_train_config, loaded_payload = load_checkpoint(checkpoint)

    assert isinstance(loaded_model, BlockProbabilityRouter)
    assert loaded_payload["format_version"] == 1
    assert loaded_train_config.missing_mass_tolerances == (0.02, 0.05)
