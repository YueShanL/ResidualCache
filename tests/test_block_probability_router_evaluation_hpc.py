from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from block_probability_router.evaluation_hpc import (
    EvaluationHPCPipeline,
    load_hpc_config,
    validate_evaluation_hpc_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    config = load_hpc_config(
        PROJECT_ROOT
        / "configs"
        / "block_probability_router_evaluation_convomem4096_hpc.json"
    )
    config = deepcopy(config)
    config["paths"]["output_root"] = str(tmp_path / "output")
    config["paths"]["use_tmp_workspace"] = False
    return config


def test_evaluation_config_exposes_mass_and_optional_block_limit(tmp_path):
    config = _config(tmp_path)
    validate_evaluation_hpc_config(config)

    assert config["evaluation"]["epsilon"] == [0.02, 0.05, 0.1, 0.2, 0.5]
    assert config["evaluation"]["max_block"] == -1
    assert config["data"]["sequences"] == 290
    assert config["data"]["sequence_length"] == 4096
    assert config["data"]["retrieval_local_context_length"] == 512
    assert config["collection"]["local_context_length"] == 512
    assert config["collection"]["block_size"] == 64
    assert config["collection"]["residual_layer"] == 40
    assert config["collection"]["teacher_layers"] == [29, 35, 41]
    assert config["router"]["checkpoint"].endswith(
        "block_probability_router_convomem_4096tokens_4096train_random_v1/training/best.pt"
    )
    assert config["qa"]["enabled"] is True
    assert config["qa"]["maximum_samples"] is None
    assert config["qa"]["maximum_new_tokens"] == 64


def test_scalar_epsilon_and_omitted_max_block_are_valid(tmp_path):
    config = _config(tmp_path)
    config["evaluation"]["epsilon"] = 0.06
    config["evaluation"].pop("max_block")

    validate_evaluation_hpc_config(config)
    pipeline = EvaluationHPCPipeline(config)

    assert pipeline.config["evaluation"].get("max_block", -1) == -1


@pytest.mark.parametrize("maximum", [0, -2, 3.5, True])
def test_evaluation_config_rejects_invalid_block_limit(tmp_path, maximum):
    config = _config(tmp_path)
    config["evaluation"]["max_block"] = maximum

    with pytest.raises(ValueError, match="evaluation.max_block"):
        validate_evaluation_hpc_config(config)


def test_independent_pipeline_collects_then_evaluates_without_training(
    tmp_path,
    monkeypatch,
):
    pipeline = EvaluationHPCPipeline(_config(tmp_path))
    commands = []

    def fake_run(stage, arguments):
        command = list(arguments)
        commands.append((stage, command))
        if stage == "evaluate":
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "checkpoint": command[command.index("--checkpoint") + 1],
                        "checkpoint_epoch": 8,
                        "sample_count": 290,
                        "retrieval_policy": {},
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(pipeline, "_run_command", fake_run)

    pipeline.prepare()
    pipeline.collect()
    pipeline.evaluate()

    assert [stage for stage, _ in commands] == ["prepare", "collect", "evaluate"]
    assert commands[0][1][:2] == ["-m", "learnable_index.prepare_convomem"]
    assert commands[1][1][:2] == [
        "-m",
        "block_probability_router.streaming_collection",
    ]
    assert "--no-store-kv-payload" in commands[1][1]
    evaluate = commands[2][1]
    assert evaluate[:3] == ["-m", "block_probability_router", "evaluate"]
    assert (
        evaluate[evaluate.index("--missing-mass-tolerances") + 1]
        == "0.02,0.05,0.1,0.2,0.5"
    )
    assert evaluate[evaluate.index("--max-block") + 1] == "-1"
    result = json.loads(pipeline._metrics_path().read_text(encoding="utf-8"))
    assert result["config_fingerprint"] == pipeline.fingerprint
    assert result["data"]["sequence_count"] == 290


def test_qa_stage_runs_autoregressive_evaluator_and_merges_summary(
    tmp_path,
    monkeypatch,
):
    pipeline = EvaluationHPCPipeline(_config(tmp_path))
    pipeline.output_root.mkdir(parents=True, exist_ok=True)
    pipeline._metrics_path().write_text(
        json.dumps({"config_fingerprint": pipeline.fingerprint}),
        encoding="utf-8",
    )
    commands = []

    def fake_run(stage, arguments):
        command = list(arguments)
        commands.append((stage, command))
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "qa_schema_version": 3,
                    "evaluation_kind": "greedy_autoregressive_long_context_qa",
                    "teacher_forcing": False,
                    "summary": {"sample_count": 290, "conditions": {}},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(pipeline, "_run_command", fake_run)
    pipeline.qa()

    assert [stage for stage, _ in commands] == ["qa"]
    qa_command = commands[0][1]
    assert qa_command[:2] == ["-m", "block_probability_router.qa"]
    assert qa_command[qa_command.index("--max-block") + 1] == "-1"
    assert (
        qa_command[qa_command.index("--missing-mass-tolerances") + 1]
        == "0.02,0.05,0.1,0.2,0.5"
    )
    result = json.loads(pipeline._metrics_path().read_text(encoding="utf-8"))
    assert result["qa"]["teacher_forcing"] is False
    assert result["qa"]["per_sample_artifact_retained"] is False


def test_tmp_workspace_exports_only_metrics_and_is_safely_removed(tmp_path):
    config = _config(tmp_path)
    config["paths"]["use_tmp_workspace"] = True
    config["paths"]["tmp_workspace_root"] = str(tmp_path / "node_tmp")
    pipeline = EvaluationHPCPipeline(config)
    pipeline.output_root.mkdir(parents=True, exist_ok=True)
    pipeline._metrics_path().write_text(
        json.dumps({"config_fingerprint": pipeline.fingerprint}),
        encoding="utf-8",
    )

    pipeline._export()
    pipeline._cleanup_tmp_workspace()

    exported = list(pipeline.persistent_output_root.rglob("*"))
    assert exported == [pipeline.persistent_output_root / "metrics.json"]
    assert not pipeline.output_root.exists()
