from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from block_probability_router.hpc import HPCPipeline, load_hpc_config, validate_hpc_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    config = load_hpc_config(
        PROJECT_ROOT / "configs" / "block_probability_router_convomem4096_hpc.json"
    )
    config = deepcopy(config)
    config["paths"]["output_root"] = str(tmp_path / "output")
    config["paths"]["use_tmp_workspace"] = False
    return config


def test_example_config_preserves_frozen_model_collection_controls(tmp_path):
    config = _config(tmp_path)
    validate_hpc_config(config)
    pipeline = HPCPipeline(config)

    assert config["collection"]["residual_layer"] == 40
    assert config["collection"]["teacher_layers"] == [29, 35, 41]
    assert config["collection"]["block_size"] == 64
    assert config["collection"]["local_context_length"] == 512
    assert config["collection"]["future_horizon"] == 64
    assert config["data"]["maximum_future_horizon"] == 64
    assert config["data"]["evidence_placement"] == "stratified_random"
    assert config["data"]["splits"]["train"]["sequences"] == 4096
    assert pipeline._expected_samples("test") == 290
    assert "--no-store-kv-payload" in pipeline._collection_arguments("test")


def test_hpc_train_and_evaluation_use_independent_module_entry(tmp_path, monkeypatch):
    pipeline = HPCPipeline(_config(tmp_path))
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )

    pipeline.train()
    pipeline.evaluate()

    train = commands[0][1]
    assert train[:3] == ["-m", "block_probability_router", "train"]
    assert "--temperature" not in train
    assert (
        train[train.index("--missing-mass-tolerances") + 1]
        == "0.01,0.02,0.05,0.1,0.5"
    )
    assert [stage for stage, _ in commands[1:]] == ["evaluate:validation", "evaluate:test"]
    assert all(command[:3] == ["-m", "block_probability_router", "evaluate"] for _, command in commands[1:])


def test_collection_uses_dedicated_block_streaming_entry(tmp_path, monkeypatch):
    pipeline = HPCPipeline(_config(tmp_path))
    input_path = pipeline._input_path("train")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("{}\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )

    pipeline.collect(("train",))

    assert commands[0][0] == "collect:train"
    assert commands[0][1][:2] == [
        "-m",
        "block_probability_router.streaming_collection",
    ]
    assert commands[0][1][commands[0][1].index("--residual-layer") + 1] == "40"
