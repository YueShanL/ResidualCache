from copy import deepcopy
from pathlib import Path

from output_preserving_region_router.hpc import (
    HPCPipeline,
    load_hpc_config,
    validate_hpc_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    config = load_hpc_config(
        PROJECT_ROOT
        / "configs"
        / "output_preserving_region_router_convomem4096_smoke_hpc.json"
    )
    config = deepcopy(config)
    config["paths"]["output_root"] = str(tmp_path / "output")
    config["paths"]["use_tmp_workspace"] = False
    return config


def _full_config():
    return load_hpc_config(
        PROJECT_ROOT
        / "configs"
        / "output_preserving_region_router_convomem4096_full_hpc.json"
    )


def test_example_config_is_one_region_output_preservation_smoke(tmp_path):
    config = _config(tmp_path)
    validate_hpc_config(config)

    assert config["data"]["sequence_length"] == 4096
    assert config["data"]["splits"]["train"]["sequences"] == 64
    assert config["collection"]["local_context_length"] == 512
    assert config["collection"]["block_size"] == 64
    assert config["collection"]["residual_layer"] == 40
    assert config["objective"]["maximum_excess_output_kl"] == 0.02
    assert config["collection"]["maximum_candidate_blocks"] is None
    assert "teacher_layers" not in config["collection"]
    assert config["data"]["persist_prepared_inputs"] is False


def test_full_config_preserves_system_and_uses_official_scale():
    config = _full_config()
    validate_hpc_config(config)

    assert config["data"]["splits"]["train"]["sequences"] == 4096
    assert config["data"]["splits"]["validation"]["sequences"] == 295
    assert config["data"]["splits"]["test"]["sequences"] == 290
    assert config["training"]["epochs"] == 10
    assert config["training"]["early_stopping_patience"] == 2
    assert config["training"]["validation_fraction"] == 0.1
    assert config["collection"]["maximum_candidate_blocks"] is None
    assert config["evaluation"]["maximum_samples"] is None
    assert config["evaluation"]["qa_maximum_samples"] is None


def test_hpc_train_and_hard_evaluation_use_independent_entry(tmp_path, monkeypatch):
    pipeline = HPCPipeline(_config(tmp_path))
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )

    pipeline.train()
    pipeline.evaluate("validation")

    train = commands[0][1]
    assert train[:3] == ["-m", "output_preserving_region_router", "train"]
    assert "--maximum-excess-output-kl" in train
    assert "--teacher-layers" not in train
    assert "--maximum-candidate-blocks" not in train
    assert "--input-jsonl" in train
    evaluation = commands[1][1]
    assert evaluation[:3] == [
        "-m",
        "output_preserving_region_router",
        "evaluate",
    ]
    assert commands[1][0] == "evaluate:validation"
