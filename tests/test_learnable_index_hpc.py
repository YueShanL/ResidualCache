from __future__ import annotations

import json
from pathlib import Path

import pytest

from learnable_index.hpc import (
    HPCPipeline,
    load_hpc_config,
    validate_hpc_config,
)


def _config(tmp_path):
    return {
        "schema_version": 1,
        "run_id": "test",
        "seed": 13,
        "paths": {
            "output_root": str(tmp_path / "output"),
        },
        "model": {
            "name": "google/gemma-4-E4B-it",
            "device": "cuda",
            "dtype": "bfloat16",
            "allow_network": True,
        },
        "data": {
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "cache_dir": None,
            "allow_network": True,
            "sequence_length": 1024,
            "splits": {
                "train": {"sequences": 2, "article_stride": 1},
                "validation": {"sequences": 1, "article_stride": 1},
                "test": {"sequences": 1, "article_stride": 1},
            },
        },
        "collection": {
            "local_context_length": 256,
            "block_size": 64,
            "future_horizon": 16,
            "retrieval_interval": 128,
            "minimum_candidate_blocks": 2,
            "maximum_candidate_blocks": 8,
            "progress_every": 25,
        },
        "router": {
            "projection_dim": 128,
            "hidden_dim": 256,
            "depth": 2,
            "dropout": 0.1,
            "temperature": 0.07,
        },
        "training": {
            "epochs": 5,
            "early_stopping_patience": 2,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "validation_fraction": 0.1,
            "top_n": 4,
            "device": "cuda",
        },
        "evaluation": {"top_n": 4, "device": "cuda"},
        "replay": {
            "enabled": True,
            "split": "test",
            "policy": "fixed",
            "top_n": 4,
            "score_threshold": 0.0,
            "maximum_samples": 4,
            "router_device": "cuda",
        },
        "stages": {
            "prepare": True,
            "collect": True,
            "train": True,
            "evaluate": True,
            "replay": True,
        },
    }


def test_hpc_config_loads_hugging_face_ids_without_path_expansion(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_hpc_config(path)
    assert loaded["model"]["name"] == "google/gemma-4-E4B-it"
    assert loaded["data"]["dataset_name"] == "Salesforce/wikitext"


def test_hpc_sample_count_and_hf_arguments(tmp_path):
    config = _config(tmp_path)
    validate_hpc_config(config)
    pipeline = HPCPipeline(config)

    assert pipeline._expected_samples("train") == 10
    assert pipeline._model_arguments() == [
        "--model-name",
        "google/gemma-4-E4B-it",
        "--model-device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--allow-network",
    ]
    assert pipeline._huggingface_arguments() == ["--allow-network"]


def test_prepare_stage_passes_hf_ids_directly_to_python_runner(tmp_path, monkeypatch):
    pipeline = HPCPipeline(_config(tmp_path))
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )

    pipeline.prepare_inputs()

    assert [stage for stage, _ in commands] == [
        "prepare:train",
        "prepare:validation",
        "prepare:test",
    ]
    for _, arguments in commands:
        assert arguments[arguments.index("--dataset-name") + 1] == "Salesforce/wikitext"
        assert arguments[arguments.index("--tokenizer") + 1] == "google/gemma-4-E4B-it"
        assert "--arrow-dir" not in arguments
        assert "--allow-network" in arguments


def test_convomem_hpc_preparation_uses_answer_aligned_single_points(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["data"] = {
        "source": "convomem",
        "dataset_name": "Salesforce/ConvoMem",
        "cache_dir": None,
        "allow_network": True,
        "sequence_length": 4096,
        "maximum_answer_tokens": 64,
        "maximum_future_horizon": 16,
        "splits": {
            "train": {"sequences": 8},
            "validation": {"sequences": 2},
            "test": {"sequences": 2},
        },
    }
    config["collection"]["retrieval_point_policy"] = "metadata"
    config["collection"]["teacher_prefill_chunk_size"] = 256
    config["data"]["splits"]["train"]["store_kv_payload"] = False
    config["collection"].update(
        {
            "residual_layer": 40,
            "query_summary": "mean",
            "query_summary_length": 16,
            "teacher_layers": [29, 35, 41],
            "teacher_heads": "all",
            "future_reduction": "mean",
            "length_normalize_blocks": False,
        }
    )
    validate_hpc_config(config)
    pipeline = HPCPipeline(config)
    assert pipeline._expected_samples("train") == 8
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )
    pipeline.prepare_inputs()
    assert all(
        arguments[1] == "learnable_index.prepare_convomem"
        for _, arguments in commands
    )
    collection_arguments = pipeline._collection_arguments()
    assert collection_arguments[
        collection_arguments.index("--retrieval-point-policy") + 1
    ] == "metadata"
    assert collection_arguments[
        collection_arguments.index("--teacher-prefill-chunk-size") + 1
    ] == "256"
    assert "--no-store-kv-payload" in pipeline._collection_arguments("train")
    assert "--no-store-kv-payload" not in pipeline._collection_arguments("test")


def test_ephemeral_inputs_are_synthesized_collected_and_removed_per_split(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config["_config_path"] = str(tmp_path / "config.json")
    config["data"].update(
        {
            "source": "convomem",
            "dataset_name": "Salesforce/ConvoMem",
            "persist_prepared_inputs": False,
        }
    )
    for split in ("train", "validation", "test"):
        config["data"]["splits"][split].pop("article_stride")
    pipeline = HPCPipeline(config)
    events = []

    def prepare(splits=("train", "validation", "test")):
        split = tuple(splits)[0]
        events.append(f"prepare:{split}")
        path = pipeline._input_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        path.with_suffix(".manifest.json").write_text("{}\n", encoding="utf-8")

    def collect(splits=("train", "validation", "test")):
        split = tuple(splits)[0]
        events.append(f"collect:{split}")
        output = pipeline._collection_dir(split)
        output.mkdir(parents=True, exist_ok=True)
        (output / "collection_manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "prepare_inputs", prepare)
    monkeypatch.setattr(pipeline, "collect", collect)
    monkeypatch.setattr(pipeline, "train", lambda: None)
    monkeypatch.setattr(pipeline, "evaluate", lambda: None)
    monkeypatch.setattr(pipeline, "replay", lambda: None)
    pipeline.run()

    assert events == [
        "prepare:train",
        "collect:train",
        "prepare:validation",
        "collect:validation",
        "prepare:test",
        "collect:test",
    ]
    for split in ("train", "validation", "test"):
        assert not pipeline._input_path(split).exists()
        assert not pipeline._input_path(split).with_suffix(".manifest.json").exists()


def test_train_and_replay_commands_are_query_key_only(tmp_path, monkeypatch):
    pipeline = HPCPipeline(_config(tmp_path))
    commands = []
    monkeypatch.setattr(
        pipeline,
        "_run_command",
        lambda stage, arguments: commands.append((stage, list(arguments))),
    )

    pipeline.train()
    pipeline.replay()

    train_arguments = commands[0][1]
    replay_arguments = commands[1][1]
    assert commands[0][0] == "train"
    assert train_arguments[train_arguments.index("--early-stopping-patience") + 1] == "2"
    assert "--demand-weight" not in train_arguments
    assert "--demand-loss" not in train_arguments
    assert commands[1][0] == "replay:test"
    assert replay_arguments[replay_arguments.index("--policy") + 1] == "fixed"
    assert replay_arguments[replay_arguments.index("--score-threshold") + 1] == "0.0"
    assert "--demand-threshold" not in replay_arguments


@pytest.mark.parametrize("field", ["model", "dataset"])
def test_hpc_config_rejects_absolute_model_and_dataset_paths(tmp_path, field):
    config = _config(tmp_path)
    if field == "model":
        config["model"]["name"] = "C:\\models\\gemma"
    else:
        config["data"]["dataset_name"] = "/datasets/wikitext"

    with pytest.raises(ValueError, match="repository ID, not a path"):
        validate_hpc_config(config)


def test_fixed_slurm_script_directly_launches_runner_without_resubmission():
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / "scripts" / "submit_learnable_index_hpc.slurm").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --gres=gpu:1" in script
    assert "run_learnable_index_hpc.py" in script
    assert "learnable_index_wikitext4096_hpc.json" in script
    assert "exec sbatch" not in script
    assert "--print-sbatch-args" not in script
    assert "exec srun" not in script
    assert 'export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME}/token}"' in script
    assert "cat ${HF_TOKEN_PATH}" not in script


def test_wikitext4096_config_preserves_control_variables_and_full_replay():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hpc_config(
        project_root / "configs" / "learnable_index_wikitext4096_hpc.json"
    )
    validate_hpc_config(config)
    pipeline = HPCPipeline(config)

    assert config["data"]["splits"]["train"]["sequences"] == 4096
    assert config["data"]["splits"]["validation"]["sequences"] == 59
    assert config["data"]["splits"]["test"]["sequences"] == 58
    assert pipeline._expected_samples("train") == 20_480
    assert pipeline._expected_samples("validation") == 295
    assert pipeline._expected_samples("test") == 290
    assert config["training"]["validation_fraction"] == pytest.approx(0.1)
    assert round(4096 * config["training"]["validation_fraction"]) == 410
    assert config["training"]["epochs"] == 10
    assert config["training"]["early_stopping_patience"] == 2
    assert config["router"] == {
        "projection_dim": 128,
        "hidden_dim": 256,
        "depth": 2,
        "dropout": 0.1,
        "temperature": 0.07,
    }
    assert config["training"]["learning_rate"] == pytest.approx(3e-4)
    assert config["training"]["weight_decay"] == pytest.approx(1e-4)
    assert config["replay"]["maximum_samples"] is None
    assert config["paths"]["use_tmp_workspace"] is True
    assert config["paths"]["tmp_workspace_root"] == "/tmp"


def test_convomem4096_config_is_ephemeral_and_preserves_training_controls():
    project_root = Path(__file__).resolve().parents[1]
    config = load_hpc_config(
        project_root / "configs" / "learnable_index_convomem4096_hpc.json"
    )
    validate_hpc_config(config)
    pipeline = HPCPipeline(config)

    assert config["data"]["source"] == "convomem"
    assert config["data"]["sequence_length"] == 4096
    assert config["data"]["persist_prepared_inputs"] is False
    assert config["data"]["splits"]["train"]["sequences"] == 4096
    assert config["data"]["splits"]["validation"]["sequences"] == 295
    assert config["data"]["splits"]["test"]["sequences"] == 290
    assert pipeline._expected_samples("test") == 290
    assert config["data"]["splits"]["train"]["store_kv_payload"] is False
    assert config["data"]["splits"]["validation"]["store_kv_payload"] is False
    assert config["data"]["splits"]["test"]["store_kv_payload"] is True
    assert config["collection"]["retrieval_point_policy"] == "metadata"
    assert config["collection"]["maximum_candidate_blocks"] is None
    assert config["training"]["validation_fraction"] == pytest.approx(0.1)
    assert config["training"]["epochs"] == 10
    assert config["training"]["early_stopping_patience"] == 2
    assert config["replay"]["maximum_samples"] is None


def test_tmp_workspace_exports_only_best_model_and_training_metrics(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["_config_path"] = str(tmp_path / "config.json")
    config["paths"]["use_tmp_workspace"] = True
    config["paths"]["tmp_workspace_root"] = str(tmp_path / "node-local")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    pipeline = HPCPipeline(config)

    assert pipeline.output_root.parent.parent == (tmp_path / "node-local").resolve()
    assert pipeline.persistent_output_root == (tmp_path / "output").resolve()
    assert pipeline.environment["HF_HOME"].startswith(str(pipeline.output_root))
    assert pipeline.environment["HF_DATASETS_CACHE"].startswith(
        str(pipeline.output_root)
    )
    pipeline._initialize()
    assert not pipeline.persistent_output_root.exists()

    training = pipeline.output_root / "training"
    training.mkdir(parents=True)
    (training / "best.pt").write_bytes(b"checkpoint")
    (training / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
    (training / "summary.json").write_text('{"best_epoch": 1}\n', encoding="utf-8")
    (training / "final.pt").write_bytes(b"not exported")
    pipeline._export_training_artifacts()

    exported = {
        path.relative_to(pipeline.persistent_output_root).as_posix()
        for path in pipeline.persistent_output_root.rglob("*")
        if path.is_file()
    }
    assert exported == {
        "training/best.pt",
        "training/metrics.jsonl",
        "training/summary.json",
    }


def test_tmp_workspace_preserves_token_path_from_original_hf_home(
    tmp_path, monkeypatch
):
    persistent_hf_home = tmp_path / "persistent-huggingface"
    persistent_hf_home.mkdir()
    token_path = persistent_hf_home / "token"
    token_path.write_text("secret-not-for-logs", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(persistent_hf_home))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    config = _config(tmp_path)
    config["paths"]["use_tmp_workspace"] = True
    config["paths"]["tmp_workspace_root"] = str(tmp_path / "node-local")

    pipeline = HPCPipeline(config)

    assert pipeline.environment["HF_TOKEN_PATH"] == str(token_path.resolve())
    assert pipeline.environment["HF_HOME"].startswith(str(pipeline.output_root))
    assert "secret-not-for-logs" not in repr(pipeline.environment["HF_TOKEN_PATH"])


def test_tmp_workspace_root_must_be_absolute(tmp_path):
    config = _config(tmp_path)
    config["paths"]["use_tmp_workspace"] = True
    config["paths"]["tmp_workspace_root"] = "relative/scratch"

    with pytest.raises(ValueError, match="must be an absolute path"):
        validate_hpc_config(config)
