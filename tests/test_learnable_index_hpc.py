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
        "router": {},
        "loss": {},
        "training": {"validation_fraction": 0.1},
        "evaluation": {},
        "replay": {},
        "stages": {},
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
    script = (project_root / "scripts" / "submit_learnable_index_hpc.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --gres=gpu:1" in script
    assert 'exec "${PYTHON_BIN}" -u' in script
    assert "run_learnable_index_hpc.py" in script
    assert "exec sbatch" not in script
    assert "--print-sbatch-args" not in script
    assert "exec srun" not in script
