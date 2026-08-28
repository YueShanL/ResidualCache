from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_router_validation.hpc import (
    EXPORTED_METRIC_ARTIFACTS,
    ClusterRouterHPCPipeline,
    load_hpc_config,
    validate_hpc_config,
)


def _config(tmp_path: Path) -> dict:
    router = tmp_path / "router.pt"
    router.write_bytes(b"router")
    return {
        "schema_version": 1,
        "run_id": "validation-test",
        "seed": 13,
        "paths": {
            "output_root": str(tmp_path / "persistent"),
            "use_tmp_workspace": True,
            "tmp_workspace_root": str(tmp_path / "node-local"),
            "cleanup_tmp_workspace": True,
        },
        "router": {"path": str(router), "device": "cuda"},
        "model": {
            "name": "google/gemma-4-E4B-it",
            "device": "cuda",
            "dtype": "bfloat16",
            "allow_network": True,
        },
        "data": {
            "dataset_name": "Salesforce/ConvoMem",
            "split": "test",
            "allow_network": True,
            "sequence_count": 256,
            "sequence_length": 4096,
            "sampling_seed": 97,
            "maximum_answer_tokens": 64,
            "maximum_future_horizon": 64,
            "evidence_placement_bins": 4,
        },
        "streaming": {
            "block_size": 64,
            "local_context_length": 512,
            "residual_layer": 40,
            "query_summary_length": 16,
            "prefill_chunk_size": 64,
            "ingestion_replay_policy": "full_memory",
        },
        "memory": {
            "eviction_enabled": True,
            "slot_capacity": 128,
            "initial_record_capacity": 4096,
            "candidate_capacity": 8,
            "locality_bits": 8,
            "locality_probe_radius": 1,
            "write_chunk_size": 64,
            "alpha": 0.1,
            "tau_new": 0.5,
            "index_mode": "mean_kv",
            "usage_ema_rate": 0.25,
            "eviction_usage_threshold": 0.001,
            "eviction_min_recall_count": 1,
            "eviction_min_records_per_cluster": 1,
        },
        "evaluation": {
            "budgets": [4],
            "fixed_policy": "recent",
            "oracle_signal": "evidence",
            "maximum_samples": 256,
            "continue_on_error": False,
            "resume": True,
        },
        "metrics": {
            "distance_boundaries": [1024, 2048, 4096],
            "length_boundaries": [4096, 8192],
            "kv_ratio_thresholds": [0.25, 0.5],
            "allow_incomplete_state": False,
        },
    }


def test_hpc_config_loads_and_requires_explicit_router_path(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_hpc_config(path)
    validate_hpc_config(loaded)

    assert loaded["router"]["path"] == str(tmp_path / "router.pt")
    loaded["router"]["path"] = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="router checkpoint is missing"):
        validate_hpc_config(loaded)


def test_generated_collect_config_uses_hf_ids_router_and_tmp_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "410")
    pipeline = ClusterRouterHPCPipeline(_config(tmp_path))

    generated = pipeline._collect_config()
    dataset = generated["dataset"]["kwargs"]
    model = generated["model"]["kwargs"]

    assert dataset["dataset_name"] == "Salesforce/ConvoMem"
    assert dataset["tokenizer_name"] == "google/gemma-4-E4B-it"
    assert dataset["dataset_cache_dir"].startswith(str(pipeline.workspace))
    assert dataset["sequence_count"] == 256
    assert dataset["sequence_length"] == 4096
    assert model["model_name"] == "google/gemma-4-E4B-it"
    assert model["checkpoint_path"] == str((tmp_path / "router.pt").resolve())
    assert dataset["local_context_length"] == 512
    assert model["local_context_length"] == 512
    assert model["ingestion_replay_policy"] == "full_memory"
    assert model["memory_config"]["write_chunk_size"] == 64
    assert generated["output_dir"] == str(pipeline.state_dir)


def test_tmp_cleanup_exports_only_metric_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "411")
    pipeline = ClusterRouterHPCPipeline(_config(tmp_path))
    workspace = pipeline.workspace

    def fake_collect(_config_path):
        pipeline.state_dir.mkdir(parents=True)
        (pipeline.state_dir / "samples.jsonl").write_text("state\n", encoding="utf-8")
        (pipeline.state_dir / "test_database.bin").write_bytes(b"temporary database")

    def fake_metrics(_config_path):
        pipeline.metric_dir.mkdir(parents=True)
        for name in EXPORTED_METRIC_ARTIFACTS:
            (pipeline.metric_dir / name).write_text(name + "\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_collect", fake_collect)
    monkeypatch.setattr(pipeline, "_metrics", fake_metrics)

    result = pipeline.run()

    assert result["temporary_workspace_cleaned"] is True
    assert not workspace.exists()
    files = {
        path.relative_to(pipeline.persistent_output_root).as_posix()
        for path in pipeline.persistent_output_root.rglob("*")
        if path.is_file()
    }
    assert files == {f"metrics/{name}" for name in EXPORTED_METRIC_ARTIFACTS}
    assert not (pipeline.persistent_output_root / "state").exists()
    assert not any("database" in path for path in files)


def test_tmp_cleanup_requires_tmp_mode(tmp_path):
    config = _config(tmp_path)
    config["paths"]["use_tmp_workspace"] = False

    with pytest.raises(ValueError, match="cleanup_tmp_workspace requires"):
        validate_hpc_config(config)


def test_model_and_dataset_must_be_hugging_face_ids(tmp_path):
    config = _config(tmp_path)
    config["model"]["name"] = str(tmp_path / "model")

    with pytest.raises(ValueError, match="Hugging Face repository ID"):
        validate_hpc_config(config)


def test_persistent_metrics_only_destination_rejects_foreign_files(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    output = Path(config["paths"]["output_root"])
    output.mkdir()
    (output / "database.bin").write_bytes(b"must not be retained")
    monkeypatch.setenv("SLURM_JOB_ID", "412")
    pipeline = ClusterRouterHPCPipeline(config)

    with pytest.raises(RuntimeError, match="unexpected artifacts"):
        pipeline._initialize()
