"""Config-driven HPC wrapper for clustered-router end-to-end validation.

The expensive collection state and all model/dataset caches can live in a
node-local temporary workspace.  A successful run exports only the three
offline metric artifacts to persistent storage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMP_WORKSPACE_ROOT = "/tmp"
EXPORTED_METRIC_ARTIFACTS = (
    "metrics.json",
    "sample_metrics.jsonl",
    "condition_summary.csv",
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "run_id",
    "seed",
    "paths",
    "router",
    "model",
    "data",
    "streaming",
    "memory",
    "evaluation",
    "metrics",
    "_config_path",
}
_MEMORY_FIELDS = {
    "memory_budget_bytes",
    "slot_capacity",
    "candidate_capacity",
    "locality_bits",
    "locality_probe_radius",
    "write_chunk_size",
    "alpha",
    "tau_new",
    "count_exponent",
    "concentration_prior_mass",
    "maximum_concentration",
    "router_count_exponent",
    "router_concentration_prior_mass",
    "router_maximum_concentration",
    "index_mode",
    "locality_seed",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_config(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_run_id(value: Any) -> str:
    run_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value or "")
    ).strip("_")
    if not run_id:
        raise ValueError("run_id must contain a filesystem-safe character")
    return run_id


def _require_hf_id(value: Any, field: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{field} must be a non-empty Hugging Face repository ID")
    if identifier.startswith(("/", "\\", "./", "../", ".\\", "..\\")):
        raise ValueError(f"{field} must be a Hugging Face repository ID, not a path")
    if Path(identifier).is_absolute() or (
        len(identifier) >= 3
        and identifier[1] == ":"
        and identifier[2] in {"/", "\\"}
    ):
        raise ValueError(f"{field} must be a Hugging Face repository ID, not a path")
    return identifier


def _require_boolean(payload: Mapping[str, Any], field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def load_hpc_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HPC configuration root must be an object")
    payload["_config_path"] = str(config_path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("HPC config schema_version must be 1")
    return payload


def _persistent_output_path(config: Mapping[str, Any]) -> Path:
    value = str(config["paths"].get("output_root", "")).strip()
    if not value:
        raise ValueError("paths.output_root must be non-empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _router_path(config: Mapping[str, Any]) -> Path:
    value = str(config["router"].get("path", "")).strip()
    if not value:
        raise ValueError("router.path must be non-empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _temporary_workspace_path(config: Mapping[str, Any]) -> Path:
    root = Path(
        str(config["paths"].get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT))
    )
    job_id = os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_id = f"{job_id}-{array_id}" if array_id is not None else job_id
    return (
        root
        / "residualcache_cluster_router_validation"
        / f"{_safe_run_id(config['run_id'])}-{task_id}"
    ).resolve()


def validate_hpc_config(config: Mapping[str, Any]) -> None:
    unknown = set(config).difference(_TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown HPC config fields: {sorted(unknown)}")
    for section in (
        "paths",
        "router",
        "model",
        "data",
        "streaming",
        "memory",
        "evaluation",
        "metrics",
    ):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"missing or invalid config section: {section}")
    _safe_run_id(config.get("run_id"))
    int(config.get("seed", 13))

    paths = config["paths"]
    _persistent_output_path(config)
    use_tmp = _require_boolean(paths, "use_tmp_workspace", False)
    cleanup_tmp = _require_boolean(paths, "cleanup_tmp_workspace", False)
    if cleanup_tmp and not use_tmp:
        raise ValueError(
            "paths.cleanup_tmp_workspace requires paths.use_tmp_workspace=true"
        )
    tmp_root = str(paths.get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT)).strip()
    if not tmp_root:
        raise ValueError("paths.tmp_workspace_root must be non-empty")
    if not (tmp_root.startswith("/") or Path(tmp_root).is_absolute()):
        raise ValueError("paths.tmp_workspace_root must be an absolute path")

    router = config["router"]
    checkpoint = _router_path(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"router checkpoint is missing: {checkpoint}")
    if not str(router.get("device", "cuda")).strip():
        raise ValueError("router.device must be non-empty")

    model = config["model"]
    _require_hf_id(model.get("name"), "model.name")
    if not str(model.get("device", "cuda")).strip():
        raise ValueError("model.device must be non-empty")
    if not str(model.get("dtype", "bfloat16")).strip():
        raise ValueError("model.dtype must be non-empty")
    _require_boolean(model, "allow_network", False)

    data = config["data"]
    _require_hf_id(data.get("dataset_name"), "data.dataset_name")
    if not str(data.get("split", "test")).strip():
        raise ValueError("data.split must be non-empty")
    sequence_count = _require_positive_int(data.get("sequence_count"), "data.sequence_count")
    _require_positive_int(data.get("sequence_length"), "data.sequence_length")
    _require_positive_int(
        data.get("maximum_answer_tokens", 64), "data.maximum_answer_tokens"
    )
    _require_positive_int(
        data.get("maximum_future_horizon", 64), "data.maximum_future_horizon"
    )
    _require_positive_int(
        data.get("evidence_placement_bins", 4), "data.evidence_placement_bins"
    )
    _require_boolean(data, "allow_network", False)

    streaming = config["streaming"]
    block_size = _require_positive_int(streaming.get("block_size"), "streaming.block_size")
    local_context = _require_positive_int(
        streaming.get("local_context_length"), "streaming.local_context_length"
    )
    prefill = _require_positive_int(
        streaming.get("prefill_chunk_size", block_size),
        "streaming.prefill_chunk_size",
    )
    query_length = _require_positive_int(
        streaming.get("query_summary_length", 16),
        "streaming.query_summary_length",
    )
    if local_context < block_size:
        raise ValueError("streaming.local_context_length must be at least block_size")
    if prefill != block_size:
        raise ValueError("streaming.prefill_chunk_size must equal block_size")
    if query_length > local_context:
        raise ValueError(
            "streaming.query_summary_length cannot exceed local_context_length"
        )
    if int(streaming.get("residual_layer", 40)) < 0:
        raise ValueError("streaming.residual_layer cannot be negative")

    memory = config["memory"]
    unknown_memory = set(memory).difference(_MEMORY_FIELDS)
    if unknown_memory:
        raise ValueError(f"unknown memory fields: {sorted(unknown_memory)}")
    for field in (
        "memory_budget_bytes",
        "slot_capacity",
        "candidate_capacity",
        "locality_bits",
        "write_chunk_size",
    ):
        _require_positive_int(memory.get(field), f"memory.{field}")
    if int(memory["write_chunk_size"]) != block_size:
        raise ValueError("memory.write_chunk_size must equal streaming.block_size")
    if int(memory["slot_capacity"]) < int(memory["write_chunk_size"]):
        raise ValueError("memory.slot_capacity must be at least write_chunk_size")
    if int(memory.get("locality_probe_radius", 1)) not in {0, 1}:
        raise ValueError("memory.locality_probe_radius must be 0 or 1")
    if str(memory.get("index_mode", "mean_kv")) not in {"key", "mean_kv"}:
        raise ValueError("memory.index_mode must be key or mean_kv")

    evaluation = config["evaluation"]
    budgets = tuple(int(value) for value in evaluation.get("budgets", ()))
    if not budgets or budgets != tuple(sorted(set(budgets))) or min(budgets) <= 0:
        raise ValueError("evaluation.budgets must be sorted unique positive integers")
    maximum_samples = evaluation.get("maximum_samples")
    if maximum_samples is not None:
        maximum_samples = _require_positive_int(
            maximum_samples, "evaluation.maximum_samples"
        )
        if maximum_samples > sequence_count:
            raise ValueError(
                "evaluation.maximum_samples cannot exceed data.sequence_count"
            )
    if str(evaluation.get("fixed_policy", "recent")) != "recent":
        raise ValueError("evaluation.fixed_policy currently supports only recent")
    if str(evaluation.get("oracle_signal", "evidence")) not in {
        "auto",
        "teacher_attention",
        "evidence",
    }:
        raise ValueError("evaluation.oracle_signal is invalid")
    continue_on_error = _require_boolean(evaluation, "continue_on_error", False)
    _require_boolean(evaluation, "resume", True)

    metrics = config["metrics"]
    allowed_metrics = {
        "distance_boundaries",
        "length_boundaries",
        "kv_ratio_thresholds",
        "allow_incomplete_state",
    }
    unknown_metrics = set(metrics).difference(allowed_metrics)
    if unknown_metrics:
        raise ValueError(f"unknown metrics fields: {sorted(unknown_metrics)}")
    allow_incomplete = _require_boolean(metrics, "allow_incomplete_state", False)
    if continue_on_error and not allow_incomplete:
        raise ValueError(
            "metrics.allow_incomplete_state must be true when evaluation.continue_on_error is true"
        )


class ClusterRouterHPCPipeline:
    """Run collection and offline metrics under one scheduler allocation."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_hpc_config(config)
        self.config = config
        self.fingerprint = _fingerprint(config)
        self.persistent_output_root = _persistent_output_path(config)
        self.use_tmp_workspace = bool(config["paths"].get("use_tmp_workspace", False))
        self.cleanup_tmp_workspace = bool(
            config["paths"].get("cleanup_tmp_workspace", False)
        )
        self.workspace = (
            _temporary_workspace_path(config)
            if self.use_tmp_workspace
            else self.persistent_output_root
        )
        self.state_dir = self.workspace / "state"
        self.metric_dir = self.workspace / "metrics"
        self.generated_config_dir = self.workspace / "generated_configs"
        self.environment = self._build_environment()

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": str(self.config.get("seed", 13)),
            }
        )
        if not environment.get("HF_TOKEN") and not environment.get("HF_TOKEN_PATH"):
            original_hf_home = Path(
                environment.get(
                    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
                )
            ).expanduser()
            original_token_path = original_hf_home / "token"
            if original_token_path.is_file():
                environment["HF_TOKEN_PATH"] = str(original_token_path.resolve())
        current_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(PROJECT_ROOT), current_pythonpath) if value
        )
        if self.use_tmp_workspace:
            cache_root = self.workspace / "cache"
            huggingface_root = cache_root / "huggingface"
            environment.update(
                {
                    "HF_HOME": str(huggingface_root),
                    "HF_HUB_CACHE": str(huggingface_root / "hub"),
                    "HF_DATASETS_CACHE": str(huggingface_root / "datasets"),
                    "TORCH_HOME": str(cache_root / "torch"),
                    "XDG_CACHE_HOME": str(cache_root),
                }
            )
        return environment

    def _emit(self, event: str, **fields: Any) -> None:
        row = {"time": _utc_now(), "event": event, **fields}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        with (self.workspace / "pipeline_events.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _validate_persistent_destination(self) -> None:
        if not self.use_tmp_workspace or not self.persistent_output_root.exists():
            return
        allowed = {
            self.persistent_output_root / "metrics" / name
            for name in EXPORTED_METRIC_ARTIFACTS
        }
        unexpected = [
            path
            for path in self.persistent_output_root.rglob("*")
            if (path.is_file() or path.is_symlink()) and path not in allowed
        ]
        if unexpected:
            raise RuntimeError(
                "metrics-only destination contains unexpected artifacts: "
                + ", ".join(str(path) for path in unexpected[:5])
            )

    def _initialize(self) -> None:
        self._validate_persistent_destination()
        if self.use_tmp_workspace:
            try:
                self.persistent_output_root.relative_to(self.workspace)
            except ValueError:
                pass
            else:
                raise ValueError("persistent output cannot be inside the temporary workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        resolved_path = self.workspace / "resolved_config.json"
        payload = {
            "config_fingerprint": self.fingerprint,
            "source_config": self.config.get("_config_path"),
            "resolved_config": _canonical_config(self.config),
        }
        if resolved_path.exists():
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"workspace belongs to a different config: {self.workspace}"
                )
        else:
            resolved_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _hub_cache(self) -> str | None:
        return self.environment.get("HF_HUB_CACHE")

    def _collect_config(self) -> dict[str, Any]:
        data = self.config["data"]
        model = self.config["model"]
        streaming = self.config["streaming"]
        evaluation = self.config["evaluation"]
        hub_cache = self._hub_cache()
        return {
            "dataset": {
                "factory": (
                    "cluster_router_experiment.convomem:"
                    "dynamic_convomem_dataset_factory"
                ),
                "kwargs": {
                    "dataset_name": str(data["dataset_name"]),
                    "tokenizer_name": str(model["name"]),
                    "dataset_cache_dir": hub_cache,
                    "tokenizer_cache_dir": hub_cache,
                    "local_files_only": not bool(data.get("allow_network", False)),
                    "split": str(data.get("split", "test")),
                    "sequence_count": int(data["sequence_count"]),
                    "sequence_length": int(data["sequence_length"]),
                    "seed": int(self.config.get("seed", 13)),
                    "sampling_seed": int(
                        data.get("sampling_seed", self.config.get("seed", 13))
                    ),
                    "maximum_answer_tokens": int(
                        data.get("maximum_answer_tokens", 64)
                    ),
                    "maximum_future_horizon": int(
                        data.get("maximum_future_horizon", 64)
                    ),
                    "evidence_placement_bins": int(
                        data.get("evidence_placement_bins", 4)
                    ),
                    "block_size": int(streaming["block_size"]),
                    "local_context_length": int(streaming["local_context_length"]),
                },
            },
            "model": {
                "factory": (
                    "cluster_router_experiment.gemma4:"
                    "gemma4_cluster_router_model_factory"
                ),
                "kwargs": {
                    "checkpoint_path": str(_router_path(self.config)),
                    "model_name": str(model["name"]),
                    "model_cache_dir": hub_cache,
                    "local_files_only": not bool(model.get("allow_network", False)),
                    "device": str(model.get("device", "cuda")),
                    "dtype": str(model.get("dtype", "bfloat16")),
                    "router_device": str(self.config["router"].get("device", "cuda")),
                    "block_size": int(streaming["block_size"]),
                    "local_context_length": int(streaming["local_context_length"]),
                    "residual_layer": int(streaming.get("residual_layer", 40)),
                    "query_summary_length": int(
                        streaming.get("query_summary_length", 16)
                    ),
                    "prefill_chunk_size": int(
                        streaming.get("prefill_chunk_size", streaming["block_size"])
                    ),
                    "memory_budget_bytes_per_layer": None,
                    "memory_config": dict(self.config["memory"]),
                },
            },
            "output_dir": str(self.state_dir),
            "run": {
                "budgets": list(evaluation["budgets"]),
                "fixed_policy": str(evaluation.get("fixed_policy", "recent")),
                "oracle_signal": str(evaluation.get("oracle_signal", "evidence")),
                "maximum_samples": evaluation.get("maximum_samples"),
                "continue_on_error": bool(
                    evaluation.get("continue_on_error", False)
                ),
                "resume": bool(evaluation.get("resume", True)),
            },
        }

    def _write_generated_configs(self) -> tuple[Path, Path]:
        self.generated_config_dir.mkdir(parents=True, exist_ok=True)
        collect_path = self.generated_config_dir / "collect.json"
        metric_path = self.generated_config_dir / "metrics.json"
        collect_path.write_text(
            json.dumps(self._collect_config(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metric_path.write_text(
            json.dumps(dict(self.config["metrics"]), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return collect_path, metric_path

    def _run_command(self, stage: str, arguments: Iterable[str]) -> None:
        command = [sys.executable, "-u", *map(str, arguments)]
        self._emit("stage_start", stage=stage, command=command)
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=self.environment,
                check=True,
            )
        except BaseException as error:
            self._emit("stage_failed", stage=stage, error=repr(error))
            raise
        self._emit("stage_complete", stage=stage)

    def _collect(self, config_path: Path) -> None:
        self._run_command(
            "collect",
            ["-m", "cluster_router_validation", "collect", "--config", config_path],
        )

    def _metrics(self, config_path: Path) -> None:
        if all((self.metric_dir / name).is_file() for name in EXPORTED_METRIC_ARTIFACTS):
            self._emit("stage_skip", stage="metrics", reason="complete")
            return
        self._run_command(
            "metrics",
            [
                "-m",
                "cluster_router_validation",
                "metrics",
                "--state-dir",
                self.state_dir,
                "--output-dir",
                self.metric_dir,
                "--config",
                config_path,
            ],
        )

    def _export_metrics(self) -> None:
        if not self.use_tmp_workspace:
            return
        missing = [
            name
            for name in EXPORTED_METRIC_ARTIFACTS
            if not (self.metric_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "cannot export incomplete metric artifacts: " + ", ".join(missing)
            )
        destination = self.persistent_output_root / "metrics"
        destination.mkdir(parents=True, exist_ok=True)
        for name in EXPORTED_METRIC_ARTIFACTS:
            target = destination / name
            staging = destination / f".{name}.tmp-{os.getpid()}"
            shutil.copy2(self.metric_dir / name, staging)
            os.replace(staging, target)
            self._emit("artifact_exported", artifact=name, destination=str(target))

    def _cleanup_workspace(self) -> None:
        if not (self.use_tmp_workspace and self.cleanup_tmp_workspace):
            return
        expected_parent = "residualcache_cluster_router_validation"
        resolved = self.workspace.resolve()
        marker = resolved / "resolved_config.json"
        if resolved.parent.name != expected_parent or not marker.is_file():
            raise RuntimeError(f"refusing to clean unverified temporary path: {resolved}")
        stored = json.loads(marker.read_text(encoding="utf-8"))
        if stored.get("config_fingerprint") != self.fingerprint:
            raise RuntimeError(f"refusing to clean foreign temporary path: {resolved}")
        shutil.rmtree(resolved)

    def run(self) -> dict[str, Any]:
        self._initialize()
        collect_config, metric_config = self._write_generated_configs()
        self._emit(
            "pipeline_start",
            config_fingerprint=self.fingerprint,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            workspace=str(self.workspace),
            metrics_only_export=self.use_tmp_workspace,
        )
        self._collect(collect_config)
        self._metrics(metric_config)
        self._export_metrics()
        result = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "persistent_output_root": str(self.persistent_output_root),
            "metrics": {
                name: str(
                    (self.persistent_output_root if self.use_tmp_workspace else self.workspace)
                    / "metrics"
                    / name
                )
                for name in EXPORTED_METRIC_ARTIFACTS
            },
            "temporary_workspace_cleaned": bool(
                self.use_tmp_workspace and self.cleanup_tmp_workspace
            ),
        }
        self._emit("pipeline_complete", metrics=list(EXPORTED_METRIC_ARTIFACTS))
        self._cleanup_workspace()
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Config-driven clustered-router HPC validation pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "cluster_router_validation_convomem4096_hpc.json"
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = load_hpc_config(arguments.config)
    validate_hpc_config(config)
    pipeline = ClusterRouterHPCPipeline(config)
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "config_fingerprint": pipeline.fingerprint,
                    "router_path": str(_router_path(config)),
                    "model_name": config["model"]["name"],
                    "dataset_name": config["data"]["dataset_name"],
                    "persistent_output_root": str(pipeline.persistent_output_root),
                    "workspace": str(pipeline.workspace),
                    "use_tmp_workspace": pipeline.use_tmp_workspace,
                    "cleanup_tmp_workspace": pipeline.cleanup_tmp_workspace,
                    "persistent_artifacts": [
                        f"metrics/{name}" for name in EXPORTED_METRIC_ARTIFACTS
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(json.dumps(pipeline.run(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
