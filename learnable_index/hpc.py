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
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPLITS = ("train", "validation", "test")
DEFAULT_TMP_WORKSPACE_ROOT = "/tmp"
EXPORTED_TRAINING_ARTIFACTS = ("best.pt", "metrics.jsonl", "summary.json")


def load_hpc_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("HPC config schema_version must be 1")
    return config


def _output_path(config: dict[str, Any]) -> Path:
    value = Path(config["paths"]["output_root"])
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _require_hf_id(value: Any, field: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{field} must be a non-empty Hugging Face repository ID")
    if identifier.startswith(("/", "\\")) or Path(identifier).is_absolute() or (
        len(identifier) >= 3
        and identifier[1] == ":"
        and identifier[2] in {"/", "\\"}
    ):
        raise ValueError(f"{field} must be a Hugging Face repository ID, not a path")
    if identifier.startswith(("./", "../", ".\\", "..\\")):
        raise ValueError(f"{field} must be a Hugging Face repository ID, not a path")
    return identifier


def validate_hpc_config(config: dict[str, Any]) -> None:
    for section in (
        "paths",
        "model",
        "data",
        "collection",
        "router",
        "training",
        "evaluation",
        "replay",
        "stages",
    ):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    splits = config["data"].get("splits", {})
    if tuple(splits) != REQUIRED_SPLITS:
        raise ValueError(f"data.splits must be ordered as {REQUIRED_SPLITS}")
    for split in REQUIRED_SPLITS:
        split_config = splits[split]
        if int(split_config["sequences"]) <= 0:
            raise ValueError(f"{split} sequence count must be positive")
        if int(split_config["article_stride"]) <= 0:
            raise ValueError(f"{split} article_stride must be positive")
    if int(config["data"]["sequence_length"]) < 2:
        raise ValueError("data.sequence_length must be at least 2")
    _require_hf_id(config["model"].get("name"), "model.name")
    _require_hf_id(config["data"].get("dataset_name"), "data.dataset_name")
    if not str(config["data"].get("dataset_config", "")).strip():
        raise ValueError("data.dataset_config must be non-empty")
    if not 0 < float(config["training"]["validation_fraction"]) < 1:
        raise ValueError("training.validation_fraction must be strictly between 0 and 1")
    patience = config["training"].get("early_stopping_patience")
    if patience is not None and int(patience) <= 0:
        raise ValueError("training.early_stopping_patience must be positive when set")
    if int(config["collection"].get("progress_every", 25)) < 0:
        raise ValueError("collection.progress_every must be non-negative")
    if not isinstance(config["paths"].get("use_tmp_workspace", False), bool):
        raise ValueError("paths.use_tmp_workspace must be a boolean")
    temporary_root = str(
        config["paths"].get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT)
    ).strip()
    if not temporary_root:
        raise ValueError("paths.tmp_workspace_root must be non-empty")
    if not (temporary_root.startswith("/") or Path(temporary_root).is_absolute()):
        raise ValueError("paths.tmp_workspace_root must be an absolute path")


def _temporary_output_path(config: dict[str, Any]) -> Path:
    run_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(config["run_id"])
    ).strip("_")
    if not run_id:
        raise ValueError("run_id must contain at least one filesystem-safe character")
    job_id = os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_id = f"{job_id}-{array_id}" if array_id is not None else job_id
    temporary_root = Path(
        config["paths"].get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT)
    )
    return (
        temporary_root
        / "residualcache_learnable_index"
        / f"{run_id}-{task_id}"
    ).resolve()


def _fingerprint(config: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if key != "_config_path"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HPCPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        validate_hpc_config(config)
        self.config = config
        self.persistent_output_root = _output_path(config)
        self.use_tmp_workspace = bool(config["paths"].get("use_tmp_workspace", False))
        self.output_root = (
            _temporary_output_path(config)
            if self.use_tmp_workspace
            else self.persistent_output_root
        )
        self.fingerprint = _fingerprint(config)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": str(config.get("seed", 13)),
            }
        )
        current_pythonpath = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(PROJECT_ROOT), current_pythonpath) if part
        )
        if self.use_tmp_workspace:
            cache_root = self.output_root / "cache"
            huggingface_root = cache_root / "huggingface"
            self.environment.update(
                {
                    "HF_HOME": str(huggingface_root),
                    "HF_HUB_CACHE": str(huggingface_root / "hub"),
                    "HF_DATASETS_CACHE": str(huggingface_root / "datasets"),
                    "TORCH_HOME": str(cache_root / "torch"),
                    "XDG_CACHE_HOME": str(cache_root),
                }
            )

    def _emit(self, event: str, **fields: Any) -> None:
        row = {"time": _utc_now(), "event": event, **fields}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "pipeline_events.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _initialize(self) -> None:
        if self.use_tmp_workspace and self.persistent_output_root.exists():
            allowed = {
                self.persistent_output_root / "training" / name
                for name in EXPORTED_TRAINING_ARTIFACTS
            }
            unexpected = [
                path
                for path in self.persistent_output_root.rglob("*")
                if path.is_file() and path not in allowed
            ]
            if unexpected:
                raise RuntimeError(
                    "temporary-workspace destination contains non-export artifacts: "
                    + ", ".join(str(path) for path in unexpected[:5])
                )
        self.output_root.mkdir(parents=True, exist_ok=True)
        path = self.output_root / "resolved_config.json"
        payload = {
            "config_fingerprint": self.fingerprint,
            "source_config": self.config["_config_path"],
            "resolved_config": {
                key: value for key, value in self.config.items() if key != "_config_path"
            },
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"output root belongs to a different config: {self.output_root}"
                )
        else:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _export_training_artifacts(self) -> None:
        if not self.use_tmp_workspace:
            return
        source = self.output_root / "training"
        missing = [name for name in EXPORTED_TRAINING_ARTIFACTS if not (source / name).is_file()]
        if missing:
            raise RuntimeError(
                "cannot export incomplete training artifacts: " + ", ".join(missing)
            )
        destination = self.persistent_output_root / "training"
        destination.mkdir(parents=True, exist_ok=True)
        for name in EXPORTED_TRAINING_ARTIFACTS:
            target = destination / name
            staging = destination / f".{name}.tmp-{os.getpid()}"
            shutil.copy2(source / name, staging)
            os.replace(staging, target)
            self._emit("artifact_exported", artifact=name, destination=str(target))

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

    def _enabled(self, stage: str) -> bool:
        return bool(self.config["stages"].get(stage, True))

    def _input_path(self, split: str) -> Path:
        count = int(self.config["data"]["splits"][split]["sequences"])
        length = int(self.config["data"]["sequence_length"])
        return self.output_root / "inputs" / f"{split}_{count}x{length}.jsonl"

    def _collection_dir(self, split: str) -> Path:
        return self.output_root / "collection" / split

    def _expected_samples(self, split: str) -> int:
        data = self.config["data"]
        collection = self.config["collection"]
        first = (
            int(collection["local_context_length"])
            + int(collection["minimum_candidate_blocks"])
            * int(collection["block_size"])
            - 1
        )
        last = (
            int(data["sequence_length"])
            - int(collection["future_horizon"])
            - 2
        )
        count_per_sequence = (
            0
            if first > last
            else (last - first) // int(collection["retrieval_interval"]) + 1
        )
        return count_per_sequence * int(data["splits"][split]["sequences"])

    def prepare_inputs(self) -> None:
        if not self._enabled("prepare"):
            return
        for split in REQUIRED_SPLITS:
            split_config = self.config["data"]["splits"][split]
            output = self._input_path(split)
            manifest = output.with_suffix(".manifest.json")
            if output.exists() and manifest.exists():
                stored = json.loads(manifest.read_text(encoding="utf-8"))
                if (
                    stored.get("split") != split
                    or int(stored.get("sequence_count", -1))
                    != int(split_config["sequences"])
                    or int(stored.get("sequence_length", -1))
                    != int(self.config["data"]["sequence_length"])
                ):
                    raise RuntimeError(f"prepared input manifest mismatch: {manifest}")
                self._emit("stage_skip", stage=f"prepare:{split}", reason="complete")
                continue
            if output.exists() or manifest.exists():
                raise RuntimeError(f"partial prepared input exists for split={split}")
            self._run_command(
                f"prepare:{split}",
                [
                    "-m",
                    "learnable_index.prepare_wikitext",
                    "--dataset-name",
                    str(self.config["data"]["dataset_name"]),
                    "--dataset-config",
                    str(self.config["data"]["dataset_config"]),
                    "--tokenizer",
                    str(self.config["model"]["name"]),
                    "--output",
                    str(output),
                    "--split",
                    split,
                    "--sequence-length",
                    str(self.config["data"]["sequence_length"]),
                    "--sequences",
                    str(split_config["sequences"]),
                    "--article-stride",
                    str(split_config["article_stride"]),
                    "--seed",
                    str(self.config.get("seed", 13)),
                    *self._huggingface_arguments(),
                ],
            )

    def _huggingface_arguments(self) -> list[str]:
        arguments: list[str] = []
        cache_dir = self.config["data"].get("cache_dir")
        if cache_dir:
            arguments.extend(["--cache-dir", str(cache_dir)])
        if bool(self.config["data"].get("allow_network", False)):
            arguments.append("--allow-network")
        return arguments

    def _model_arguments(self) -> list[str]:
        arguments = [
            "--model-name",
            str(self.config["model"]["name"]),
            "--model-device",
            str(self.config["model"]["device"]),
            "--dtype",
            str(self.config["model"]["dtype"]),
        ]
        if bool(self.config["model"].get("allow_network", False)):
            arguments.append("--allow-network")
        return arguments

    def _collection_arguments(self) -> list[str]:
        config = self.config["collection"]
        teacher_layers = config.get("teacher_layers", "all")
        teacher_heads = config.get("teacher_heads", "all")
        arguments = [
            "--max-tokens",
            str(self.config["data"]["sequence_length"]),
            "--local-context-length",
            str(config["local_context_length"]),
            "--block-size",
            str(config["block_size"]),
            "--future-horizon",
            str(config["future_horizon"]),
            "--retrieval-interval",
            str(config["retrieval_interval"]),
            "--minimum-candidate-blocks",
            str(config["minimum_candidate_blocks"]),
            "--maximum-candidate-blocks",
            str(config["maximum_candidate_blocks"]),
            "--residual-layer",
            str(config["residual_layer"]),
            "--query-summary",
            str(config["query_summary"]),
            "--query-summary-length",
            str(config["query_summary_length"]),
            "--teacher-layers",
            _index_specification(teacher_layers),
            "--teacher-heads",
            _index_specification(teacher_heads),
            "--future-reduction",
            str(config["future_reduction"]),
            "--progress-every",
            str(config.get("progress_every", 25)),
        ]
        if bool(config.get("length_normalize_blocks", False)):
            arguments.append("--length-normalize-blocks")
        return arguments

    def collect(self) -> None:
        if not self._enabled("collect"):
            return
        for split in REQUIRED_SPLITS:
            output = self._collection_dir(split)
            manifest_path = output / "collection_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_sequences = int(
                    self.config["data"]["splits"][split]["sequences"]
                )
                if (
                    int(manifest.get("sequence_count", -1)) != expected_sequences
                    or int(manifest.get("sample_count", -1)) != self._expected_samples(split)
                ):
                    raise RuntimeError(f"collection manifest mismatch: {manifest_path}")
                self._emit("stage_skip", stage=f"collect:{split}", reason="complete")
                continue
            if not self._input_path(split).exists():
                raise FileNotFoundError(f"prepared input is missing for split={split}")
            self._run_command(
                f"collect:{split}",
                [
                    "-m",
                    "learnable_index",
                    "collect",
                    *self._model_arguments(),
                    "--input-jsonl",
                    str(self._input_path(split)),
                    "--output-dir",
                    str(output),
                    *self._collection_arguments(),
                ],
            )

    def train(self) -> None:
        if not self._enabled("train"):
            return
        output = self.output_root / "training"
        if (output / "summary.json").exists() and (output / "best.pt").exists():
            self._emit("stage_skip", stage="train", reason="complete")
            return
        router = self.config["router"]
        train = self.config["training"]
        arguments = [
            "-m",
            "learnable_index",
            "train",
            "--dataset-dir",
            str(self._collection_dir("train") / "dataset"),
            "--output-dir",
            str(output),
            "--projection-dim",
            str(router["projection_dim"]),
            "--hidden-dim",
            str(router["hidden_dim"]),
            "--depth",
            str(router["depth"]),
            "--dropout",
            str(router["dropout"]),
            "--temperature",
            str(router["temperature"]),
            "--epochs",
            str(train["epochs"]),
            "--batch-size",
            str(train["batch_size"]),
            "--learning-rate",
            str(train["learning_rate"]),
            "--weight-decay",
            str(train["weight_decay"]),
            "--validation-fraction",
            str(train["validation_fraction"]),
            "--top-n",
            str(train["top_n"]),
            "--seed",
            str(self.config.get("seed", 13)),
            "--device",
            str(train["device"]),
        ]
        if train.get("early_stopping_patience") is not None:
            arguments.extend(
                ["--early-stopping-patience", str(train["early_stopping_patience"])]
            )
        self._run_command("train", arguments)

    def evaluate(self) -> None:
        if not self._enabled("evaluate"):
            return
        checkpoint = self.output_root / "training" / "best.pt"
        for split in ("validation", "test"):
            output = self.output_root / "evaluation" / f"{split}.json"
            if output.exists():
                self._emit("stage_skip", stage=f"evaluate:{split}", reason="complete")
                continue
            self._run_command(
                f"evaluate:{split}",
                [
                    "-m",
                    "learnable_index",
                    "evaluate",
                    "--dataset-dir",
                    str(self._collection_dir(split) / "dataset"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--device",
                    str(self.config["evaluation"]["device"]),
                    "--top-n",
                    str(self.config["evaluation"]["top_n"]),
                ],
            )

    def replay(self) -> None:
        if not self._enabled("replay") or not bool(self.config["replay"]["enabled"]):
            return
        replay = self.config["replay"]
        split = str(replay["split"])
        output = self.output_root / "replay" / split
        if (output / "summary.json").exists():
            self._emit("stage_skip", stage=f"replay:{split}", reason="complete")
            return
        arguments = [
            "-m",
            "learnable_index",
            "replay",
            *self._model_arguments(),
            "--collection-dir",
            str(self._collection_dir(split)),
            "--checkpoint",
            str(self.output_root / "training" / "best.pt"),
            "--output-dir",
            str(output),
            "--policy",
            str(replay["policy"]),
            "--replay-top-n",
            str(replay["top_n"]),
            "--score-threshold",
            str(replay.get("score_threshold", 0.0)),
            "--router-device",
            str(replay["router_device"]),
        ]
        if replay.get("maximum_samples") is not None:
            arguments.extend(["--maximum-replay-samples", str(replay["maximum_samples"])])
        self._run_command(f"replay:{split}", arguments)

    def run(self) -> None:
        self._initialize()
        self._emit(
            "pipeline_start",
            config_fingerprint=self.fingerprint,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        self.prepare_inputs()
        self.collect()
        self.train()
        self.evaluate()
        self.replay()
        manifest = {
            "schema_version": 1,
            "completed_at": _utc_now(),
            "config_fingerprint": self.fingerprint,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "paths": {
                "training": "training",
                "evaluation": "evaluation",
                "replay": "replay",
                "collection": "collection",
            },
        }
        (self.output_root / "hpc_pipeline_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._export_training_artifacts()
        self._emit("pipeline_complete", manifest="hpc_pipeline_manifest.json")
        if self.use_tmp_workspace:
            shutil.rmtree(self.output_root)


def _index_specification(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(index) for index in value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumeable learnable_index HPC pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "learnable_index_wikitext4096_hpc.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = load_hpc_config(arguments.config)
    validate_hpc_config(config)
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "config_fingerprint": _fingerprint(config),
                    "model_name": config["model"]["name"],
                    "dataset_name": config["data"]["dataset_name"],
                    "dataset_config": config["data"]["dataset_config"],
                    "output_root": str(_output_path(config)),
                    "use_tmp_workspace": bool(
                        config["paths"].get("use_tmp_workspace", False)
                    ),
                    "tmp_workspace_root": str(
                        config["paths"].get(
                            "tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    HPCPipeline(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
