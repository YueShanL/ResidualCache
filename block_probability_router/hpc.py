from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from learnable_index.hpc import (
    EXPORTED_TRAINING_ARTIFACTS,
    HPCPipeline as AlignedCollectionHPCPipeline,
    PROJECT_ROOT,
    REQUIRED_SPLITS,
    _fingerprint,
    _output_path,
    _utc_now,
    load_hpc_config,
    validate_hpc_config as validate_collection_hpc_config,
)


EXPORTED_EVALUATION_ARTIFACTS = ("validation.json", "test.json")


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
    validate_collection_hpc_config(config)
    if config["replay"] != {"enabled": False}:
        raise ValueError("the probability-router training pipeline does not implement replay")
    router = config["router"]
    for name in ("feature_dim", "hidden_dim", "depth"):
        if int(router[name]) <= 0:
            raise ValueError(f"router.{name} must be positive")
    if float(router.get("positive_floor", 1e-6)) <= 0:
        raise ValueError("router.positive_floor must be positive")
    if float(router.get("normalization_epsilon", 1e-12)) <= 0:
        raise ValueError("router.normalization_epsilon must be positive")
    tolerances = tuple(
        float(value) for value in config["training"]["missing_mass_tolerances"]
    )
    if tuple(sorted(set(tolerances))) != tolerances or any(
        not 0 < value < 1 for value in tolerances
    ):
        raise ValueError(
            "training.missing_mass_tolerances must be sorted, unique, and in (0, 1)"
        )


class HPCPipeline(AlignedCollectionHPCPipeline):
    """End-to-end runner with shared collection and an independent router backend."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_hpc_config(config)
        super().__init__(config)
        self.fingerprint = _fingerprint(config)

    def _initialize(self) -> None:
        if self.use_tmp_workspace and self.persistent_output_root.exists():
            allowed = {
                self.persistent_output_root / "training" / name
                for name in EXPORTED_TRAINING_ARTIFACTS
            } | {
                self.persistent_output_root / "evaluation" / name
                for name in EXPORTED_EVALUATION_ARTIFACTS
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
                raise RuntimeError(f"output root belongs to a different config: {self.output_root}")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
            "block_probability_router",
            "train",
            "--dataset-dir",
            str(self._collection_dir("train") / "dataset"),
            "--output-dir",
            str(output),
            "--feature-dim",
            str(router["feature_dim"]),
            "--hidden-dim",
            str(router["hidden_dim"]),
            "--depth",
            str(router["depth"]),
            "--dropout",
            str(router["dropout"]),
            "--positive-floor",
            str(router.get("positive_floor", 1e-6)),
            "--normalization-epsilon",
            str(router.get("normalization_epsilon", 1e-12)),
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
            "--missing-mass-tolerances",
            ",".join(str(value) for value in train["missing_mass_tolerances"]),
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
        evaluation = self.config["evaluation"]
        tolerances = evaluation.get(
            "missing_mass_tolerances",
            self.config["training"]["missing_mass_tolerances"],
        )
        for split in ("validation", "test"):
            output = self.output_root / "evaluation" / f"{split}.json"
            if output.exists():
                self._emit("stage_skip", stage=f"evaluate:{split}", reason="complete")
                continue
            self._run_command(
                f"evaluate:{split}",
                [
                    "-m",
                    "block_probability_router",
                    "evaluate",
                    "--dataset-dir",
                    str(self._collection_dir(split) / "dataset"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--device",
                    str(evaluation["device"]),
                    "--top-n",
                    str(evaluation["top_n"]),
                    "--missing-mass-tolerances",
                    ",".join(str(value) for value in tolerances),
                ],
            )

    def _export_artifacts(self) -> None:
        if not self.use_tmp_workspace:
            return
        groups = (
            ("training", EXPORTED_TRAINING_ARTIFACTS),
            ("evaluation", EXPORTED_EVALUATION_ARTIFACTS),
        )
        for directory, names in groups:
            source = self.output_root / directory
            missing = [name for name in names if not (source / name).is_file()]
            if missing:
                raise RuntimeError(
                    f"cannot export incomplete {directory} artifacts: " + ", ".join(missing)
                )
            destination = self.persistent_output_root / directory
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                target = destination / name
                staging = destination / f".{name}.tmp-{os.getpid()}"
                shutil.copy2(source / name, staging)
                os.replace(staging, target)
                self._emit("artifact_exported", artifact=f"{directory}/{name}", destination=str(target))

    def run(self) -> None:
        self._initialize()
        self._emit(
            "pipeline_start",
            model_kind="positive_block_probability_router",
            config_fingerprint=self.fingerprint,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        if self._prepared_inputs_are_ephemeral():
            for split in REQUIRED_SPLITS:
                self.prepare_inputs((split,))
                self.collect((split,))
                if (self._collection_dir(split) / "collection_manifest.json").is_file():
                    self._cleanup_prepared_input(split)
        else:
            self.prepare_inputs()
            self.collect()
        self.train()
        self.evaluate()
        manifest = {
            "schema_version": 1,
            "model_kind": "positive_block_probability_router",
            "completed_at": _utc_now(),
            "config_fingerprint": self.fingerprint,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "paths": {
                "training": "training",
                "evaluation": "evaluation",
                "collection": "collection",
            },
        }
        (self.output_root / "hpc_pipeline_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._export_artifacts()
        self._emit("pipeline_complete", manifest="hpc_pipeline_manifest.json")
        if self.use_tmp_workspace:
            shutil.rmtree(self.output_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Positive block-probability router HPC pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "block_probability_router_convomem4096_hpc.json",
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
                    "config": str(arguments.config.resolve()),
                    "output_root": str(_output_path(config)),
                    "model_kind": "positive_block_probability_router",
                },
                indent=2,
            )
        )
        return 0
    HPCPipeline(config).run()
    return 0
