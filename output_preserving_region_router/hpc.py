from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from learnable_index.hpc import (
    HPCPipeline as PreparationPipeline,
    PROJECT_ROOT,
    REQUIRED_SPLITS,
    _fingerprint,
    _output_path,
    _utc_now,
    load_hpc_config,
    validate_hpc_config as validate_preparation_config,
)


EXPORTED_TRAINING_ARTIFACTS = (
    "best.pt",
    "final.pt",
    "metrics.jsonl",
    "summary.json",
    "run_config.json",
)
EXPORTED_EVALUATION_ARTIFACTS = (
    "validation.json",
    "validation_qa.json",
    "test.json",
    "test_qa.json",
)


def validate_hpc_config(config: dict[str, Any]) -> None:
    validate_preparation_config(config)
    if config["replay"] != {"enabled": False}:
        raise ValueError("output-preserving training has no separate replay stage")
    collection = config["collection"]
    for name in (
        "local_context_length",
        "block_size",
        "future_horizon",
        "retrieval_interval",
        "minimum_candidate_blocks",
        "residual_layer",
        "query_summary_length",
    ):
        if int(collection[name]) <= 0:
            raise ValueError(f"collection.{name} must be positive")
    if int(collection["local_context_length"]) % int(collection["block_size"]):
        raise ValueError("local context must be divisible by block size")
    if collection.get("retrieval_point_policy") not in {"interval", "metadata"}:
        raise ValueError("unsupported retrieval-point policy")
    if collection.get("query_summary") not in {"last", "mean"}:
        raise ValueError("unsupported query summary")
    if collection.get("maximum_candidate_blocks") is not None:
        raise ValueError(
            "the region router must observe all completed candidate blocks; "
            "maximum_candidate_blocks must be null"
        )
    router = config["router"]
    for name in ("feature_dim", "hidden_dim", "depth"):
        if int(router[name]) <= 0:
            raise ValueError(f"router.{name} must be positive")
    for name in (
        "minimum_scale",
        "radius",
        "gate_temperature",
        "gate_epsilon",
    ):
        if float(router[name]) <= 0:
            raise ValueError(f"router.{name} must be positive")
    if not 0 <= float(router.get("dropout", 0.0)) < 1:
        raise ValueError("router.dropout must be in [0, 1)")
    if float(router["gate_epsilon"]) >= 1:
        raise ValueError("router.gate_epsilon must be less than one")
    objective = config.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("missing objective section")
    if float(objective["maximum_excess_output_kl"]) < 0:
        raise ValueError("objective.maximum_excess_output_kl must be non-negative")
    if float(objective.get("output_temperature", 1.0)) <= 0:
        raise ValueError("objective.output_temperature must be positive")
    for name in ("preservation_weight", "sparsity_weight"):
        if float(objective[name]) <= 0:
            raise ValueError(f"objective.{name} must be positive")
    if float(objective.get("gate_entropy_weight", 0.0)) < 0:
        raise ValueError("objective.gate_entropy_weight must be non-negative")
    training = config["training"]
    if int(training.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    if int(training.get("prefill_chunk_size", 256)) <= 0:
        raise ValueError("training.prefill_chunk_size must be positive")
    if float(training.get("final_gate_temperature", 0.05)) <= 0:
        raise ValueError("training.final_gate_temperature must be positive")
    if not bool(config["evaluation"].get("qa_enabled", True)):
        raise ValueError("evaluation.qa_enabled must remain true for system validation")


class HPCPipeline(PreparationPipeline):
    """Prepare dynamic ConvoMem inputs and run the independent region system."""

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
                    "temporary destination contains non-export artifacts: "
                    + ", ".join(str(path) for path in unexpected[:5])
                )
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_fingerprint": self.fingerprint,
            "source_config": self.config["_config_path"],
            "resolved_config": {
                key: value for key, value in self.config.items() if key != "_config_path"
            },
        }
        (self.output_root / "resolved_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _plan_arguments(self) -> list[str]:
        collection = self.config["collection"]
        arguments = [
            "--local-context-length",
            str(collection["local_context_length"]),
            "--block-size",
            str(collection["block_size"]),
            "--future-horizon",
            str(collection["future_horizon"]),
            "--retrieval-interval",
            str(collection["retrieval_interval"]),
            "--retrieval-point-policy",
            str(collection["retrieval_point_policy"]),
            "--minimum-candidate-blocks",
            str(collection["minimum_candidate_blocks"]),
            "--residual-layer",
            str(collection["residual_layer"]),
            "--query-summary",
            str(collection["query_summary"]),
            "--query-summary-length",
            str(collection["query_summary_length"]),
        ]
        return arguments

    def train(self) -> None:
        if not self._enabled("train"):
            return
        output = self.output_root / "training"
        if (output / "summary.json").is_file() and (output / "best.pt").is_file():
            self._emit("stage_skip", stage="train", reason="complete")
            return
        router = self.config["router"]
        objective = self.config["objective"]
        training = self.config["training"]
        arguments = [
            "-m",
            "output_preserving_region_router",
            "train",
            *self._model_arguments(),
            *self._plan_arguments(),
            "--input-jsonl",
            str(self._input_path("train")),
            "--output-dir",
            str(output),
            "--maximum-tokens",
            str(self.config["data"]["sequence_length"]),
            "--feature-dim",
            str(router["feature_dim"]),
            "--hidden-dim",
            str(router["hidden_dim"]),
            "--depth",
            str(router["depth"]),
            "--dropout",
            str(router["dropout"]),
            "--minimum-scale",
            str(router["minimum_scale"]),
            "--radius",
            str(router["radius"]),
            "--gate-temperature",
            str(router["gate_temperature"]),
            "--final-gate-temperature",
            str(training.get("final_gate_temperature", 0.05)),
            "--gate-epsilon",
            str(router["gate_epsilon"]),
            "--output-temperature",
            str(objective.get("output_temperature", 1.0)),
            "--maximum-excess-output-kl",
            str(objective["maximum_excess_output_kl"]),
            "--preservation-weight",
            str(objective["preservation_weight"]),
            "--sparsity-weight",
            str(objective["sparsity_weight"]),
            "--gate-entropy-weight",
            str(objective.get("gate_entropy_weight", 0.0)),
            "--epochs",
            str(training["epochs"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--gradient-accumulation-steps",
            str(training.get("gradient_accumulation_steps", 1)),
            "--gradient-clip-norm",
            str(training.get("gradient_clip_norm", 1.0)),
            "--validation-fraction",
            str(training["validation_fraction"]),
            "--seed",
            str(self.config.get("seed", 13)),
            "--router-device",
            str(training["device"]),
            "--prefill-chunk-size",
            str(training.get("prefill_chunk_size", 256)),
            "--progress-every",
            str(training.get("progress_every", 1)),
        ]
        if training.get("early_stopping_patience") is not None:
            arguments.extend(
                [
                    "--early-stopping-patience",
                    str(training["early_stopping_patience"]),
                ]
            )
        for field, flag in (
            ("maximum_train_samples", "--maximum-train-samples"),
            ("maximum_validation_samples", "--maximum-validation-samples"),
        ):
            if training.get(field) is not None:
                arguments.extend([flag, str(training[field])])
        self._run_command("train", arguments)

    def evaluate(self, split: str) -> None:
        if not self._enabled("evaluate"):
            return
        output = self.output_root / "evaluation" / f"{split}.json"
        qa_output = self.output_root / "evaluation" / f"{split}_qa.json"
        if output.is_file() and qa_output.is_file():
            self._emit("stage_skip", stage=f"evaluate:{split}", reason="complete")
            return
        if output.is_file() or qa_output.is_file():
            raise RuntimeError(f"partial evaluation output exists for split={split}")
        evaluation = self.config["evaluation"]
        arguments = [
            "-m",
            "output_preserving_region_router",
            "evaluate",
            *self._model_arguments(),
            "--input-jsonl",
            str(self._input_path(split)),
            "--checkpoint",
            str(self.output_root / "training" / "best.pt"),
            "--output",
            str(output),
            "--maximum-tokens",
            str(self.config["data"]["sequence_length"]),
            "--router-device",
            str(evaluation["device"]),
            "--progress-every",
            str(evaluation.get("progress_every", 1)),
        ]
        if evaluation.get("maximum_samples") is not None:
            arguments.extend(
                ["--maximum-samples", str(evaluation["maximum_samples"])]
            )
        if bool(evaluation.get("qa_enabled", True)):
            arguments.extend(
                [
                    "--qa-output",
                    str(qa_output),
                    "--qa-samples-output",
                    str(
                        self.output_root
                        / "evaluation"
                        / f"{split}_qa_samples.jsonl"
                    ),
                    "--maximum-new-tokens",
                    str(evaluation.get("maximum_new_tokens", 64)),
                    "--qa-prefill-chunk-size",
                    str(evaluation.get("prefill_chunk_size", 256)),
                ]
            )
            if evaluation.get("qa_maximum_samples") is not None:
                arguments.extend(
                    [
                        "--qa-maximum-samples",
                        str(evaluation["qa_maximum_samples"]),
                    ]
                )
        self._run_command(f"evaluate:{split}", arguments)

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
                    f"cannot export incomplete {directory}: " + ", ".join(missing)
                )
            destination = self.persistent_output_root / directory
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                target = destination / name
                staging = destination / f".{name}.tmp-{os.getpid()}"
                shutil.copy2(source / name, staging)
                os.replace(staging, target)
                self._emit(
                    "artifact_exported",
                    artifact=f"{directory}/{name}",
                    destination=str(target),
                )

    def run(self) -> None:
        self._initialize()
        self._emit(
            "pipeline_start",
            model_kind="output_preserving_gaussian_region_router",
            config_fingerprint=self.fingerprint,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        self.prepare_inputs(("train",))
        self.train()
        self._cleanup_prepared_input("train")
        for split in ("validation", "test"):
            self.prepare_inputs((split,))
            self.evaluate(split)
            self._cleanup_prepared_input(split)
        manifest = {
            "schema_version": 1,
            "model_kind": "output_preserving_gaussian_region_router",
            "training_protocol": "full_logits_soft_block_gate_streaming_kv_v1",
            "completed_at": _utc_now(),
            "config_fingerprint": self.fingerprint,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "paths": {"training": "training", "evaluation": "evaluation"},
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
    parser = argparse.ArgumentParser(
        description="Output-preserving Gaussian region router HPC pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "output_preserving_region_router_convomem4096_smoke_hpc.json",
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
                    "model_kind": "output_preserving_gaussian_region_router",
                },
                indent=2,
            )
        )
        return 0
    HPCPipeline(config).run()
    return 0


__all__ = ["HPCPipeline", "main", "validate_hpc_config"]
