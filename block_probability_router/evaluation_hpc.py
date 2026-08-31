from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from learnable_index.hpc import (
    DEFAULT_TMP_WORKSPACE_ROOT,
    PROJECT_ROOT,
    _fingerprint,
    _output_path,
    _require_hf_id,
    _utc_now,
    load_hpc_config,
)

from .streaming_collection import STUDENT_STATE_PROTOCOL


OUTPUT_NAME = "metrics.json"
QA_OUTPUT_NAME = "qa_metrics.json"
QA_SAMPLES_NAME = "qa_samples.jsonl"


def _epsilon_values(config: dict[str, Any]) -> tuple[float, ...]:
    if "epsilon" not in config["evaluation"]:
        raise ValueError("evaluation.epsilon is required")
    raw = config["evaluation"].get("epsilon")
    values = raw if isinstance(raw, list) else [raw]
    try:
        tolerances = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation.epsilon must be a number or list of numbers") from error
    if not tolerances:
        raise ValueError("evaluation.epsilon cannot be empty")
    if any(not 0 < value < 1 for value in tolerances):
        raise ValueError("evaluation.epsilon values must be in (0, 1)")
    if tuple(sorted(set(tolerances))) != tolerances:
        raise ValueError("evaluation.epsilon values must be sorted and unique")
    return tolerances


def _checkpoint_path(config: dict[str, Any]) -> Path:
    value = str(config["router"].get("checkpoint", "")).strip()
    if not value:
        raise ValueError("router.checkpoint must be non-empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _max_block(config: dict[str, Any]) -> int:
    raw = config["evaluation"].get("max_block", -1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("evaluation.max_block must be an integer")
    if raw != -1 and raw <= 0:
        raise ValueError("evaluation.max_block must be -1 or a positive integer")
    return raw


def _temporary_output_path(config: dict[str, Any]) -> Path:
    run_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(config.get("run_id", ""))
    ).strip("_")
    if not run_id:
        raise ValueError("run_id must contain at least one filesystem-safe character")
    job_id = os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}"
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_id = f"{job_id}-{array_id}" if array_id is not None else job_id
    root = Path(
        config["paths"].get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT)
    ).resolve()
    return (root / "residualcache_probability_evaluation" / f"{run_id}-{task_id}").resolve()


def validate_evaluation_hpc_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("HPC config schema_version must be 1")
    for section in (
        "paths",
        "model",
        "router",
        "data",
        "collection",
        "evaluation",
        "qa",
    ):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    if not str(config.get("run_id", "")).strip():
        raise ValueError("run_id must be non-empty")

    paths = config["paths"]
    if not str(paths.get("output_root", "")).strip():
        raise ValueError("paths.output_root must be non-empty")
    if not isinstance(paths.get("use_tmp_workspace", False), bool):
        raise ValueError("paths.use_tmp_workspace must be a boolean")
    temporary_root = str(
        paths.get("tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT)
    ).strip()
    if not temporary_root or not (
        temporary_root.startswith("/") or Path(temporary_root).is_absolute()
    ):
        raise ValueError("paths.tmp_workspace_root must be an absolute path")

    _require_hf_id(config["model"].get("name"), "model.name")
    if str(config["model"].get("dtype", "")) not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise ValueError("model.dtype is invalid")
    if not str(config["model"].get("device", "")).strip():
        raise ValueError("model.device must be non-empty")
    data = config["data"]
    if str(data.get("source", "convomem")) != "convomem":
        raise ValueError("this evaluation runner currently requires data.source='convomem'")
    _require_hf_id(data.get("dataset_name"), "data.dataset_name")
    if str(data.get("split", "test")) not in {"train", "validation", "test"}:
        raise ValueError("data.split must be train, validation, or test")
    if int(data.get("sequences", 0)) <= 0:
        raise ValueError("data.sequences must be positive")
    if int(data.get("sequence_length", 0)) < 512:
        raise ValueError("data.sequence_length must be at least 512")
    if not isinstance(data.get("persist_prepared_inputs", False), bool):
        raise ValueError("data.persist_prepared_inputs must be a boolean")
    if str(data.get("evidence_placement", "stratified_random")) not in {
        "fixed_start",
        "stratified_random",
    }:
        raise ValueError("data.evidence_placement must be fixed_start or stratified_random")
    for field in (
        "maximum_answer_tokens",
        "maximum_future_horizon",
        "evidence_placement_bins",
        "placement_block_size",
        "retrieval_local_context_length",
    ):
        if int(data[field]) <= 0:
            raise ValueError(f"data.{field} must be positive")

    collection = config["collection"]
    for field in (
        "local_context_length",
        "block_size",
        "future_horizon",
        "retrieval_interval",
        "minimum_candidate_blocks",
        "query_summary_length",
    ):
        if int(collection[field]) <= 0:
            raise ValueError(f"collection.{field} must be positive")
    maximum_candidates = collection.get("maximum_candidate_blocks")
    if maximum_candidates is not None and int(maximum_candidates) <= 0:
        raise ValueError("collection.maximum_candidate_blocks must be positive when set")
    if int(collection.get("progress_every", 25)) < 0:
        raise ValueError("collection.progress_every must be non-negative")
    if int(collection.get("residual_layer", -1)) < 0:
        raise ValueError("collection.residual_layer must be non-negative")
    if str(collection.get("query_summary", "")) not in {"last", "mean"}:
        raise ValueError("collection.query_summary must be last or mean")
    if str(collection.get("future_reduction", "")) not in {"mean", "sum"}:
        raise ValueError("collection.future_reduction must be mean or sum")
    if str(collection.get("retrieval_point_policy", "metadata")) != "metadata":
        raise ValueError("this evaluation runner requires metadata retrieval points")
    if int(collection["local_context_length"]) != int(
        data["retrieval_local_context_length"]
    ):
        raise ValueError("collection and data local-context lengths must match")
    if int(collection["block_size"]) != int(data["placement_block_size"]):
        raise ValueError("collection block size must match evidence placement block size")
    if int(collection["local_context_length"]) % int(collection["block_size"]):
        raise ValueError("collection local context must be divisible by block size")
    if int(collection["future_horizon"]) > int(data["maximum_future_horizon"]):
        raise ValueError("collection future horizon exceeds synthesized maximum")
    teacher_prefill = collection.get("teacher_prefill_chunk_size")
    if teacher_prefill is not None and int(teacher_prefill) <= 0:
        raise ValueError("collection.teacher_prefill_chunk_size must be positive when set")

    evaluation = config["evaluation"]
    _epsilon_values(config)
    _max_block(config)
    if int(evaluation.get("top_n", 4)) <= 0:
        raise ValueError("evaluation.top_n must be positive")
    if not str(evaluation.get("device", "cpu")).strip():
        raise ValueError("evaluation.device must be non-empty")

    qa = config["qa"]
    if not isinstance(qa.get("enabled", True), bool):
        raise ValueError("qa.enabled must be a boolean")
    for field in ("maximum_new_tokens", "prefill_chunk_size", "bootstrap_iterations"):
        if int(qa.get(field, 0)) <= 0:
            raise ValueError(f"qa.{field} must be positive")
    if int(qa.get("progress_every", 10)) < 0:
        raise ValueError("qa.progress_every must be non-negative")
    maximum_qa_samples = qa.get("maximum_samples")
    if maximum_qa_samples is not None:
        if int(maximum_qa_samples) <= 0:
            raise ValueError("qa.maximum_samples must be positive when set")
        if int(maximum_qa_samples) > int(data["sequences"]):
            raise ValueError("qa.maximum_samples cannot exceed data.sequences")
    if not str(qa.get("router_device", "cpu")).strip():
        raise ValueError("qa.router_device must be non-empty")
    _checkpoint_path(config)


def _index_specification(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(index) for index in value)


class EvaluationHPCPipeline:
    """Synthesize, collect, score, and autoregressively QA one router checkpoint."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_evaluation_hpc_config(config)
        self.config = config
        self.fingerprint = _fingerprint(config)
        self.persistent_output_root = _output_path(config)
        self.use_tmp_workspace = bool(config["paths"].get("use_tmp_workspace", False))
        self.output_root = (
            _temporary_output_path(config)
            if self.use_tmp_workspace
            else self.persistent_output_root
        )
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": str(config.get("seed", 13)),
            }
        )
        if not self.environment.get("HF_TOKEN") and not self.environment.get(
            "HF_TOKEN_PATH"
        ):
            original_hf_home = Path(
                self.environment.get(
                    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
                )
            ).expanduser()
            token_path = original_hf_home / "token"
            if token_path.is_file():
                self.environment["HF_TOKEN_PATH"] = str(token_path.resolve())
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

    @property
    def split(self) -> str:
        return str(self.config["data"].get("split", "test"))

    def _input_path(self) -> Path:
        data = self.config["data"]
        return (
            self.output_root
            / "inputs"
            / f"{self.split}_{int(data['sequences'])}x{int(data['sequence_length'])}.jsonl"
        )

    def _collection_dir(self) -> Path:
        return self.output_root / "collection" / self.split

    def _metrics_path(self) -> Path:
        return self.output_root / OUTPUT_NAME

    def _qa_output_path(self) -> Path:
        return self.output_root / QA_OUTPUT_NAME

    def _qa_samples_path(self) -> Path:
        return self.output_root / QA_SAMPLES_NAME

    def _emit(self, event: str, **fields: Any) -> None:
        row = {"time": _utc_now(), "event": event, **fields}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "pipeline_events.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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

    def _initialize(self) -> bool:
        persistent_metrics = self.persistent_output_root / OUTPUT_NAME
        if persistent_metrics.is_file():
            existing = json.loads(persistent_metrics.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"existing metrics belong to a different config: {persistent_metrics}"
                )
            return True
        if self.use_tmp_workspace and self.persistent_output_root.exists():
            unexpected = [
                path
                for path in self.persistent_output_root.rglob("*")
                if path.is_file()
            ]
            if unexpected:
                raise RuntimeError(
                    "temporary-workspace destination contains unexpected artifacts: "
                    + ", ".join(str(path) for path in unexpected[:5])
                )
        self.output_root.mkdir(parents=True, exist_ok=True)
        resolved = {
            "config_fingerprint": self.fingerprint,
            "source_config": self.config.get("_config_path"),
            "resolved_config": {
                key: value for key, value in self.config.items() if key != "_config_path"
            },
        }
        (self.output_root / "resolved_config.json").write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return False

    def prepare(self) -> None:
        output = self._input_path()
        manifest = output.with_suffix(".manifest.json")
        if output.is_file() and manifest.is_file():
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                int(stored.get("sequence_count", -1))
                != int(self.config["data"]["sequences"])
                or int(stored.get("sequence_length", -1))
                != int(self.config["data"]["sequence_length"])
            ):
                raise RuntimeError(f"prepared input manifest mismatch: {manifest}")
            self._emit("stage_skip", stage="prepare", reason="complete")
            return
        if output.exists() or manifest.exists():
            raise RuntimeError("partial prepared input exists")

        data = self.config["data"]
        arguments = [
            "-m",
            "learnable_index.prepare_convomem",
            "--dataset-name",
            str(data["dataset_name"]),
            "--tokenizer",
            str(self.config["model"]["name"]),
            "--output",
            str(output),
            "--split",
            self.split,
            "--sequence-length",
            str(data["sequence_length"]),
            "--sequences",
            str(data["sequences"]),
            "--seed",
            str(self.config.get("seed", 13)),
            "--maximum-answer-tokens",
            str(data["maximum_answer_tokens"]),
            "--maximum-future-horizon",
            str(data["maximum_future_horizon"]),
            "--evidence-placement",
            str(data["evidence_placement"]),
            "--evidence-placement-bins",
            str(data["evidence_placement_bins"]),
            "--placement-block-size",
            str(data["placement_block_size"]),
            "--retrieval-local-context-length",
            str(data["retrieval_local_context_length"]),
        ]
        if data.get("sampling_seed") is not None:
            arguments.extend(["--sampling-seed", str(data["sampling_seed"])])
        if data.get("cache_dir"):
            arguments.extend(["--cache-dir", str(data["cache_dir"])])
        if bool(data.get("allow_network", False)):
            arguments.append("--allow-network")
        self._run_command("prepare", arguments)

    def collect(self) -> None:
        output = self._collection_dir()
        manifest_path = output / "collection_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = int(self.config["data"]["sequences"])
            if (
                manifest.get("student_state_protocol") != STUDENT_STATE_PROTOCOL
                or int(manifest.get("sequence_count", -1)) != expected
                or int(manifest.get("sample_count", -1)) != expected
            ):
                raise RuntimeError(
                    f"streaming collection manifest mismatch: {manifest_path}"
                )
            self._emit("stage_skip", stage="collect", reason="complete")
            return
        collection = self.config["collection"]
        arguments = [
            "-m",
            "block_probability_router.streaming_collection",
            "--model-name",
            str(self.config["model"]["name"]),
            "--model-device",
            str(self.config["model"]["device"]),
            "--dtype",
            str(self.config["model"]["dtype"]),
            "--input-jsonl",
            str(self._input_path()),
            "--output-dir",
            str(output),
            "--max-sequences",
            str(self.config["data"]["sequences"]),
            "--max-tokens",
            str(self.config["data"]["sequence_length"]),
            "--local-context-length",
            str(collection["local_context_length"]),
            "--block-size",
            str(collection["block_size"]),
            "--future-horizon",
            str(collection["future_horizon"]),
            "--retrieval-interval",
            str(collection["retrieval_interval"]),
            "--retrieval-point-policy",
            str(collection.get("retrieval_point_policy", "metadata")),
            "--minimum-candidate-blocks",
            str(collection["minimum_candidate_blocks"]),
            "--residual-layer",
            str(collection["residual_layer"]),
            "--query-summary",
            str(collection["query_summary"]),
            "--query-summary-length",
            str(collection["query_summary_length"]),
            "--teacher-layers",
            _index_specification(collection["teacher_layers"]),
            "--teacher-heads",
            _index_specification(collection.get("teacher_heads", "all")),
            "--future-reduction",
            str(collection["future_reduction"]),
            "--progress-every",
            str(collection.get("progress_every", 25)),
            "--no-store-kv-payload",
        ]
        if bool(self.config["model"].get("allow_network", False)):
            arguments.append("--allow-network")
        if collection.get("maximum_candidate_blocks") is not None:
            arguments.extend(
                [
                    "--maximum-candidate-blocks",
                    str(collection["maximum_candidate_blocks"]),
                ]
            )
        if bool(collection.get("length_normalize_blocks", False)):
            arguments.append("--length-normalize-blocks")
        if collection.get("teacher_prefill_chunk_size") is not None:
            arguments.extend(
                [
                    "--teacher-prefill-chunk-size",
                    str(collection["teacher_prefill_chunk_size"]),
                ]
            )
        self._run_command("collect", arguments)

    def _cleanup_prepared_input(self) -> None:
        if bool(self.config["data"].get("persist_prepared_inputs", False)):
            return
        for path in (
            self._input_path(),
            self._input_path().with_suffix(".manifest.json"),
        ):
            if path.is_file():
                path.unlink()
        self._emit("prepared_input_removed", split=self.split)

    def evaluate(self) -> None:
        output = self._metrics_path()
        if output.is_file():
            self._emit("stage_skip", stage="evaluate", reason="complete")
            return
        evaluation = self.config["evaluation"]
        tolerances = _epsilon_values(self.config)
        self._run_command(
            "evaluate",
            [
                "-m",
                "block_probability_router",
                "evaluate",
                "--dataset-dir",
                str(self._collection_dir() / "dataset"),
                "--checkpoint",
                str(_checkpoint_path(self.config)),
                "--output",
                str(output),
                "--device",
                str(evaluation.get("device", "cpu")),
                "--top-n",
                str(evaluation.get("top_n", 4)),
                "--missing-mass-tolerances",
                ",".join(str(value) for value in tolerances),
                "--max-block",
                str(_max_block(self.config)),
            ],
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        result.update(
            {
                "schema_version": 1,
                "run_id": self.config["run_id"],
                "config_fingerprint": self.fingerprint,
                "completed_at": _utc_now(),
                "data": {
                    "source": "convomem",
                    "split": self.split,
                    "sequence_count": int(self.config["data"]["sequences"]),
                    "sequence_length": int(self.config["data"]["sequence_length"]),
                    "evidence_placement": self.config["data"]["evidence_placement"],
                    "sampling_seed": self.config["data"].get("sampling_seed"),
                },
            }
        )
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def qa(self) -> None:
        qa = self.config["qa"]
        if not bool(qa.get("enabled", True)):
            self._emit("stage_skip", stage="qa", reason="disabled")
            return
        metrics_path = self._metrics_path()
        if not metrics_path.is_file():
            raise RuntimeError("router metrics must exist before QA evaluation")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "qa" in metrics:
            self._emit("stage_skip", stage="qa", reason="complete")
            return

        tolerances = _epsilon_values(self.config)
        arguments = [
            "-m",
            "block_probability_router.qa",
            "--collection-dir",
            str(self._collection_dir()),
            "--checkpoint",
            str(_checkpoint_path(self.config)),
            "--output",
            str(self._qa_output_path()),
            "--samples-output",
            str(self._qa_samples_path()),
            "--model-name",
            str(self.config["model"]["name"]),
            "--model-device",
            str(self.config["model"]["device"]),
            "--dtype",
            str(self.config["model"]["dtype"]),
            "--router-device",
            str(qa.get("router_device", "cpu")),
            "--missing-mass-tolerances",
            ",".join(str(value) for value in tolerances),
            "--max-block",
            str(_max_block(self.config)),
            "--maximum-new-tokens",
            str(qa["maximum_new_tokens"]),
            "--prefill-chunk-size",
            str(qa["prefill_chunk_size"]),
            "--progress-every",
            str(qa.get("progress_every", 10)),
            "--bootstrap-iterations",
            str(qa["bootstrap_iterations"]),
            "--seed",
            str(self.config.get("seed", 13)),
        ]
        if qa.get("maximum_samples") is not None:
            arguments.extend(["--maximum-samples", str(qa["maximum_samples"])])
        if self.config["model"].get("cache_dir"):
            arguments.extend(
                ["--model-cache-dir", str(self.config["model"]["cache_dir"])]
            )
        if bool(self.config["model"].get("allow_network", False)):
            arguments.append("--allow-network")
        self._run_command("qa", arguments)

        qa_result = json.loads(self._qa_output_path().read_text(encoding="utf-8"))
        expected_samples = int(
            qa.get("maximum_samples") or self.config["data"]["sequences"]
        )
        if int(qa_result.get("summary", {}).get("sample_count", -1)) != expected_samples:
            raise RuntimeError("QA result sample count does not match the config")
        qa_result["per_sample_artifact_retained"] = False
        metrics["qa"] = qa_result
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _export(self) -> None:
        if not self.use_tmp_workspace:
            return
        source = self._metrics_path()
        if not source.is_file():
            raise RuntimeError("cannot export incomplete evaluation metrics")
        self.persistent_output_root.mkdir(parents=True, exist_ok=True)
        target = self.persistent_output_root / OUTPUT_NAME
        staging = self.persistent_output_root / f".{OUTPUT_NAME}.tmp-{os.getpid()}"
        shutil.copy2(source, staging)
        os.replace(staging, target)
        self._emit("artifact_exported", artifact=OUTPUT_NAME, destination=str(target))

    def _cleanup_tmp_workspace(self) -> None:
        if not self.use_tmp_workspace:
            return
        temporary_root = Path(
            self.config["paths"].get(
                "tmp_workspace_root", DEFAULT_TMP_WORKSPACE_ROOT
            )
        ).resolve()
        output = self.output_root.resolve()
        if temporary_root not in output.parents:
            raise RuntimeError("refusing to clean a workspace outside tmp_workspace_root")
        shutil.rmtree(output)

    def run(self) -> None:
        if self._initialize():
            print(
                json.dumps(
                    {
                        "event": "pipeline_skip",
                        "reason": "complete",
                        "metrics": str(self.persistent_output_root / OUTPUT_NAME),
                    }
                ),
                flush=True,
            )
            return
        checkpoint = _checkpoint_path(self.config)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"router checkpoint is missing: {checkpoint}")
        self._emit(
            "pipeline_start",
            checkpoint=str(checkpoint),
            epsilon=list(_epsilon_values(self.config)),
            max_block=_max_block(self.config),
            qa_enabled=bool(self.config["qa"].get("enabled", True)),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        self.prepare()
        self.collect()
        if (self._collection_dir() / "collection_manifest.json").is_file():
            self._cleanup_prepared_input()
        self.evaluate()
        self.qa()
        self._export()
        self._emit("pipeline_complete", metrics=OUTPUT_NAME)
        self._cleanup_tmp_workspace()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent block-probability checkpoint evaluation pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "block_probability_router_evaluation_convomem4096_hpc.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = load_hpc_config(arguments.config)
    validate_evaluation_hpc_config(config)
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "config": str(arguments.config.resolve()),
                    "output_root": str(_output_path(config)),
                    "checkpoint": str(_checkpoint_path(config)),
                    "epsilon": list(_epsilon_values(config)),
                    "max_block": _max_block(config),
                    "qa": {
                        "enabled": bool(config["qa"].get("enabled", True)),
                        "maximum_samples": config["qa"].get("maximum_samples"),
                        "maximum_new_tokens": int(
                            config["qa"]["maximum_new_tokens"]
                        ),
                        "teacher_forcing": False,
                    },
                },
                indent=2,
            )
        )
        return 0
    EvaluationHPCPipeline(config).run()
    return 0
