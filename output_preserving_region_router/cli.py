from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from learnable_index.collectors import StudentCollectionConfig
from learnable_index.model_adapter import load_frozen_gemma
from learnable_index.planning import PlanConfig, load_sequence_records
from learnable_index.trainer import resolve_device

from .config import (
    OutputPreservationLossConfig,
    RegionRouterConfig,
    RegionTrainConfig,
)
from .trainer import (
    build_examples,
    evaluate_hard_region,
    fit_router,
    load_checkpoint,
)
from .qa import RegionQAEvaluationConfig, evaluate_autoregressive_qa


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--allow-network", action="store_true")


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-context-length", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--future-horizon", type=int, required=True)
    parser.add_argument("--retrieval-interval", type=int, required=True)
    parser.add_argument(
        "--retrieval-point-policy", choices=("interval", "metadata"), required=True
    )
    parser.add_argument("--minimum-candidate-blocks", type=int, required=True)
    parser.add_argument("--residual-layer", type=int, required=True)
    parser.add_argument("--query-summary", choices=("last", "mean"), required=True)
    parser.add_argument("--query-summary-length", type=int, required=True)


def _plan_config(arguments: argparse.Namespace) -> PlanConfig:
    return PlanConfig(
        local_context_length=arguments.local_context_length,
        block_size=arguments.block_size,
        future_horizon_length=arguments.future_horizon,
        retrieval_interval=arguments.retrieval_interval,
        minimum_candidate_blocks=arguments.minimum_candidate_blocks,
        maximum_candidate_blocks=None,
        retrieval_point_policy=arguments.retrieval_point_policy,
    )


def _student_config(arguments: argparse.Namespace) -> StudentCollectionConfig:
    return StudentCollectionConfig(
        local_context_length=arguments.local_context_length,
        residual_layer=arguments.residual_layer,
        query_summary=arguments.query_summary,
        query_summary_length=arguments.query_summary_length,
    )


def _load_bundle(arguments: argparse.Namespace):
    return load_frozen_gemma(
        arguments.model_name,
        device=arguments.model_device,
        dtype=arguments.dtype,
        cache_dir=arguments.model_cache_dir,
        local_files_only=not arguments.allow_network,
    )


def _progress(event: dict[str, Any], every: int) -> None:
    completed = int(event["completed"])
    total = int(event["total"])
    if every and (completed == 1 or completed == total or completed % every == 0):
        print(json.dumps({"event": "progress", **event}, ensure_ascii=False), flush=True)


def _train(arguments: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_bundle(arguments)
    records = load_sequence_records(
        arguments.input_jsonl,
        bundle.tokenizer,
        maximum_sequences=arguments.maximum_sequences,
        maximum_tokens=arguments.maximum_tokens,
    )
    router_config = RegionRouterConfig(
        residual_dim=int(bundle.text_config.hidden_size),
        feature_dim=arguments.feature_dim,
        hidden_dim=arguments.hidden_dim,
        depth=arguments.depth,
        dropout=arguments.dropout,
        minimum_scale=arguments.minimum_scale,
        radius=arguments.radius,
        gate_temperature=arguments.gate_temperature,
        gate_epsilon=arguments.gate_epsilon,
    )
    loss_config = OutputPreservationLossConfig(
        output_temperature=arguments.output_temperature,
        maximum_excess_output_kl=arguments.maximum_excess_output_kl,
        preservation_weight=arguments.preservation_weight,
        sparsity_weight=arguments.sparsity_weight,
        gate_entropy_weight=arguments.gate_entropy_weight,
    )
    train_config = RegionTrainConfig(
        epochs=arguments.epochs,
        early_stopping_patience=arguments.early_stopping_patience,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        gradient_clip_norm=arguments.gradient_clip_norm,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        device=arguments.router_device,
        prefill_chunk_size=arguments.prefill_chunk_size,
        final_gate_temperature=arguments.final_gate_temperature,
        maximum_train_samples=arguments.maximum_train_samples,
        maximum_validation_samples=arguments.maximum_validation_samples,
        progress_every=arguments.progress_every,
    )
    history = fit_router(
        bundle,
        records,
        arguments.output_dir,
        router_config,
        loss_config,
        train_config,
        _student_config(arguments),
        _plan_config(arguments),
        progress_callback=lambda event: _progress(event, arguments.progress_every),
    )
    return {
        "output_dir": str(arguments.output_dir),
        "epochs_completed": len(history),
        "final": history[-1],
    }


def _evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_bundle(arguments)
    (
        router,
        _router_config,
        loss_config,
        train_config,
        student_config,
        plan_config,
        payload,
    ) = load_checkpoint(arguments.checkpoint)
    if payload["model_fingerprint"] != bundle.fingerprint:
        raise ValueError("checkpoint and evaluation model fingerprints differ")
    records = load_sequence_records(
        arguments.input_jsonl,
        bundle.tokenizer,
        maximum_sequences=arguments.maximum_sequences,
        maximum_tokens=arguments.maximum_tokens,
    )
    examples = build_examples(records, plan_config)
    metrics = evaluate_hard_region(
        bundle,
        router,
        examples,
        student_config,
        plan_config,
        loss_config,
        train_config,
        device=resolve_device(arguments.router_device),
        maximum_samples=arguments.maximum_samples,
        progress_callback=lambda event: _progress(event, arguments.progress_every),
    )
    result = {
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "model_kind": payload["model_kind"],
        "training_protocol": payload["training_protocol"],
        "selection": "single_gaussian_region_hard_radius",
        "metrics": metrics,
    }
    if (arguments.qa_output is None) != (arguments.qa_samples_output is None):
        raise ValueError("qa-output and qa-samples-output must be supplied together")
    if arguments.qa_output is not None:
        qa_result = evaluate_autoregressive_qa(
            bundle,
            router,
            examples,
            student_config,
            plan_config,
            payload,
            arguments.qa_output,
            arguments.qa_samples_output,
            RegionQAEvaluationConfig(
                maximum_samples=arguments.qa_maximum_samples,
                maximum_new_tokens=arguments.maximum_new_tokens,
                prefill_chunk_size=arguments.qa_prefill_chunk_size,
                router_device=arguments.router_device,
                progress_every=arguments.progress_every,
            ),
        )
        # Keep the CLI/HPC log compact.  The complete aggregate and per-sample
        # payloads are already persisted in their dedicated artifacts.
        result["qa"] = {
            "aggregate_output": arguments.qa_output.name,
            "sample_count": int(qa_result["summary"]["sample_count"]),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Output-preserving Gaussian region router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_model_arguments(train)
    _add_plan_arguments(train)
    train.add_argument("--input-jsonl", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--maximum-sequences", type=int)
    train.add_argument("--maximum-tokens", type=int)
    train.add_argument("--feature-dim", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--depth", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--minimum-scale", type=float, default=0.05)
    train.add_argument("--radius", type=float, default=1.0)
    train.add_argument("--gate-temperature", type=float, default=0.25)
    train.add_argument("--final-gate-temperature", type=float, default=0.05)
    train.add_argument("--gate-epsilon", type=float, default=1e-6)
    train.add_argument("--output-temperature", type=float, default=1.0)
    train.add_argument("--maximum-excess-output-kl", type=float, default=0.02)
    train.add_argument("--preservation-weight", type=float, default=100.0)
    train.add_argument("--sparsity-weight", type=float, default=1.0)
    train.add_argument("--gate-entropy-weight", type=float, default=0.01)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--early-stopping-patience", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--router-device", default="cuda")
    train.add_argument("--prefill-chunk-size", type=int, default=256)
    train.add_argument("--maximum-train-samples", type=int)
    train.add_argument("--maximum-validation-samples", type=int)
    train.add_argument("--progress-every", type=int, default=1)

    evaluate = subparsers.add_parser("evaluate")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--input-jsonl", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--maximum-sequences", type=int)
    evaluate.add_argument("--maximum-tokens", type=int)
    evaluate.add_argument("--maximum-samples", type=int)
    evaluate.add_argument("--router-device", default="cuda")
    evaluate.add_argument("--progress-every", type=int, default=1)
    evaluate.add_argument("--qa-output", type=Path)
    evaluate.add_argument("--qa-samples-output", type=Path)
    evaluate.add_argument("--qa-maximum-samples", type=int)
    evaluate.add_argument("--maximum-new-tokens", type=int, default=64)
    evaluate.add_argument("--qa-prefill-chunk-size", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = _train(arguments) if arguments.command == "train" else _evaluate(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["main"]
