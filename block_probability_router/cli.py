from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from learnable_index.data import load_dataset
from learnable_index.trainer import resolve_device

from .config import ProbabilityLossConfig, ProbabilityRouterConfig, ProbabilityTrainConfig
from .streaming_collection import STUDENT_STATE_PROTOCOL
from .trainer import evaluate_model, fit_router, load_checkpoint


def _parse_missing_mass_tolerances(specification: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value.strip()) for value in specification.split(",") if value.strip()}))
    if not values:
        raise ValueError("missing-mass tolerance list cannot be empty")
    if any(not 0 < value < 1 for value in values):
        raise ValueError("every missing-mass tolerance must be in (0, 1)")
    return values


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--positive-floor", type=float, default=1e-6)
    parser.add_argument("--normalization-epsilon", type=float, default=1e-12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument(
        "--missing-mass-tolerances",
        default="0.01,0.02,0.05,0.1",
        help="global probability mass allowed to be omitted; 0.02 retains at least 0.98",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Positive block-probability router over learnable_index datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the positive probability router")
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    _add_training_arguments(train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one probability-router checkpoint")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--top-n", type=int)
    evaluate.add_argument("--missing-mass-tolerances")
    evaluate.add_argument(
        "--max-block",
        type=int,
        default=-1,
        help="hard retrieval-block limit applied after cumulative-mass selection; -1 disables it",
    )
    return parser


def _train(arguments: argparse.Namespace) -> dict:
    dataset, manifest = load_dataset(arguments.dataset_dir)
    dataset_protocol = manifest.get("metadata", {}).get("student_state_protocol")
    if dataset_protocol != STUDENT_STATE_PROTOCOL:
        raise ValueError(
            "probability-router training requires the block-aligned streaming dataset protocol"
        )
    router_config = ProbabilityRouterConfig(
        residual_dim=dataset.residual_dim,
        feature_dim=arguments.feature_dim,
        hidden_dim=arguments.hidden_dim,
        depth=arguments.depth,
        dropout=arguments.dropout,
        positive_floor=arguments.positive_floor,
        normalization_epsilon=arguments.normalization_epsilon,
    )
    train_config = ProbabilityTrainConfig(
        epochs=arguments.epochs,
        early_stopping_patience=arguments.early_stopping_patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        validation_fraction=arguments.validation_fraction,
        top_n=arguments.top_n,
        missing_mass_tolerances=_parse_missing_mass_tolerances(
            arguments.missing_mass_tolerances
        ),
        seed=arguments.seed,
        device=arguments.device,
    )
    history = fit_router(
        dataset,
        arguments.output_dir,
        router_config,
        ProbabilityLossConfig(),
        train_config,
        student_state_protocol=dataset_protocol,
    )
    return {
        "output_dir": str(arguments.output_dir),
        "dataset_samples": manifest["sample_count"],
        "epochs_completed": len(history),
        "final": history[-1],
    }


def _evaluate(arguments: argparse.Namespace) -> dict:
    if arguments.max_block != -1 and arguments.max_block <= 0:
        raise ValueError("max_block must be -1 or a positive integer")
    dataset, manifest = load_dataset(arguments.dataset_dir)
    dataset_protocol = manifest.get("metadata", {}).get("student_state_protocol")
    if dataset_protocol != STUDENT_STATE_PROTOCOL:
        raise ValueError(
            "probability-router evaluation requires the block-aligned streaming dataset protocol"
        )
    model, _, loss_config, stored_config, payload = load_checkpoint(arguments.checkpoint)
    checkpoint_protocol = payload.get("student_state_protocol")
    train_config = replace(
        stored_config,
        device=arguments.device,
        top_n=arguments.top_n if arguments.top_n is not None else stored_config.top_n,
        missing_mass_tolerances=(
            _parse_missing_mass_tolerances(arguments.missing_mass_tolerances)
            if arguments.missing_mass_tolerances is not None
            else stored_config.missing_mass_tolerances
        ),
    )
    metrics = evaluate_model(
        model,
        dataset,
        loss_config,
        train_config,
        device=resolve_device(arguments.device),
        maximum_retrieval_blocks=arguments.max_block,
    )
    result = {
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "student_state_protocol": {
            "dataset": dataset_protocol,
            "checkpoint": checkpoint_protocol,
            "matched": checkpoint_protocol == dataset_protocol,
        },
        "sample_count": len(dataset),
        "retrieval_policy": {
            "kind": "minimum_cumulative_probability_mass",
            "missing_mass_tolerances": list(train_config.missing_mass_tolerances),
            "max_block": arguments.max_block,
        },
        "metrics": metrics,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = _train(arguments) if arguments.command == "train" else _evaluate(arguments)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
