from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aligned_builder import AlignedCollectionConfig, collect_aligned_dataset
from .collectors import StudentCollectionConfig
from .config import AttentionAggregationConfig, LossConfig, RouterConfig, TrainConfig
from .data import load_dataset, save_dataset
from .model_adapter import load_frozen_gemma
from .planning import PlanConfig, load_sequence_records
from .replay import ReplayConfig, evaluate_retrieval_replay
from .retrieval import RetrievalPolicyConfig
from .synthetic import make_synthetic_samples
from .trainer import evaluate_model, fit_router, load_checkpoint, resolve_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone learnable block-attention index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("make-smoke-data", help="write a synthetic contract dataset")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--samples", type=int, default=128)
    smoke.add_argument("--residual-dim", type=int, default=16)
    smoke.add_argument("--min-blocks", type=int, default=3)
    smoke.add_argument("--max-blocks", type=int, default=8)
    smoke.add_argument("--seed", type=int, default=13)

    train = subparsers.add_parser("train", help="train a router from an aligned dataset")
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--projection-dim", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--depth", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--temperature", type=float, default=0.07)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--early-stopping-patience", type=int)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--top-n", type=int, default=4)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved router checkpoint")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--top-n", type=int)

    collect = subparsers.add_parser(
        "collect",
        help="collect aligned full-context teacher and local-256 student artifacts",
    )
    _add_model_arguments(collect)
    _add_collection_arguments(collect)

    replay = subparsers.add_parser(
        "replay",
        help="run fixed/threshold retrieval and schedule-correct restricted replay",
    )
    _add_model_arguments(replay)
    replay.add_argument("--collection-dir", type=Path, required=True)
    replay.add_argument("--checkpoint", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    _add_replay_arguments(replay)

    run = subparsers.add_parser(
        "run",
        help="collect, train, and replay the complete real-model pipeline",
    )
    _add_model_arguments(run)
    _add_collection_arguments(run, output_required=False)
    run.add_argument("--output-dir", type=Path, required=True)
    _add_training_arguments(run)
    _add_replay_arguments(run)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow from_pretrained to consult remote sources; local-only is the default",
    )


def _add_collection_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_required: bool = True,
) -> None:
    parser.add_argument("--input-jsonl", type=Path, required=True)
    if output_required:
        parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--local-context-length", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--future-horizon", type=int, default=16)
    parser.add_argument("--retrieval-interval", type=int, default=32)
    parser.add_argument("--minimum-candidate-blocks", type=int, default=1)
    parser.add_argument("--maximum-candidate-blocks", type=int)
    parser.add_argument("--residual-layer", type=int, default=-1)
    parser.add_argument("--query-summary", choices=("last", "mean"), default="mean")
    parser.add_argument("--query-summary-length", type=int, default=16)
    parser.add_argument("--teacher-layers", default="all")
    parser.add_argument("--teacher-heads", default="all")
    parser.add_argument("--future-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--length-normalize-blocks", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="emit collection progress every N retrieval samples; 0 disables it",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-device", default="auto")


def _add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy", choices=("fixed", "score_threshold"), default="fixed"
    )
    parser.add_argument("--replay-top-n", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--maximum-replay-samples", type=int)
    parser.add_argument("--router-device", default="cpu")
    parser.add_argument("--no-verify-query-summary", action="store_true")


def _parse_indices(specification: str) -> tuple[int, ...] | None:
    if specification.strip().lower() == "all":
        return None
    selected: set[int] = set()
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid index range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("index selection cannot be empty")
    return tuple(sorted(selected))


def _load_bundle(arguments: argparse.Namespace):
    return load_frozen_gemma(
        arguments.model_name,
        device=arguments.model_device,
        dtype=arguments.dtype,
        local_files_only=not arguments.allow_network,
    )


def _collection_config(arguments: argparse.Namespace) -> AlignedCollectionConfig:
    return AlignedCollectionConfig(
        plan=PlanConfig(
            local_context_length=arguments.local_context_length,
            block_size=arguments.block_size,
            future_horizon_length=arguments.future_horizon,
            retrieval_interval=arguments.retrieval_interval,
            minimum_candidate_blocks=arguments.minimum_candidate_blocks,
            maximum_candidate_blocks=arguments.maximum_candidate_blocks,
        ),
        student=StudentCollectionConfig(
            local_context_length=arguments.local_context_length,
            residual_layer=arguments.residual_layer,
            query_summary=arguments.query_summary,
            query_summary_length=arguments.query_summary_length,
        ),
        attention=AttentionAggregationConfig(
            teacher_layers=_parse_indices(arguments.teacher_layers),
            teacher_heads=_parse_indices(arguments.teacher_heads),
            future_reduction=arguments.future_reduction,
            length_normalize_blocks=arguments.length_normalize_blocks,
        ),
    )


def _collection_progress(arguments: argparse.Namespace):
    every = int(arguments.progress_every)
    if every < 0:
        raise ValueError("progress_every must be non-negative")
    if every == 0:
        return None

    def report(event: dict) -> None:
        completed = int(event["completed"])
        total = int(event["total"])
        if completed == 1 or completed == total or completed % every == 0:
            print(
                json.dumps(
                    {"event": "collection_progress", **event},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return report


def _replay_config(arguments: argparse.Namespace) -> ReplayConfig:
    return ReplayConfig(
        policy=RetrievalPolicyConfig(
            policy=arguments.policy,
            top_n=arguments.replay_top_n,
            score_threshold=arguments.score_threshold,
        ),
        maximum_samples=arguments.maximum_replay_samples,
        router_device=arguments.router_device,
        verify_query_summary=not arguments.no_verify_query_summary,
    )


def _make_smoke_data(arguments: argparse.Namespace) -> dict:
    samples = make_synthetic_samples(
        sample_count=arguments.samples,
        residual_dim=arguments.residual_dim,
        min_blocks=arguments.min_blocks,
        max_blocks=arguments.max_blocks,
        seed=arguments.seed,
    )
    save_dataset(
        arguments.output_dir,
        samples,
        metadata={
            "kind": "synthetic_contract_smoke",
            "teacher_attention_used_as_input": False,
        },
    )
    return {"output_dir": str(arguments.output_dir), "sample_count": len(samples)}


def _train(arguments: argparse.Namespace) -> dict:
    dataset, manifest = load_dataset(arguments.dataset_dir)
    router_config = RouterConfig(
        residual_dim=dataset.residual_dim,
        projection_dim=arguments.projection_dim,
        hidden_dim=arguments.hidden_dim,
        depth=arguments.depth,
        dropout=arguments.dropout,
        initial_temperature=arguments.temperature,
    )
    loss_config = LossConfig()
    train_config = TrainConfig(
        epochs=arguments.epochs,
        early_stopping_patience=arguments.early_stopping_patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        top_n=arguments.top_n,
        device=arguments.device,
    )
    history = fit_router(dataset, arguments.output_dir, router_config, loss_config, train_config)
    return {
        "output_dir": str(arguments.output_dir),
        "dataset_samples": manifest["sample_count"],
        "epochs_completed": len(history),
        "final": history[-1],
    }


def _evaluate(arguments: argparse.Namespace) -> dict:
    dataset, _ = load_dataset(arguments.dataset_dir)
    model, _, loss_config, stored_train_config, payload = load_checkpoint(arguments.checkpoint)
    overrides = dict(stored_train_config.__dict__)
    overrides["device"] = arguments.device
    if arguments.top_n is not None:
        overrides["top_n"] = arguments.top_n
    train_config = TrainConfig(**overrides)
    metrics = evaluate_model(
        model,
        dataset,
        loss_config,
        train_config,
        device=resolve_device(arguments.device),
    )
    result = {
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "sample_count": len(dataset),
        "metrics": metrics,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return result


def _collect(arguments: argparse.Namespace) -> dict:
    bundle = _load_bundle(arguments)
    records = load_sequence_records(
        arguments.input_jsonl,
        bundle.tokenizer,
        maximum_sequences=arguments.max_sequences,
        maximum_tokens=arguments.max_tokens,
    )
    _, manifest = collect_aligned_dataset(
        bundle,
        records,
        arguments.output_dir,
        _collection_config(arguments),
        progress_callback=_collection_progress(arguments),
    )
    return manifest


def _replay(arguments: argparse.Namespace) -> dict:
    bundle = _load_bundle(arguments)
    return evaluate_retrieval_replay(
        bundle,
        arguments.collection_dir,
        arguments.checkpoint,
        arguments.output_dir,
        _replay_config(arguments),
    )


def _run_real(arguments: argparse.Namespace) -> dict:
    bundle = _load_bundle(arguments)
    records = load_sequence_records(
        arguments.input_jsonl,
        bundle.tokenizer,
        maximum_sequences=arguments.max_sequences,
        maximum_tokens=arguments.max_tokens,
    )
    root = arguments.output_dir
    collection_dir = root / "collection"
    training_dir = root / "training"
    replay_dir = root / "replay"
    dataset, collection_manifest = collect_aligned_dataset(
        bundle,
        records,
        collection_dir,
        _collection_config(arguments),
        progress_callback=_collection_progress(arguments),
    )
    loss_config = LossConfig()
    router_config = RouterConfig(
        residual_dim=dataset.residual_dim,
        projection_dim=arguments.projection_dim,
        hidden_dim=arguments.hidden_dim,
        depth=arguments.depth,
        dropout=arguments.dropout,
        initial_temperature=arguments.temperature,
    )
    train_config = TrainConfig(
        epochs=arguments.epochs,
        early_stopping_patience=arguments.early_stopping_patience,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        top_n=arguments.top_n,
        device=arguments.train_device,
    )
    history = fit_router(dataset, training_dir, router_config, loss_config, train_config)
    replay_summary = evaluate_retrieval_replay(
        bundle,
        collection_dir,
        training_dir / "best.pt",
        replay_dir,
        _replay_config(arguments),
    )
    result = {
        "schema_version": 1,
        "collection": collection_manifest,
        "training": {"epochs_completed": len(history), "final": history[-1]},
        "replay": replay_summary,
        "paths": {
            "collection": str(collection_dir),
            "training": str(training_dir),
            "replay": str(replay_dir),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "make-smoke-data":
        result = _make_smoke_data(arguments)
    elif arguments.command == "train":
        result = _train(arguments)
    elif arguments.command == "evaluate":
        result = _evaluate(arguments)
    elif arguments.command == "collect":
        result = _collect(arguments)
    elif arguments.command == "replay":
        result = _replay(arguments)
    elif arguments.command == "run":
        result = _run_real(arguments)
    else:  # pragma: no cover - argparse enforces valid subcommands
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
