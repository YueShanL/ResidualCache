from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from learnable_index.data import RetrievalDataset, collate_retrieval_samples, split_dataset
from learnable_index.trainer import resolve_device

from .config import ProbabilityLossConfig, ProbabilityRouterConfig, ProbabilityTrainConfig
from .losses import probability_router_loss
from .metrics import MetricAccumulator, finite_metrics, update_probability_metrics
from .model import BlockProbabilityRouter


CHECKPOINT_FORMAT_VERSION = 1
MODEL_KIND = "positive_block_probability_router"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: RetrievalDataset,
    config: ProbabilityTrainConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_retrieval_samples,
        generator=torch.Generator().manual_seed(config.seed),
    )


def _run_epoch(
    model: BlockProbabilityRouter,
    loader: DataLoader,
    loss_config: ProbabilityLossConfig,
    train_config: ProbabilityTrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator()
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch.query_summaries, batch.block_summaries, batch.candidate_mask)
            loss = probability_router_loss(output, batch, loss_config)
            if training:
                loss.total.backward()
                if train_config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
                optimizer.step()
        batch_size = batch.query_summaries.shape[0]
        metrics.add("loss", loss.total.detach().repeat(batch_size).cpu())
        metrics.add("conditional_loss", loss.conditional.detach().repeat(batch_size).cpu())
        update_probability_metrics(
            metrics,
            output,
            batch,
            top_n=train_config.top_n,
            probability_thresholds=train_config.probability_thresholds,
        )
    return finite_metrics(metrics.compute())


@torch.no_grad()
def evaluate_model(
    model: BlockProbabilityRouter,
    dataset: RetrievalDataset,
    loss_config: ProbabilityLossConfig,
    train_config: ProbabilityTrainConfig,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or resolve_device(train_config.device)
    model.to(device)
    return _run_epoch(
        model,
        _loader(dataset, train_config, shuffle=False),
        loss_config,
        train_config,
        device,
        optimizer=None,
    )


def _save_checkpoint(
    path: Path,
    model: BlockProbabilityRouter,
    optimizer: torch.optim.Optimizer,
    router_config: ProbabilityRouterConfig,
    loss_config: ProbabilityLossConfig,
    train_config: ProbabilityTrainConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_kind": MODEL_KIND,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "router_config": asdict(router_config),
            "loss_config": asdict(loss_config),
            "train_config": asdict(train_config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[
    BlockProbabilityRouter,
    ProbabilityRouterConfig,
    ProbabilityLossConfig,
    ProbabilityTrainConfig,
    dict[str, Any],
]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if int(payload.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported probability-router checkpoint format")
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError("checkpoint is not a positive block-probability router")
    router_config = ProbabilityRouterConfig(**payload["router_config"])
    loss_config = ProbabilityLossConfig(**payload["loss_config"])
    train_config = ProbabilityTrainConfig(
        **{
            **payload["train_config"],
            "probability_thresholds": tuple(payload["train_config"]["probability_thresholds"]),
        }
    )
    model = BlockProbabilityRouter(router_config)
    model.load_state_dict(payload["model_state_dict"])
    return model, router_config, loss_config, train_config, payload


def fit_router(
    dataset: RetrievalDataset,
    output_dir: Path | str,
    router_config: ProbabilityRouterConfig,
    loss_config: ProbabilityLossConfig,
    train_config: ProbabilityTrainConfig,
) -> list[dict[str, Any]]:
    if dataset.residual_dim != router_config.residual_dim:
        raise ValueError("dataset residual dimension does not match router config")
    _seed_everything(train_config.seed)
    device = resolve_device(train_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, validation_dataset = split_dataset(
        dataset, train_config.validation_fraction, train_config.seed
    )
    model = BlockProbabilityRouter(router_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    run_config = {
        "model_kind": MODEL_KIND,
        "router": asdict(router_config),
        "loss": asdict(loss_config),
        "train": asdict(train_config),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset else 0,
        "split_policy": "grouped_by_split_group_id_or_sequence_id",
        "teacher_target": "next_block_attention_conditioned_on_historical_memory",
        "normalization": "q_dot_sum_historical_key_features",
        "information_boundary": {
            "query_input": "current_restricted_history",
            "key_input": "completed_historical_memory_blocks_only",
            "current_live_block_is_candidate": False,
            "teacher_use": "labels_only",
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    history: list[dict[str, Any]] = []
    best_value = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(1, train_config.epochs + 1):
            train_metrics = _run_epoch(
                model,
                _loader(train_dataset, train_config, shuffle=True),
                loss_config,
                train_config,
                device,
                optimizer,
            )
            validation_metrics = (
                evaluate_model(
                    model,
                    validation_dataset,
                    loss_config,
                    train_config,
                    device=device,
                )
                if validation_dataset is not None
                else {}
            )
            row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            history.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            monitored = validation_metrics.get("loss", train_metrics["loss"])
            if monitored < best_value:
                best_value = monitored
                best_epoch = epoch
                epochs_without_improvement = 0
                _save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    router_config,
                    loss_config,
                    train_config,
                    epoch,
                    row,
                )
            else:
                epochs_without_improvement += 1
            if (
                train_config.early_stopping_patience is not None
                and epochs_without_improvement >= train_config.early_stopping_patience
            ):
                stopped_early = True
                break

    final_epoch = int(history[-1]["epoch"])
    _save_checkpoint(
        output_dir / "final.pt",
        model,
        optimizer,
        router_config,
        loss_config,
        train_config,
        final_epoch,
        history[-1],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "epochs_completed": len(history),
                "maximum_epochs": train_config.epochs,
                "early_stopping_patience": train_config.early_stopping_patience,
                "stopped_early": stopped_early,
                "best_epoch": best_epoch,
                "best_monitored_loss": best_value,
                "final": history[-1],
                "best_checkpoint": "best.pt",
                "final_checkpoint": "final.pt",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    return history
