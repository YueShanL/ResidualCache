from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence

import torch

from learnable_index.collectors import StudentCollectionConfig
from learnable_index.planning import (
    PlanConfig,
    RetrievalPlan,
    SequenceRecord,
    build_retrieval_plans,
)
from learnable_index.trainer import resolve_device

from .config import (
    OutputPreservationLossConfig,
    RegionRouterConfig,
    RegionTrainConfig,
)
from .losses import output_preservation_loss
from .model import GaussianRegionRouter
from .runtime import (
    collect_training_student_state,
    full_context_future_logits,
    hard_region_future_logits,
    physical_full_attention_layers,
    soft_gated_future_logits,
)


CHECKPOINT_FORMAT_VERSION = 1
MODEL_KIND = "output_preserving_gaussian_region_router"
TRAINING_PROTOCOL = "full_logits_soft_block_gate_streaming_kv_v1"


Example = tuple[SequenceRecord, RetrievalPlan]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_examples(
    records: Iterable[SequenceRecord], plan_config: PlanConfig
) -> list[Example]:
    examples = [
        (record, plan)
        for record in records
        for plan in build_retrieval_plans(record, plan_config)
    ]
    if not examples:
        raise ValueError("input records produced no output-preservation examples")
    return examples


def split_examples(
    examples: Sequence[Example], validation_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0:
        return list(examples), []
    group_ids = sorted({record.sequence_id for record, _plan in examples})
    if len(group_ids) < 2:
        return list(examples), []
    random.Random(seed).shuffle(group_ids)
    count = max(1, round(len(group_ids) * validation_fraction))
    count = min(count, len(group_ids) - 1)
    validation_ids = set(group_ids[:count])
    return (
        [row for row in examples if row[0].sequence_id not in validation_ids],
        [row for row in examples if row[0].sequence_id in validation_ids],
    )


class _MetricMeans:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, value: float | torch.Tensor) -> None:
        number = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(value)
        self.sums[name] = self.sums.get(name, 0.0) + number
        self.counts[name] = self.counts.get(name, 0) + 1

    def compute(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in sorted(self.sums)
        }


def _checkpoint_selection_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Prefer feasible regions, then the smallest feasible expected set.

    A scalar Lagrangian is useful for gradients but is not a rigorous model
    selection rule: one fewer block could otherwise compensate for violating
    the preservation constraint.  The checkpoint decision is therefore
    lexicographic.
    """

    violation = max(float(metrics["output_kl_violation"]), 0.0)
    expected_blocks = float(metrics["expected_selected_blocks"])
    output_kl = float(metrics["output_kl"])
    if violation <= 1e-12:
        return (0.0, expected_blocks, output_kl)
    return (1.0, violation, expected_blocks)


def _gate_temperature(
    router_config: RegionRouterConfig,
    train_config: RegionTrainConfig,
    epoch: int,
) -> float:
    final = train_config.final_gate_temperature
    if final is None or train_config.epochs == 1:
        return router_config.gate_temperature
    progress = (epoch - 1) / (train_config.epochs - 1)
    return router_config.gate_temperature * (
        final / router_config.gate_temperature
    ) ** progress


def _router_output(
    router: GaussianRegionRouter,
    state,
    device: torch.device,
    gate_temperature: float,
):
    dtype = next(router.parameters()).dtype
    query = state.query_summary.unsqueeze(0).to(device=device, dtype=dtype)
    blocks = state.block_summaries.unsqueeze(0).to(device=device, dtype=dtype)
    mask = torch.ones((1, blocks.shape[1]), dtype=torch.bool, device=device)
    return router(query, blocks, mask, gate_temperature=gate_temperature), mask


def _run_epoch(
    bundle,
    router: GaussianRegionRouter,
    examples: Sequence[Example],
    student_config: StudentCollectionConfig,
    plan_config: PlanConfig,
    loss_config: OutputPreservationLossConfig,
    train_config: RegionTrainConfig,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    router.train(training)
    bundle.model.eval()
    device = next(router.parameters()).device
    layers = physical_full_attention_layers(bundle)
    temperature = _gate_temperature(router.config, train_config, epoch)
    metrics = _MetricMeans()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, (record, plan) in enumerate(examples, start=1):
        full_logits = full_context_future_logits(
            bundle,
            record,
            plan,
            prefill_chunk_size=train_config.prefill_chunk_size,
        )
        state = collect_training_student_state(
            bundle,
            record,
            plan,
            student_config,
            block_size=plan_config.block_size,
            capture_layers=layers,
        )
        all_indices = tuple(range(len(plan.candidate_blocks)))
        all_history_logits = hard_region_future_logits(
            bundle,
            record,
            plan,
            state,
            all_indices,
            full_attention_layers=layers,
            local_context_length=student_config.local_context_length,
            block_size=plan_config.block_size,
        )
        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            output, candidate_mask = _router_output(
                router, state, device, temperature
            )
            gated_logits = soft_gated_future_logits(
                bundle,
                record,
                plan,
                state,
                output.gates,
                full_attention_layers=layers,
                local_context_length=student_config.local_context_length,
                block_size=plan_config.block_size,
                gate_epsilon=router.config.gate_epsilon,
            )
            loss = output_preservation_loss(
                full_logits.to(gated_logits.device),
                gated_logits,
                all_history_logits.to(gated_logits.device),
                output.gates,
                candidate_mask,
                loss_config,
            )
            if training:
                (loss.total / train_config.gradient_accumulation_steps).backward()
                should_step = (
                    index % train_config.gradient_accumulation_steps == 0
                    or index == len(examples)
                )
                if should_step:
                    if train_config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            router.parameters(), train_config.gradient_clip_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        hard_count = output.hard_mask.sum(dim=-1).float().mean()
        candidate_count = candidate_mask.sum(dim=-1).float().mean()
        top1_agreement = (
            full_logits.to(gated_logits.device).argmax(dim=-1)
            == gated_logits.detach().argmax(dim=-1)
        ).float().mean()
        metrics.add("loss", loss.total)
        metrics.add("output_kl", loss.output_kl)
        metrics.add("all_history_output_kl", loss.all_history_output_kl)
        metrics.add("excess_output_kl", loss.excess_output_kl)
        metrics.add("output_kl_violation", loss.output_kl_violation)
        metrics.add(
            "excess_output_kl_constraint_success",
            float(loss.excess_output_kl.detach().cpu())
            <= loss_config.maximum_excess_output_kl,
        )
        metrics.add("expected_selected_blocks", loss.expected_selected_blocks)
        metrics.add("expected_selected_fraction", loss.expected_selected_fraction)
        metrics.add("hard_selected_blocks", hard_count)
        metrics.add("hard_selected_fraction", hard_count / candidate_count)
        metrics.add("candidate_blocks", candidate_count)
        metrics.add("gate_entropy", loss.gate_entropy)
        metrics.add("gate_mean", output.gates[candidate_mask].mean())
        metrics.add("query_scale_mean", output.query_scale.mean())
        metrics.add("future_top1_agreement", top1_agreement)
        metrics.add("gate_temperature", temperature)
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch,
                    "training": training,
                    "completed": index,
                    "total": len(examples),
                    "sample_id": plan.sample_id,
                    "output_kl": float(loss.output_kl.detach().cpu()),
                    "expected_selected_blocks": float(
                        loss.expected_selected_blocks.detach().cpu()
                    ),
                }
            )
        del full_logits, all_history_logits, state, output, gated_logits, loss
    return metrics.compute()


def _save_checkpoint(
    path: Path,
    router: GaussianRegionRouter,
    optimizer: torch.optim.Optimizer,
    router_config: RegionRouterConfig,
    loss_config: OutputPreservationLossConfig,
    train_config: RegionTrainConfig,
    student_config: StudentCollectionConfig,
    plan_config: PlanConfig,
    model_fingerprint: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_kind": MODEL_KIND,
            "training_protocol": TRAINING_PROTOCOL,
            "model_state_dict": router.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "router_config": asdict(router_config),
            "loss_config": asdict(loss_config),
            "train_config": asdict(train_config),
            "student_config": asdict(student_config),
            "plan_config": asdict(plan_config),
            "model_fingerprint": model_fingerprint,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
):
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if int(payload.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported region-router checkpoint format")
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError("checkpoint is not an output-preserving region router")
    if payload.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError("checkpoint training protocol is incompatible")
    router_config = RegionRouterConfig(**payload["router_config"])
    loss_config = OutputPreservationLossConfig(**payload["loss_config"])
    train_config = RegionTrainConfig(**payload["train_config"])
    student_config = StudentCollectionConfig(**payload["student_config"])
    plan_config = PlanConfig(**payload["plan_config"])
    router = GaussianRegionRouter(router_config)
    router.load_state_dict(payload["model_state_dict"])
    return (
        router,
        router_config,
        loss_config,
        train_config,
        student_config,
        plan_config,
        payload,
    )


def fit_router(
    bundle,
    records: Sequence[SequenceRecord],
    output_dir: Path | str,
    router_config: RegionRouterConfig,
    loss_config: OutputPreservationLossConfig,
    train_config: RegionTrainConfig,
    student_config: StudentCollectionConfig,
    plan_config: PlanConfig,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if router_config.residual_dim != int(bundle.text_config.hidden_size):
        raise ValueError("router residual dimension must match the frozen model")
    if any(parameter.requires_grad for parameter in bundle.model.parameters()):
        raise ValueError("the language model must be fully frozen before router training")
    if student_config.local_context_length != plan_config.local_context_length:
        raise ValueError("student and plan local context lengths must match")
    if student_config.local_context_length != int(bundle.text_config.sliding_window):
        raise ValueError("student local context must equal Gemma's native sliding window")
    _seed_everything(train_config.seed)
    examples = build_examples(records, plan_config)
    train_examples, validation_examples = split_examples(
        examples, train_config.validation_fraction, train_config.seed
    )
    random.Random(train_config.seed).shuffle(train_examples)
    if train_config.maximum_train_samples is not None:
        train_examples = train_examples[: train_config.maximum_train_samples]
    if train_config.maximum_validation_samples is not None:
        validation_examples = validation_examples[
            : train_config.maximum_validation_samples
        ]
    if not train_examples:
        raise ValueError("training split contains no examples")

    device = resolve_device(train_config.device)
    router = GaussianRegionRouter(router_config).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_kind": MODEL_KIND,
        "training_protocol": TRAINING_PROTOCOL,
        "router": asdict(router_config),
        "loss": asdict(loss_config),
        "train": asdict(train_config),
        "student": asdict(student_config),
        "plan": asdict(plan_config),
        "model_fingerprint": bundle.fingerprint,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "objective": (
            "minimum expected selected block count subject to full-context "
            "future-logit KL excess over the all-history streaming replay "
            "floor remaining within tolerance"
        ),
        "teacher_attention_used": False,
        "frozen_model_parameters_trainable": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_score = (float("inf"), float("inf"), float("inf"))
    best_monitored_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, train_config.epochs + 1):
        random.Random(train_config.seed + epoch).shuffle(train_examples)
        train_metrics = _run_epoch(
            bundle,
            router,
            train_examples,
            student_config,
            plan_config,
            loss_config,
            train_config,
            epoch=epoch,
            optimizer=optimizer,
            progress_callback=progress_callback,
        )
        validation_metrics = (
            _run_epoch(
                bundle,
                router,
                validation_examples,
                student_config,
                plan_config,
                loss_config,
                train_config,
                epoch=epoch,
                optimizer=None,
                progress_callback=progress_callback,
            )
            if validation_examples
            else {}
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        with (output_dir / "metrics.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        monitored_metrics = validation_metrics or train_metrics
        selection_score = _checkpoint_selection_score(monitored_metrics)
        if selection_score < best_score:
            best_score = selection_score
            best_monitored_loss = float(monitored_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                output_dir / "best.pt",
                router,
                optimizer,
                router_config,
                loss_config,
                train_config,
                student_config,
                plan_config,
                bundle.fingerprint,
                epoch,
                row,
            )
        else:
            stale_epochs += 1
        if (
            train_config.early_stopping_patience is not None
            and stale_epochs >= train_config.early_stopping_patience
        ):
            break
    _save_checkpoint(
        output_dir / "final.pt",
        router,
        optimizer,
        router_config,
        loss_config,
        train_config,
        student_config,
        plan_config,
        bundle.fingerprint,
        history[-1]["epoch"],
        history[-1],
    )
    summary = {
        "model_kind": MODEL_KIND,
        "training_protocol": TRAINING_PROTOCOL,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_monitored_loss": best_monitored_loss,
        "best_selection_score": {
            "constraint_infeasible": bool(best_score[0]),
            "primary": best_score[1],
            "secondary": best_score[2],
        },
        "checkpoint_selection": (
            "feasible excess-KL first; then minimum expected selected blocks"
        ),
        "best_checkpoint": "best.pt",
        "final_checkpoint": "final.pt",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return history


@torch.inference_mode()
def evaluate_hard_region(
    bundle,
    router: GaussianRegionRouter,
    examples: Sequence[Example],
    student_config: StudentCollectionConfig,
    plan_config: PlanConfig,
    loss_config: OutputPreservationLossConfig,
    train_config: RegionTrainConfig,
    *,
    device: torch.device,
    maximum_samples: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    router.to(device).eval()
    layers = physical_full_attention_layers(bundle)
    selected_examples = list(examples)
    if maximum_samples is not None:
        selected_examples = selected_examples[:maximum_samples]
    if not selected_examples:
        raise ValueError("evaluation contains no examples")
    metrics = _MetricMeans()
    for index, (record, plan) in enumerate(selected_examples, start=1):
        full_logits = full_context_future_logits(
            bundle,
            record,
            plan,
            prefill_chunk_size=train_config.prefill_chunk_size,
        )
        state = collect_training_student_state(
            bundle,
            record,
            plan,
            student_config,
            block_size=plan_config.block_size,
            capture_layers=layers,
        )
        output, candidate_mask = _router_output(
            router,
            state,
            device,
            train_config.final_gate_temperature
            or router.config.gate_temperature,
        )
        indices = tuple(
            item
            for item, selected in enumerate(output.hard_mask[0].tolist())
            if selected
        )
        hard_logits = hard_region_future_logits(
            bundle,
            record,
            plan,
            state,
            indices,
            full_attention_layers=layers,
            local_context_length=student_config.local_context_length,
            block_size=plan_config.block_size,
        )
        all_indices = tuple(range(len(plan.candidate_blocks)))
        all_history_logits = (
            hard_logits
            if indices == all_indices
            else hard_region_future_logits(
                bundle,
                record,
                plan,
                state,
                all_indices,
                full_attention_layers=layers,
                local_context_length=student_config.local_context_length,
                block_size=plan_config.block_size,
            )
        )
        hard_gates = output.hard_mask.to(output.gates.dtype)
        loss = output_preservation_loss(
            full_logits.to(hard_logits.device),
            hard_logits,
            all_history_logits.to(hard_logits.device),
            hard_gates,
            candidate_mask,
            loss_config,
        )
        candidate_count = len(plan.candidate_blocks)
        metrics.add("output_kl", loss.output_kl)
        metrics.add("all_history_output_kl", loss.all_history_output_kl)
        metrics.add("excess_output_kl", loss.excess_output_kl)
        metrics.add(
            "excess_output_kl_constraint_success",
            float(loss.excess_output_kl.cpu())
            <= loss_config.maximum_excess_output_kl,
        )
        metrics.add("selected_blocks", len(indices))
        metrics.add("candidate_blocks", candidate_count)
        metrics.add("selected_fraction", len(indices) / candidate_count)
        metrics.add(
            "future_top1_agreement",
            (
                full_logits.to(hard_logits.device).argmax(dim=-1)
                == hard_logits.argmax(dim=-1)
            ).float().mean(),
        )
        metrics.add("query_scale_mean", output.query_scale.mean())
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": index,
                    "total": len(selected_examples),
                    "sample_id": plan.sample_id,
                    "output_kl": float(loss.output_kl.cpu()),
                    "selected_blocks": len(indices),
                }
            )
    result = metrics.compute()
    result["sample_count"] = len(selected_examples)
    return result


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MODEL_KIND",
    "TRAINING_PROTOCOL",
    "build_examples",
    "evaluate_hard_region",
    "fit_router",
    "load_checkpoint",
    "split_examples",
]
