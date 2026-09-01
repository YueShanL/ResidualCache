from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from block_probability_router.qa import (
    _evidence_block_metrics,
    _evidence_only_prompt,
    _generate_full_context,
    _generate_sparse_replay,
    _generation_metrics,
)
from learnable_index.planning import RetrievalPlan, SequenceRecord
from learnable_index.trainer import resolve_device

from .model import GaussianRegionRouter
from .runtime import (
    collect_training_student_state,
    pack_block_layer_kv,
    physical_full_attention_layers,
)


QA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RegionQAEvaluationConfig:
    maximum_samples: int | None = None
    maximum_new_tokens: int = 64
    prefill_chunk_size: int = 256
    router_device: str = "cuda"
    progress_every: int = 1

    def __post_init__(self) -> None:
        if self.maximum_samples is not None and self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive when set")
        if self.maximum_new_tokens <= 0 or self.prefill_chunk_size <= 0:
            raise ValueError("generation lengths must be positive")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    names = sorted({name for row in rows for name in row["conditions"]})
    for name in names:
        values = [row["conditions"][name] for row in rows]
        numeric = sorted(
            {
                key
                for value in values
                for key, item in value.items()
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            }
            - {"predicted_token_ids"}
        )
        conditions[name] = {
            "sample_count": len(values),
            **{
                key: _mean([float(value[key]) for value in values])
                for key in numeric
            },
        }
    comparisons: dict[str, float] = {}
    for baseline in ("full_context", "all_history_upper_bound", "local_only"):
        for metric in ("answer_exact_match", "answer_token_f1", "answer_contains"):
            comparisons[f"region_minus_{baseline}/{metric}"] = _mean(
                [
                    float(row["conditions"]["region_router"][metric])
                    - float(row["conditions"][baseline][metric])
                    for row in rows
                ]
            )
    return {
        "sample_count": len(rows),
        "conditions": conditions,
        "paired_comparisons": comparisons,
    }


@torch.inference_mode()
def evaluate_autoregressive_qa(
    bundle,
    router: GaussianRegionRouter,
    examples: Sequence[tuple[SequenceRecord, RetrievalPlan]],
    student_config,
    plan_config,
    checkpoint: Mapping[str, Any],
    output_path: Path | str,
    samples_output_path: Path | str,
    config: RegionQAEvaluationConfig,
) -> dict[str, Any]:
    device = resolve_device(config.router_device)
    router.to(device).eval()
    router_dtype = next(router.parameters()).dtype
    layers = physical_full_attention_layers(bundle)
    selected_examples = list(examples)
    if config.maximum_samples is not None:
        selected_examples = selected_examples[: config.maximum_samples]
    if not selected_examples:
        raise ValueError("QA evaluation contains no examples")

    output_path = Path(output_path)
    samples_output_path = Path(samples_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with samples_output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_number, (record, plan) in enumerate(selected_examples, start=1):
            answer_start = int(record.metadata["answer_start_position"])
            if plan.first_future_position_affected_by_retrieval != answer_start - 1:
                raise ValueError("QA requires the answer-aligned retrieval schedule")
            reference = str(record.metadata["answer"])
            state = collect_training_student_state(
                bundle,
                record,
                plan,
                student_config,
                block_size=plan_config.block_size,
                capture_layers=layers,
            )
            query = state.query_summary.unsqueeze(0).to(
                device=device, dtype=router_dtype
            )
            blocks = state.block_summaries.unsqueeze(0).to(
                device=device, dtype=router_dtype
            )
            candidate_mask = torch.ones(
                (1, blocks.shape[1]), dtype=torch.bool, device=device
            )
            region = router(query, blocks, candidate_mask)
            selected_indices = tuple(
                index
                for index, keep in enumerate(region.hard_mask[0].tolist())
                if keep
            )
            all_indices = tuple(range(len(plan.candidate_blocks)))
            selected_history = pack_block_layer_kv(
                state, selected_indices, layers
            )
            all_history = pack_block_layer_kv(state, all_indices, layers)

            conditions: dict[str, dict[str, Any]] = {}
            conditions["full_context"] = _generation_metrics(
                reference,
                _generate_full_context(
                    bundle,
                    record.token_ids[:answer_start],
                    maximum_new_tokens=config.maximum_new_tokens,
                    prefill_chunk_size=config.prefill_chunk_size,
                ),
            )
            conditions["evidence_only"] = _generation_metrics(
                reference,
                _generate_full_context(
                    bundle,
                    _evidence_only_prompt(record),
                    maximum_new_tokens=config.maximum_new_tokens,
                    prefill_chunk_size=config.prefill_chunk_size,
                ),
            )
            generation_arguments = {
                "bundle": bundle,
                "local_layer_kv": state.local_layer_kv,
                "local_positions": state.local_positions,
                "initial_token_id": record.token_ids[
                    plan.first_future_position_affected_by_retrieval
                ],
                "initial_logical_position": (
                    plan.first_future_position_affected_by_retrieval
                ),
                "local_context_length": student_config.local_context_length,
                "block_size": plan_config.block_size,
                "maximum_new_tokens": config.maximum_new_tokens,
            }
            conditions["local_only"] = _generation_metrics(
                reference,
                _generate_sparse_replay(
                    historical_layer_kv={}, **generation_arguments
                ),
            )
            conditions["all_history_upper_bound"] = _generation_metrics(
                reference,
                _generate_sparse_replay(
                    historical_layer_kv=all_history, **generation_arguments
                ),
            )
            region_metrics = _generation_metrics(
                reference,
                _generate_sparse_replay(
                    historical_layer_kv=selected_history, **generation_arguments
                ),
            )
            candidate_count = len(plan.candidate_blocks)
            selected_tokens = sum(
                plan.candidate_blocks[index].length for index in selected_indices
            )
            region_metrics.update(
                {
                    "selected_block_count": len(selected_indices),
                    "selected_block_fraction": len(selected_indices)
                    / candidate_count,
                    "selected_historical_tokens": selected_tokens,
                    **_evidence_block_metrics(record, plan, selected_indices),
                }
            )
            conditions["region_router"] = region_metrics
            row = {
                "qa_schema_version": QA_SCHEMA_VERSION,
                "sample_id": plan.sample_id,
                "sequence_id": record.sequence_id,
                "reference": reference,
                "candidate_blocks": candidate_count,
                "query_scale_mean": float(region.query_scale.mean().cpu()),
                "conditions": conditions,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if config.progress_every and (
                sample_number == 1
                or sample_number == len(selected_examples)
                or sample_number % config.progress_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "event": "qa_progress",
                            "completed": sample_number,
                            "total": len(selected_examples),
                            "sample_id": plan.sample_id,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    result = {
        "qa_schema_version": QA_SCHEMA_VERSION,
        "evaluation_kind": "output_preserving_region_router_autoregressive_qa",
        "teacher_forcing": False,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_fingerprint": bundle.fingerprint,
        "qa_config": asdict(config),
        "selection": "single_diagonal_gaussian_hard_radius",
        "teacher_attention_used": False,
        "conditions": [
            "full_context",
            "evidence_only",
            "local_only",
            "all_history_upper_bound",
            "region_router",
        ],
        "summary": _aggregate(rows),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["RegionQAEvaluationConfig", "evaluate_autoregressive_qa"]
