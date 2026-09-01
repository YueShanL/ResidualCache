from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from learnable_index.collectors import StudentCollectionConfig, TeacherAttentionCollector
from learnable_index.config import AttentionAggregationConfig
from learnable_index.model_adapter import load_frozen_gemma
from learnable_index.planning import (
    PlanConfig,
    SequenceRecord,
    build_retrieval_plans,
)
from learnable_index.prepare_convomem import (
    build_convomem_long_sequences,
    iter_convomem_examples,
)

from .model import minimum_cumulative_mass_mask
from .full_context_replay import (
    FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL,
    collect_full_context_replay_state,
)
from .qa import (
    _generate_full_context,
    _generate_sparse_replay,
    _generation_metrics,
    _normalized_answer,
    _pack_selected_block_kv,
    _physical_full_attention_layers,
    token_f1,
)
SMOKE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OracleReplaySmokeConfig:
    sequence_count: int = 4
    sequence_length: int = 4096
    local_context_length: int = 512
    block_size: int = 64
    future_horizon: int = 64
    maximum_answer_tokens: int = 64
    maximum_new_tokens: int = 64
    teacher_prefill_chunk_size: int = 256
    generation_prefill_chunk_size: int = 256
    teacher_layers: tuple[int, ...] = (29, 35, 41)
    residual_layer: int = 40
    query_summary: str = "mean"
    query_summary_length: int = 16
    missing_mass_tolerance: float = 0.02
    seed: int = 13
    sampling_seed: int = 197

    def __post_init__(self) -> None:
        positive = (
            self.sequence_count,
            self.sequence_length,
            self.local_context_length,
            self.block_size,
            self.future_horizon,
            self.maximum_answer_tokens,
            self.maximum_new_tokens,
            self.teacher_prefill_chunk_size,
            self.generation_prefill_chunk_size,
            self.query_summary_length,
        )
        if min(positive) <= 0:
            raise ValueError("oracle replay smoke sizes must be positive")
        if self.local_context_length % self.block_size:
            raise ValueError("local context length must be divisible by block size")
        if not 0 < self.missing_mass_tolerance < 1:
            raise ValueError("missing mass tolerance must be in (0, 1)")
        if not self.teacher_layers or min(self.teacher_layers) < 0:
            raise ValueError("teacher layers must be non-empty and non-negative")
        if self.query_summary not in {"last", "mean"}:
            raise ValueError("query_summary must be 'last' or 'mean'")


def select_teacher_oracle_blocks(
    conditional_distribution: torch.Tensor,
    missing_mass_tolerance: float,
) -> tuple[int, ...]:
    """Select the minimum block set retaining the requested teacher mass."""

    distribution = conditional_distribution.detach().float().cpu()
    if distribution.ndim != 1 or distribution.numel() == 0:
        raise ValueError("teacher block distribution must be a non-empty vector")
    if not torch.isfinite(distribution).all() or torch.any(distribution < 0):
        raise ValueError("teacher block distribution must be finite and non-negative")
    if not torch.isclose(distribution.sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
        raise ValueError("teacher conditional block distribution must sum to one")
    mask = minimum_cumulative_mass_mask(
        distribution.unsqueeze(0),
        torch.ones((1, distribution.numel()), dtype=torch.bool),
        missing_mass_tolerance,
        -1,
    )[0]
    return tuple(index for index, selected in enumerate(mask.tolist()) if selected)


def _sequence_record(row: dict[str, Any]) -> SequenceRecord:
    metadata = dict(row)
    sequence_id = str(metadata.pop("sequence_id"))
    token_ids = tuple(int(value) for value in metadata.pop("token_ids"))
    return SequenceRecord(sequence_id, token_ids, metadata)


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        length += 1
    return length


def _ground_truth_overlap_metrics(
    record: SequenceRecord,
    plan,
    selected_indices: Sequence[int],
    block_size: int,
) -> dict[str, Any]:
    selected_global = {
        plan.candidate_blocks[index].start_position // block_size
        for index in selected_indices
    }
    evidence_global = {
        int(value) for value in record.metadata.get("evidence_block_indices", ())
    }
    eligible_evidence = evidence_global.intersection(
        block.start_position // block_size for block in plan.candidate_blocks
    )
    selected_evidence = selected_global.intersection(eligible_evidence)
    return {
        "metadata_evidence_block_count": len(eligible_evidence),
        "selected_metadata_evidence_block_count": len(selected_evidence),
        "metadata_evidence_block_recall": (
            0.0
            if not eligible_evidence
            else len(selected_evidence) / len(eligible_evidence)
        ),
        "any_metadata_evidence_hit": bool(selected_evidence),
    }


def _mean(rows: Sequence[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / len(values)


def run_oracle_replay_smoke(
    *,
    model_name: str,
    convomem_snapshot: Path | str,
    output_dir: Path | str,
    model_device: str,
    dtype: str,
    local_files_only: bool,
    model_cache_dir: str | None,
    config: OracleReplaySmokeConfig,
) -> dict[str, Any]:
    """Run four independent full-context versus teacher-oracle replay checks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"
    if samples_path.exists() or summary_path.exists():
        raise FileExistsError("oracle replay smoke refuses to overwrite existing results")

    snapshot = Path(convomem_snapshot)
    evidence_root = snapshot / "core_benchmark" / "evidence_questions"
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"ConvoMem evidence root is missing: {evidence_root}")

    started = time.perf_counter()
    bundle = load_frozen_gemma(
        model_name,
        device=model_device,
        dtype=dtype,
        local_files_only=local_files_only,
        cache_dir=model_cache_dir,
    )
    native_sliding_window = int(bundle.text_config.sliding_window)
    if config.local_context_length != native_sliding_window:
        raise ValueError(
            "oracle replay smoke local context must equal the model's native "
            f"sliding window ({native_sliding_window})"
        )
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model_fingerprint": bundle.fingerprint,
                "native_sliding_window": native_sliding_window,
            }
        ),
        flush=True,
    )

    synthesized = build_convomem_long_sequences(
        iter_convomem_examples(evidence_root),
        bundle.tokenizer,
        split="test",
        sequence_length=config.sequence_length,
        sequence_count=config.sequence_count,
        seed=config.seed,
        sampling_seed=config.sampling_seed,
        maximum_answer_tokens=config.maximum_answer_tokens,
        maximum_future_horizon=config.future_horizon,
        evidence_placement="stratified_random",
        evidence_placement_bins=config.sequence_count,
        placement_block_size=config.block_size,
        retrieval_local_context_length=config.local_context_length,
    )
    records = [_sequence_record(row) for row in synthesized]
    plan_config = PlanConfig(
        local_context_length=config.local_context_length,
        block_size=config.block_size,
        future_horizon_length=config.future_horizon,
        retrieval_interval=128,
        minimum_candidate_blocks=2,
        retrieval_point_policy="metadata",
    )
    teacher = TeacherAttentionCollector(
        bundle,
        AttentionAggregationConfig(
            teacher_layers=config.teacher_layers,
            teacher_heads=None,
            future_reduction="mean",
            length_normalize_blocks=False,
        ),
        prefill_chunk_size=config.teacher_prefill_chunk_size,
    )
    full_attention_layers = _physical_full_attention_layers(bundle)

    rows: list[dict[str, Any]] = []
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_number, record in enumerate(records, start=1):
            plans = build_retrieval_plans(record, plan_config)
            if len(plans) != 1:
                raise RuntimeError("oracle replay smoke requires one answer retrieval plan")
            plan = plans[0]
            answer_start = int(record.metadata["answer_start_position"])
            if plan.future_start != answer_start - 1:
                raise RuntimeError("oracle replay plan is not aligned to the first answer token")
            reference = str(record.metadata["answer"])
            print(
                json.dumps(
                    {
                        "event": "sample_start",
                        "completed": sample_number - 1,
                        "total": len(records),
                        "sample_id": plan.sample_id,
                    }
                ),
                flush=True,
            )

            full_result = _generate_full_context(
                bundle,
                record.token_ids[:answer_start],
                maximum_new_tokens=config.maximum_new_tokens,
                prefill_chunk_size=config.generation_prefill_chunk_size,
            )
            print(
                json.dumps(
                    {
                        "event": "full_context_complete",
                        "sample_id": plan.sample_id,
                        "latency_seconds": full_result.latency_seconds,
                    }
                ),
                flush=True,
            )

            target = teacher.collect(record, plan)
            selected_indices = select_teacher_oracle_blocks(
                target.conditional_block_distribution,
                config.missing_mass_tolerance,
            )
            selected_teacher_mass = float(
                target.conditional_block_distribution[list(selected_indices)].sum()
            )
            print(
                json.dumps(
                    {
                        "event": "teacher_oracle_complete",
                        "sample_id": plan.sample_id,
                        "selected_blocks": len(selected_indices),
                        "candidate_blocks": len(plan.candidate_blocks),
                        "retained_teacher_mass": selected_teacher_mass,
                    }
                ),
                flush=True,
            )

            state = collect_full_context_replay_state(
                bundle,
                record,
                plan,
                StudentCollectionConfig(
                    local_context_length=config.local_context_length,
                    residual_layer=config.residual_layer,
                    query_summary=config.query_summary,
                    query_summary_length=config.query_summary_length,
                ),
                block_size=config.block_size,
                prefill_chunk_size=config.generation_prefill_chunk_size,
                capture_layers=full_attention_layers,
            )
            historical = _pack_selected_block_kv(
                state.block_layer_kv,
                selected_indices,
                full_attention_layers,
            )
            initial_position = plan.future_start
            replay_result = _generate_sparse_replay(
                bundle,
                local_layer_kv=state.local_layer_kv,
                local_positions=state.local_positions,
                initial_token_id=record.token_ids[initial_position],
                initial_logical_position=initial_position,
                historical_layer_kv=historical,
                local_context_length=config.local_context_length,
                block_size=config.block_size,
                maximum_new_tokens=config.maximum_new_tokens,
            )
            all_indices = tuple(range(len(plan.candidate_blocks)))
            all_historical = _pack_selected_block_kv(
                state.block_layer_kv,
                all_indices,
                full_attention_layers,
            )
            all_replay_result = _generate_sparse_replay(
                bundle,
                local_layer_kv=state.local_layer_kv,
                local_positions=state.local_positions,
                initial_token_id=record.token_ids[initial_position],
                initial_logical_position=initial_position,
                historical_layer_kv=all_historical,
                local_context_length=config.local_context_length,
                block_size=config.block_size,
                maximum_new_tokens=config.maximum_new_tokens,
            )

            full_metrics = _generation_metrics(reference, full_result)
            replay_metrics = _generation_metrics(reference, replay_result)
            all_replay_metrics = _generation_metrics(reference, all_replay_result)
            selected_tokens = sum(
                plan.candidate_blocks[index].length for index in selected_indices
            )
            row = {
                "smoke_schema_version": SMOKE_SCHEMA_VERSION,
                "sample_id": plan.sample_id,
                "sequence_id": record.sequence_id,
                "reference_answer": reference,
                "question": record.metadata.get("question"),
                "evidence_placement_bin": record.metadata.get("evidence_placement_bin"),
                "evidence_distance_tokens": record.metadata.get(
                    "evidence_to_answer_distance_tokens"
                ),
                "candidate_block_count": len(plan.candidate_blocks),
                "selected_block_count": len(selected_indices),
                "selected_block_indices": list(selected_indices),
                "selected_historical_tokens": selected_tokens,
                "selected_block_fraction": len(selected_indices) / len(plan.candidate_blocks),
                "teacher_total_historical_mass": float(target.total_historical_mass),
                "retained_conditional_teacher_mass": selected_teacher_mass,
                "full_context": full_metrics,
                "teacher_oracle_replay": replay_metrics,
                "all_historical_replay": all_replay_metrics,
                "replay_vs_full": {
                    "normalized_exact_match": float(
                        _normalized_answer(replay_result.text)
                        == _normalized_answer(full_result.text)
                    ),
                    "token_f1": token_f1(full_result.text, replay_result.text),
                    "generated_token_exact_match": float(
                        replay_result.token_ids == full_result.token_ids
                    ),
                    "generated_token_common_prefix_length": _common_prefix_length(
                        full_result.token_ids, replay_result.token_ids
                    ),
                    "preserved_full_answer_success": float(
                        not bool(full_metrics["answer_contains"])
                        or bool(replay_metrics["answer_contains"])
                    ),
                },
                "all_historical_replay_vs_full": {
                    "normalized_exact_match": float(
                        _normalized_answer(all_replay_result.text)
                        == _normalized_answer(full_result.text)
                    ),
                    "token_f1": token_f1(full_result.text, all_replay_result.text),
                    "generated_token_exact_match": float(
                        all_replay_result.token_ids == full_result.token_ids
                    ),
                    "generated_token_common_prefix_length": _common_prefix_length(
                        full_result.token_ids, all_replay_result.token_ids
                    ),
                },
                "replay_source_state": {
                    "protocol": FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL,
                    "local_context_start": state.local_positions[0],
                    "local_context_end": state.local_positions[-1] + 1,
                    "local_context_length": len(state.local_positions),
                    "forward_calls": state.forward_calls,
                    "forwarded_tokens": state.forwarded_tokens,
                },
                **_ground_truth_overlap_metrics(
                    record,
                    plan,
                    selected_indices,
                    config.block_size,
                ),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "completed": sample_number,
                        "total": len(records),
                        "sample_id": plan.sample_id,
                        "full_answer_contains": full_metrics["answer_contains"],
                        "replay_answer_contains": replay_metrics["answer_contains"],
                        "replay_vs_full_token_f1": row["replay_vs_full"]["token_f1"],
                        "all_replay_token_exact_match": row[
                            "all_historical_replay_vs_full"
                        ]["generated_token_exact_match"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del all_historical, historical, state, target
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    full_successes = sum(bool(row["full_context"]["answer_contains"]) for row in rows)
    replay_successes = sum(
        bool(row["teacher_oracle_replay"]["answer_contains"]) for row in rows
    )
    preserved_successes = sum(
        bool(row["full_context"]["answer_contains"])
        and bool(row["teacher_oracle_replay"]["answer_contains"])
        for row in rows
    )
    summary = {
        "smoke_schema_version": SMOKE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "teacher_attention_oracle_block_replay_smoke",
        "uses_router": False,
        "uses_memory": False,
        "model_fingerprint": bundle.fingerprint,
        "config": asdict(config),
        "sample_count": len(rows),
        "full_answer_success_count": full_successes,
        "teacher_oracle_replay_answer_success_count": replay_successes,
        "replay_preserved_full_success_count": preserved_successes,
        "replay_preserved_full_success_rate": (
            1.0 if full_successes == 0 else preserved_successes / full_successes
        ),
        "mean_selected_block_count": _mean(rows, ("selected_block_count",)),
        "mean_selected_block_fraction": _mean(rows, ("selected_block_fraction",)),
        "mean_retained_conditional_teacher_mass": _mean(
            rows, ("retained_conditional_teacher_mass",)
        ),
        "mean_full_answer_token_f1": _mean(
            rows, ("full_context", "answer_token_f1")
        ),
        "mean_replay_answer_token_f1": _mean(
            rows, ("teacher_oracle_replay", "answer_token_f1")
        ),
        "mean_replay_vs_full_token_f1": _mean(
            rows, ("replay_vs_full", "token_f1")
        ),
        "normalized_output_match_count": sum(
            bool(row["replay_vs_full"]["normalized_exact_match"]) for row in rows
        ),
        "all_historical_generated_token_exact_match_count": sum(
            bool(
                row["all_historical_replay_vs_full"][
                    "generated_token_exact_match"
                ]
            )
            for row in rows
        ),
        "mean_all_historical_replay_vs_full_token_f1": _mean(
            rows, ("all_historical_replay_vs_full", "token_f1")
        ),
        "replay_structure_valid": all(
            bool(
                row["all_historical_replay_vs_full"][
                    "generated_token_exact_match"
                ]
            )
            for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "samples": "samples.jsonl",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "smoke_complete", **summary}, ensure_ascii=False), flush=True)
    return summary


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not layers:
        raise ValueError("teacher layer list cannot be empty")
    return layers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-sample full-context teacher-oracle KV replay smoke"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--convomem-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--sequence-count", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--local-context-length", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--future-horizon", type=int, default=64)
    parser.add_argument("--maximum-answer-tokens", type=int, default=64)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-prefill-chunk-size", type=int, default=256)
    parser.add_argument("--generation-prefill-chunk-size", type=int, default=256)
    parser.add_argument("--teacher-layers", default="29,35,41")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sampling-seed", type=int, default=197)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = OracleReplaySmokeConfig(
        sequence_count=arguments.sequence_count,
        sequence_length=arguments.sequence_length,
        local_context_length=arguments.local_context_length,
        block_size=arguments.block_size,
        future_horizon=arguments.future_horizon,
        maximum_answer_tokens=arguments.maximum_answer_tokens,
        maximum_new_tokens=arguments.maximum_new_tokens,
        teacher_prefill_chunk_size=arguments.teacher_prefill_chunk_size,
        generation_prefill_chunk_size=arguments.generation_prefill_chunk_size,
        teacher_layers=_parse_layers(arguments.teacher_layers),
        missing_mass_tolerance=arguments.epsilon,
        seed=arguments.seed,
        sampling_seed=arguments.sampling_seed,
    )
    run_oracle_replay_smoke(
        model_name=arguments.model_name,
        convomem_snapshot=arguments.convomem_snapshot,
        output_dir=arguments.output_dir,
        model_device=arguments.model_device,
        dtype=arguments.dtype,
        local_files_only=not arguments.allow_network,
        model_cache_dir=arguments.model_cache_dir,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OracleReplaySmokeConfig",
    "collect_full_context_replay_state",
    "run_oracle_replay_smoke",
    "select_teacher_oracle_blocks",
]
