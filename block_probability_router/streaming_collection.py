from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from cluster_router_experiment.block_streaming import (
    BlockAlignedRollingContextCollector,
    EvictedCompleteBlock,
)
from learnable_index.aligned_builder import AlignedCollectionConfig
from learnable_index.collectors import StudentCollectionConfig, TeacherAttentionCollector
from learnable_index.config import AttentionAggregationConfig
from learnable_index.contracts import RetrievalSample
from learnable_index.data import RetrievalDataset, save_dataset
from learnable_index.model_adapter import ModelBundle, load_frozen_gemma
from learnable_index.planning import (
    PlanConfig,
    RetrievalPlan,
    SequenceRecord,
    build_retrieval_plans,
    load_sequence_records,
)


STUDENT_STATE_PROTOCOL = "single_pass_block_aligned_streaming_v1"


@dataclass(frozen=True)
class StreamingStudentState:
    """Student states captured from one causal, block-aligned stream."""

    query_summary: torch.Tensor
    block_summaries: torch.Tensor
    block_layer_kv: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]
    local_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_positions: tuple[int, ...]
    forward_calls: int
    forwarded_tokens: int
    evicted_blocks: int
    evicted_tokens: int
    maximum_forward_context_length: int


def collect_streaming_student_state(
    bundle: ModelBundle,
    record: SequenceRecord,
    plan: RetrievalPlan | RetrievalSample,
    student_config: StudentCollectionConfig,
    *,
    block_size: int,
    capture_layers: Sequence[int] = (),
) -> StreamingStudentState:
    """Capture query, block, local, and optional payload states in one pass.

    Complete historical blocks are captured only by the collector's atomic
    block-eviction callback.  Neither historical blocks nor the final local
    cache are recomputed from isolated windows.
    """

    block_size = int(block_size)
    capture_layers = tuple(sorted({int(layer) for layer in capture_layers}))
    if block_size <= 0 or any(layer < 0 for layer in capture_layers):
        raise ValueError("block size and capture layers must be non-negative")
    nominal_local_length = int(plan.local_context_end - plan.local_context_start)
    if nominal_local_length != student_config.local_context_length:
        raise ValueError("retrieval plan and streaming student context lengths differ")
    candidate_index = {
        block.block_id: index for index, block in enumerate(plan.candidate_blocks)
    }
    if len(candidate_index) != len(plan.candidate_blocks):
        raise ValueError("candidate block identifiers must be unique")

    ready_summaries: dict[object, torch.Tensor] = {}
    captured_kv: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    evicted_candidates: set[object] = set()

    def on_block_ready(block, residual_summary: torch.Tensor) -> None:
        if block.block_id in candidate_index:
            if block.block_id in ready_summaries:
                raise RuntimeError("candidate block summary was produced twice")
            ready_summaries[block.block_id] = residual_summary.detach()

    def on_evict(evicted: EvictedCompleteBlock) -> None:
        block_id = evicted.block.block_id
        index = candidate_index.get(block_id)
        if index is None:
            return
        expected_positions = tuple(
            range(evicted.block.start_position, evicted.block.end_position)
        )
        if evicted.logical_positions != expected_positions:
            raise RuntimeError("candidate entered memory as an incomplete block")
        if len(expected_positions) != block_size:
            raise RuntimeError("candidate memory record is not one complete block")
        if block_id in evicted_candidates:
            raise RuntimeError("candidate block was evicted twice")
        evicted_candidates.add(block_id)
        if capture_layers:
            if capture_layers[-1] >= len(evicted.layer_kv):
                raise IndexError("capture layer is outside the physical cache")
            captured_kv[index] = {
                layer: (
                    evicted.layer_kv[layer][0].detach(),
                    evicted.layer_kv[layer][1].detach(),
                )
                for layer in capture_layers
            }

    query_length = (
        1
        if student_config.query_summary == "last"
        else student_config.query_summary_length
    )
    result = BlockAlignedRollingContextCollector(
        bundle,
        local_context_length=student_config.local_context_length,
        block_size=block_size,
        residual_layer=student_config.residual_layer,
        query_summary_length=query_length,
    ).collect(
        record,
        plan,
        on_block_ready=on_block_ready,
        on_evict=on_evict,
    )

    expected_ids = set(candidate_index)
    if set(ready_summaries) != expected_ids:
        missing = sorted(map(str, expected_ids.difference(ready_summaries)))
        raise RuntimeError(f"streaming collection missed candidate summaries: {missing[:3]}")
    if evicted_candidates != expected_ids:
        missing = sorted(map(str, expected_ids.difference(evicted_candidates)))
        raise RuntimeError(f"streaming collection missed candidate evictions: {missing[:3]}")
    if capture_layers and set(captured_kv) != set(range(len(plan.candidate_blocks))):
        raise RuntimeError("streaming collection did not capture every candidate payload")
    local_length = len(result.local_positions)
    if not (
        student_config.local_context_length
        <= local_length
        < student_config.local_context_length + block_size
    ):
        raise RuntimeError("final block-aligned local cache is outside its dynamic bounds")
    if result.local_positions[0] % block_size:
        raise RuntimeError("final local cache does not start on a block boundary")

    return StreamingStudentState(
        query_summary=result.query_summary.detach(),
        block_summaries=torch.stack(
            [ready_summaries[block.block_id] for block in plan.candidate_blocks]
        ).detach(),
        block_layer_kv=captured_kv,
        local_layer_kv=result.local_layer_kv,
        local_positions=result.local_positions,
        forward_calls=int(result.forward_calls),
        forwarded_tokens=int(result.forwarded_tokens),
        evicted_blocks=int(result.evicted_blocks),
        evicted_tokens=int(result.evicted_tokens),
        maximum_forward_context_length=int(result.maximum_forward_context_length),
    )


def _save_sequences(root: Path, records: Sequence[SequenceRecord]) -> None:
    torch.save(
        {
            "schema_version": 1,
            "records": [
                {
                    "sequence_id": record.sequence_id,
                    "token_ids": list(record.token_ids),
                    "metadata": record.metadata,
                }
                for record in records
            ],
        },
        root / "sequences.pt",
    )


def _sample_from_states(
    record: SequenceRecord,
    plan: RetrievalPlan,
    state: StreamingStudentState,
    target,
) -> RetrievalSample:
    return RetrievalSample(
        sample_id=plan.sample_id,
        sequence_id=record.sequence_id,
        retrieval_position=plan.retrieval_position,
        first_future_position_affected_by_retrieval=(
            plan.first_future_position_affected_by_retrieval
        ),
        future_horizon_length=plan.future_horizon_length,
        local_context_start=plan.local_context_start,
        local_context_end=plan.local_context_end,
        candidate_blocks=plan.candidate_blocks,
        query_summary=state.query_summary.detach().float().cpu(),
        block_summaries=state.block_summaries.detach().float().cpu(),
        absolute_teacher_block_mass=target.absolute_block_mass,
        total_teacher_historical_mass=target.total_historical_mass,
        conditional_teacher_distribution=target.conditional_block_distribution,
        per_future_teacher_block_mass=target.per_future_absolute_block_mass,
        teacher_layer_head_future_block_mass=target.per_layer_head_future_block_mass,
        aggregation_metadata={
            **target.metadata,
            "student_state_protocol": STUDENT_STATE_PROTOCOL,
        },
        logical_position_metadata={
            "tokenization": "single_aligned_sequence",
            "position_semantics": "original_logical_positions",
            "student_visible_range": [
                state.local_positions[0],
                state.local_positions[-1] + 1,
            ],
            "nominal_student_boundary": [
                plan.local_context_start,
                plan.local_context_end,
            ],
            "teacher_visible_range": [0, plan.future_end],
            "first_future_position_affected_by_retrieval": plan.future_start,
            "student_memory_policy": STUDENT_STATE_PROTOCOL,
            "split_group_id": record.metadata.get(
                "split_group_id", record.sequence_id
            ),
        },
    ).validate()


def collect_streaming_aligned_dataset(
    bundle: ModelBundle,
    records: Iterable[SequenceRecord],
    output_dir: Path | str,
    config: AlignedCollectionConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RetrievalDataset, dict[str, Any]]:
    """Collect full-context labels and block-aligned streaming student states."""

    if config.store_kv_payload:
        raise ValueError(
            "probability-router streaming collection never persists replay KV payloads"
        )
    if config.student.local_context_length % config.plan.block_size:
        raise ValueError("streaming local context must be divisible by block size")
    native_sliding_window = int(bundle.text_config.sliding_window)
    if config.student.local_context_length != native_sliding_window:
        raise ValueError(
            "probability-router streaming context must equal the model's native "
            f"sliding window ({native_sliding_window})"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_list = list(records)
    _save_sequences(output_dir, record_list)
    teacher = TeacherAttentionCollector(
        bundle,
        config.attention,
        prefill_chunk_size=config.teacher_prefill_chunk_size,
    )
    record_plans = [
        (record, build_retrieval_plans(record, config.plan)) for record in record_list
    ]
    total_plans = sum(len(plans) for _record, plans in record_plans)
    samples: list[RetrievalSample] = []
    plan_rows: list[dict[str, Any]] = []
    for record, plans in record_plans:
        for plan in plans:
            state = collect_streaming_student_state(
                bundle,
                record,
                plan,
                config.student,
                block_size=config.plan.block_size,
            )
            target = teacher.collect(record, plan)
            samples.append(_sample_from_states(record, plan, state, target))
            plan_rows.append(
                {
                    "sample_id": plan.sample_id,
                    "sequence_id": plan.sequence_id,
                    "retrieval_position": plan.retrieval_position,
                    "first_future_position_affected_by_retrieval": (
                        plan.first_future_position_affected_by_retrieval
                    ),
                    "future_horizon_length": plan.future_horizon_length,
                    "local_context_start": plan.local_context_start,
                    "local_context_end": plan.local_context_end,
                    "candidate_blocks": [
                        block.to_dict() for block in plan.candidate_blocks
                    ],
                    "streaming_local_start": state.local_positions[0],
                    "streaming_local_end": state.local_positions[-1] + 1,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": len(samples),
                        "total": total_plans,
                        "sample_id": plan.sample_id,
                    }
                )
    if not samples:
        raise ValueError("streaming collection produced no retrieval samples")
    with (output_dir / "plans.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in plan_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_dataset(
        output_dir / "dataset",
        samples,
        metadata={
            "kind": "aligned_gemma4_teacher_block_streaming_student",
            "model_fingerprint": bundle.fingerprint,
            "collection_config": asdict(config),
            "student_state_protocol": STUDENT_STATE_PROTOCOL,
            "teacher_attention_used_as_input": False,
            "sequences": "../sequences.pt",
            "plans": "../plans.jsonl",
        },
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint": bundle.fingerprint,
        "collection_config": asdict(config),
        "student_state_protocol": STUDENT_STATE_PROTOCOL,
        "sequence_count": len(record_list),
        "sample_count": len(samples),
        "kv_block_count": 0,
        "kv_payload_stored": False,
        "dataset_dir": "dataset",
        "plans": "plans.jsonl",
        "sequences": "sequences.pt",
        "student_memory_policy": (
            "single_pass_block_aligned_no_recurrent_retrieval_during_collection"
        ),
    }
    (output_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return RetrievalDataset(samples), manifest


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Block-aligned streaming collection for the probability router"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--local-context-length", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--future-horizon", type=int, required=True)
    parser.add_argument("--retrieval-interval", type=int, required=True)
    parser.add_argument(
        "--retrieval-point-policy",
        choices=("interval", "metadata"),
        required=True,
    )
    parser.add_argument("--minimum-candidate-blocks", type=int, required=True)
    parser.add_argument("--maximum-candidate-blocks", type=int)
    parser.add_argument("--residual-layer", type=int, required=True)
    parser.add_argument("--query-summary", choices=("last", "mean"), required=True)
    parser.add_argument("--query-summary-length", type=int, required=True)
    parser.add_argument("--teacher-layers", default="all")
    parser.add_argument("--teacher-heads", default="all")
    parser.add_argument("--future-reduction", choices=("mean", "sum"), required=True)
    parser.add_argument("--teacher-prefill-chunk-size", type=int)
    parser.add_argument("--no-store-kv-payload", action="store_true")
    parser.add_argument("--length-normalize-blocks", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _config(arguments: argparse.Namespace) -> AlignedCollectionConfig:
    if not arguments.no_store_kv_payload:
        raise ValueError("streaming probability-router collection requires no KV persistence")
    return AlignedCollectionConfig(
        plan=PlanConfig(
            local_context_length=arguments.local_context_length,
            block_size=arguments.block_size,
            future_horizon_length=arguments.future_horizon,
            retrieval_interval=arguments.retrieval_interval,
            minimum_candidate_blocks=arguments.minimum_candidate_blocks,
            maximum_candidate_blocks=arguments.maximum_candidate_blocks,
            retrieval_point_policy=arguments.retrieval_point_policy,
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
        teacher_prefill_chunk_size=arguments.teacher_prefill_chunk_size,
        store_kv_payload=False,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    bundle = load_frozen_gemma(
        arguments.model_name,
        device=arguments.model_device,
        dtype=arguments.dtype,
        local_files_only=not arguments.allow_network,
    )
    records = load_sequence_records(
        arguments.input_jsonl,
        bundle.tokenizer,
        maximum_sequences=arguments.max_sequences,
        maximum_tokens=arguments.max_tokens,
    )

    def progress(event: Mapping[str, Any]) -> None:
        completed = int(event["completed"])
        total = int(event["total"])
        if (
            arguments.progress_every
            and (
                completed == 1
                or completed == total
                or completed % arguments.progress_every == 0
            )
        ):
            print(
                json.dumps(
                    {"event": "streaming_collection_progress", **event},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    _dataset, manifest = collect_streaming_aligned_dataset(
        bundle,
        records,
        arguments.output_dir,
        _config(arguments),
        progress_callback=progress,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STUDENT_STATE_PROTOCOL",
    "StreamingStudentState",
    "collect_streaming_aligned_dataset",
    "collect_streaming_student_state",
]
