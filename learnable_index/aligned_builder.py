from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .collectors import (
    RestrictedStudentCollector,
    StudentCollectionConfig,
    TeacherAttentionCollector,
    assemble_retrieval_sample,
)
from .config import AttentionAggregationConfig
from .data import RetrievalDataset, save_dataset
from .kv_store import KVBlockStore
from .model_adapter import ModelBundle
from .planning import PlanConfig, RetrievalPlan, SequenceRecord, build_retrieval_plans


@dataclass(frozen=True)
class AlignedCollectionConfig:
    plan: PlanConfig = PlanConfig()
    student: StudentCollectionConfig = StudentCollectionConfig()
    attention: AttentionAggregationConfig = AttentionAggregationConfig()
    teacher_prefill_chunk_size: int | None = None
    store_kv_payload: bool = True

    def __post_init__(self) -> None:
        if self.plan.local_context_length != self.student.local_context_length:
            raise ValueError("plan and student local context lengths must match")
        if self.teacher_prefill_chunk_size is not None and self.teacher_prefill_chunk_size <= 0:
            raise ValueError("teacher_prefill_chunk_size must be positive when set")


def _save_sequences(root: Path, records: list[SequenceRecord]) -> None:
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


def load_collected_sequences(root: Path | str) -> list[SequenceRecord]:
    try:
        payload = torch.load(Path(root) / "sequences.pt", map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(Path(root) / "sequences.pt", map_location="cpu")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported collected sequence schema")
    return [
        SequenceRecord(
            sequence_id=str(row["sequence_id"]),
            token_ids=tuple(int(token_id) for token_id in row["token_ids"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in payload["records"]
    ]


def _plan_payload(plan: RetrievalPlan) -> dict:
    return {
        "sample_id": plan.sample_id,
        "sequence_id": plan.sequence_id,
        "retrieval_position": plan.retrieval_position,
        "first_future_position_affected_by_retrieval": (
            plan.first_future_position_affected_by_retrieval
        ),
        "future_horizon_length": plan.future_horizon_length,
        "local_context_start": plan.local_context_start,
        "local_context_end": plan.local_context_end,
        "candidate_blocks": [block.to_dict() for block in plan.candidate_blocks],
    }


def collect_aligned_dataset(
    bundle: ModelBundle,
    records: Iterable[SequenceRecord],
    output_dir: Path | str,
    config: AlignedCollectionConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[RetrievalDataset, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_list = list(records)
    _save_sequences(output_dir, record_list)
    store = KVBlockStore(output_dir / "kv_store")
    teacher = TeacherAttentionCollector(
        bundle,
        config.attention,
        prefill_chunk_size=config.teacher_prefill_chunk_size,
    )
    student = RestrictedStudentCollector(bundle, config.student)
    record_plans = [
        (record, build_retrieval_plans(record, config.plan)) for record in record_list
    ]
    total_plans = sum(len(plans) for _, plans in record_plans)
    samples = []
    plan_rows = []
    for record, plans in record_plans:
        for plan in plans:
            blocks = student.ensure_blocks(
                record,
                plan,
                store,
                persist=config.store_kv_payload,
            )
            query_summary = student.collect_query(record, plan)
            target = teacher.collect(record, plan)
            samples.append(
                assemble_retrieval_sample(
                    record,
                    plan,
                    query_summary,
                    blocks,
                    target,
                )
            )
            plan_rows.append(_plan_payload(plan))
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": len(samples),
                        "total": total_plans,
                        "sequence_id": record.sequence_id,
                        "sample_id": plan.sample_id,
                    }
                )
    if not samples:
        raise ValueError("collection produced no retrieval samples; sequence or plan is too short")
    with (output_dir / "plans.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_dir = output_dir / "dataset"
    save_dataset(
        dataset_dir,
        samples,
        metadata={
            "kind": "aligned_gemma4_teacher_student",
            "model_fingerprint": bundle.fingerprint,
            "collection_config": asdict(config),
            "teacher_attention_used_as_input": False,
            "kv_store": "../kv_store",
            "plans": "../plans.jsonl",
            "sequences": "../sequences.pt",
        },
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint": bundle.fingerprint,
        "collection_config": asdict(config),
        "sequence_count": len(record_list),
        "sample_count": len(samples),
        "kv_block_count": len(store.manifest["blocks"]),
        "kv_payload_stored": config.store_kv_payload,
        "dataset_dir": "dataset",
        "kv_store_dir": "kv_store",
        "student_memory_policy": "local_only_no_recurrent_retrieval_during_collection",
    }
    with (output_dir / "collection_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return RetrievalDataset(samples), manifest
