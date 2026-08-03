from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch.nn import functional as F

from .aligned_builder import load_collected_sequences
from .collectors import RestrictedStudentCollector, StudentCollectionConfig
from .data import load_dataset
from .kv_store import KVBlockStore, merge_layer_kv
from .model_adapter import (
    ModelBundle,
    build_sparse_prefix_mask,
    cache_from_layer_kv,
    concatenate_layer_kv,
    forward_tokens,
    layer_kv_from_cache,
    trim_prefix_and_local_kv,
)
from .planning import SequenceRecord
from .retrieval import (
    RetrievalPolicyConfig,
    decide_retrieval,
    oracle_indices,
    recent_indices,
    score_retrieval_sample,
)
from .trainer import load_checkpoint, resolve_device


@dataclass(frozen=True)
class ReplayConfig:
    policy: RetrievalPolicyConfig = RetrievalPolicyConfig()
    maximum_samples: int | None = None
    router_device: str = "cpu"
    verify_query_summary: bool = True
    query_verification_atol: float = 1e-4

    def __post_init__(self) -> None:
        if self.maximum_samples is not None and self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        if self.query_verification_atol <= 0:
            raise ValueError("query_verification_atol must be positive")


def _move_layer_kv(layer_kv, devices):
    if len(layer_kv) != len(devices):
        raise ValueError("KV physical layer count does not match model device map")
    return tuple(
        (key.to(devices[layer_index]), value.to(devices[layer_index]))
        for layer_index, (key, value) in enumerate(layer_kv)
    )


def _kv_bytes(layer_kv) -> int:
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in layer_kv
    )


@torch.no_grad()
def _full_context_logits(
    bundle: ModelBundle,
    record: SequenceRecord,
    sample,
) -> tuple[torch.Tensor, int]:
    end = sample.first_future_position_affected_by_retrieval + sample.future_horizon_length
    output = forward_tokens(
        bundle,
        record.token_ids[:end],
        range(end),
        use_cache=False,
    )
    start = sample.first_future_position_affected_by_retrieval
    # Logical per-layer query/key pairs actually computed by this direct full
    # teacher-forced forward (causal triangle over the visible prefix).
    attention_pairs = end * (end + 1) // 2
    return output.logits[0, start:end].detach().float().cpu(), attention_pairs


@torch.no_grad()
def _restricted_logits(
    bundle: ModelBundle,
    record: SequenceRecord,
    sample,
    local_layer_kv,
    prefix_layer_kv,
) -> tuple[torch.Tensor, int, int]:
    layer_devices = bundle.cache_layer_devices
    prefix_layer_kv = _move_layer_kv(prefix_layer_kv, layer_devices) if prefix_layer_kv else ()
    local_layer_kv = _move_layer_kv(local_layer_kv, layer_devices)
    prefix_length = (
        int(prefix_layer_kv[0][0].shape[2]) if prefix_layer_kv else 0
    )
    combined = concatenate_layer_kv(prefix_layer_kv, local_layer_kv)
    future_start = sample.first_future_position_affected_by_retrieval
    future_end = future_start + sample.future_horizon_length
    logits: list[torch.Tensor] = []
    attention_pairs = 0
    for logical_position in range(future_start, future_end):
        # Including the current token, visible local content must stay <= 256.
        combined = trim_prefix_and_local_kv(
            combined,
            prefix_length=prefix_length,
            maximum_local_tokens=(sample.local_context_end - sample.local_context_start) - 1,
        )
        local_past_length = int(combined[0][0].shape[2]) - prefix_length
        cache = cache_from_layer_kv(combined)
        mask = build_sparse_prefix_mask(
            bundle,
            query_length=1,
            prefix_length=prefix_length,
            local_past_length=local_past_length,
        )
        attention_pairs += prefix_length + local_past_length + 1
        output = forward_tokens(
            bundle,
            [record.token_ids[logical_position]],
            [logical_position],
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
        )
        logits.append(output.logits[0, -1].detach().float().cpu())
        combined = tuple(
            (key.detach(), value.detach())
            for key, value in layer_kv_from_cache(output.past_key_values)
        )
    return torch.stack(logits), _kv_bytes(prefix_layer_kv), attention_pairs


def _condition_metrics(
    logits: torch.Tensor,
    full_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    nll = F.cross_entropy(logits, targets, reduction="mean")
    full_probabilities = full_logits.softmax(dim=-1)
    kl = F.kl_div(
        logits.log_softmax(dim=-1),
        full_probabilities,
        reduction="batchmean",
    )
    agreement = (logits.argmax(dim=-1) == full_logits.argmax(dim=-1)).float().mean()
    target_accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return {
        "next_token_nll": float(nll),
        "kl_from_full_context": max(0.0, float(kl)),
        "full_argmax_agreement": float(agreement),
        "teacher_forced_token_accuracy": float(target_accuracy),
        "predicted_token_ids": logits.argmax(dim=-1).tolist(),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({condition for row in rows for condition in row["conditions"]})
    result: dict[str, Any] = {"sample_count": len(rows), "conditions": {}}
    for condition in conditions:
        metrics = [row["conditions"][condition] for row in rows if condition in row["conditions"]]
        names = sorted({name for metric in metrics for name in metric if isinstance(metric[name], (int, float))})
        result["conditions"][condition] = {
            name: sum(float(metric[name]) for metric in metrics) / len(metrics)
            for name in names
        }
    return result


def evaluate_retrieval_replay(
    bundle: ModelBundle,
    collection_dir: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    config: ReplayConfig,
) -> dict[str, Any]:
    collection_dir = Path(collection_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, dataset_manifest = load_dataset(collection_dir / "dataset")
    if dataset_manifest["metadata"]["model_fingerprint"] != bundle.fingerprint:
        raise ValueError("collection and replay model fingerprints do not match")
    records = {record.sequence_id: record for record in load_collected_sequences(collection_dir)}
    store = KVBlockStore(collection_dir / "kv_store")
    with (collection_dir / "collection_manifest.json").open("r", encoding="utf-8") as handle:
        collection_manifest = json.load(handle)
    student_config = StudentCollectionConfig(
        **collection_manifest["collection_config"]["student"]
    )
    student = RestrictedStudentCollector(bundle, student_config)
    router, _, loss_config, _, checkpoint = load_checkpoint(checkpoint_path)
    router_device = resolve_device(config.router_device)
    rows: list[dict[str, Any]] = []

    samples: Iterable = dataset.samples
    if config.maximum_samples is not None:
        samples = dataset.samples[: config.maximum_samples]
    for sample in samples:
        record = records[sample.sequence_id]
        if sample.first_future_position_affected_by_retrieval + sample.future_horizon_length >= len(
            record.token_ids
        ):
            raise ValueError("replay sample lacks next-token targets after its horizon")
        start_time = time.perf_counter()
        fresh_query = student.collect_query(record, sample)
        if config.verify_query_summary and not torch.allclose(
            fresh_query,
            sample.query_summary,
            atol=config.query_verification_atol,
            rtol=0,
        ):
            maximum_error = float((fresh_query - sample.query_summary).abs().max())
            raise RuntimeError(f"student query recollection drifted by {maximum_error}")
        local_layer_kv = student.collect_local_cache(record, sample)
        full_start = time.perf_counter()
        full_logits, full_attention_pairs = _full_context_logits(bundle, record, sample)
        full_latency = time.perf_counter() - full_start
        targets = torch.tensor(
            record.token_ids[
                sample.first_future_position_affected_by_retrieval
                + 1 : sample.first_future_position_affected_by_retrieval
                + sample.future_horizon_length
                + 1
            ],
            dtype=torch.long,
        )
        router_start = time.perf_counter()
        scores, demand_logit = score_retrieval_sample(
            router,
            sample,
            device=router_device,
        )
        decision = decide_retrieval(
            sample,
            scores,
            demand_logit,
            config.policy,
            demand_loss=loss_config.demand_loss,
        )
        router_latency = time.perf_counter() - router_start
        predicted_blocks = store.load_many(decision.selected_block_ids)
        if any(block.model_fingerprint != bundle.fingerprint for block in predicted_blocks):
            raise RuntimeError("predicted KV payload model fingerprint mismatch")
        predicted_prefix = merge_layer_kv(predicted_blocks, device="cpu")
        matched_budget = decision.requested_top_n
        oracle_idx = oracle_indices(sample, matched_budget) if matched_budget else ()
        oracle_ids = [sample.candidate_blocks[index].block_id for index in oracle_idx]
        oracle_blocks = store.load_many(oracle_ids)
        if any(block.model_fingerprint != bundle.fingerprint for block in oracle_blocks):
            raise RuntimeError("oracle KV payload model fingerprint mismatch")
        oracle_prefix = merge_layer_kv(oracle_blocks, device="cpu")
        recent_idx = recent_indices(sample, matched_budget) if matched_budget else ()
        recent_ids = [sample.candidate_blocks[index].block_id for index in recent_idx]
        recent_blocks = store.load_many(recent_ids)
        if any(block.model_fingerprint != bundle.fingerprint for block in recent_blocks):
            raise RuntimeError("recent KV payload model fingerprint mismatch")
        recent_prefix = merge_layer_kv(recent_blocks, device="cpu")

        conditions: dict[str, dict[str, float]] = {}
        full_metrics = _condition_metrics(full_logits, full_logits, targets)
        full_metrics["selected_block_count"] = 0
        full_metrics["kv_bytes_visible"] = 0
        full_metrics["replay_latency_seconds"] = full_latency
        full_metrics["attention_query_key_pairs_per_layer"] = full_attention_pairs
        conditions["full_context"] = full_metrics
        for name, prefix in (
            ("local_256", ()),
            ("predicted", predicted_prefix),
            ("oracle_top_n", oracle_prefix),
            ("recent_top_n", recent_prefix),
        ):
            replay_start = time.perf_counter()
            logits, kv_bytes, attention_pairs = _restricted_logits(
                bundle,
                record,
                sample,
                local_layer_kv,
                prefix,
            )
            replay_latency = time.perf_counter() - replay_start
            metric = _condition_metrics(logits, full_logits, targets)
            metric["selected_block_count"] = (
                0
                if name == "local_256"
                else len(
                    decision.selected_block_ids
                    if name == "predicted"
                    else oracle_ids if name == "oracle_top_n" else recent_ids
                )
            )
            metric["kv_bytes_visible"] = kv_bytes
            metric["replay_latency_seconds"] = replay_latency
            metric["attention_query_key_pairs_per_layer"] = attention_pairs
            conditions[name] = metric
        rows.append(
            {
                "sample_id": sample.sample_id,
                "sequence_id": sample.sequence_id,
                "retrieval_position": sample.retrieval_position,
                "first_future_position_affected_by_retrieval": (
                    sample.first_future_position_affected_by_retrieval
                ),
                "decision": asdict(decision),
                "oracle_block_ids": oracle_ids,
                "recent_block_ids": recent_ids,
                "target_token_ids": targets.tolist(),
                "router_latency_seconds": router_latency,
                "conditions": conditions,
                "wall_time_seconds": time.perf_counter() - start_time,
            }
        )

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = _summary(rows)
    summary.update(
        {
            "schema_version": 1,
            "model_fingerprint": bundle.fingerprint,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "replay_config": asdict(config),
            "schedule_semantics": (
                "retrieval computed at t; first affected forward consumes token t+1 "
                "and predicts token t+2"
            ),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
