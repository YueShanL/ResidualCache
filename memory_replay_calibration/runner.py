from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from cluster_router_experiment.convomem import DynamicConvoMemDataset
from cluster_router_experiment.gemma4 import Gemma4ClusterRouterModel
from cluster_router_validation.metrics import exact_match, token_f1


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _distribution_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    targets: Sequence[int],
) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate logits must have the same shape")
    target = torch.tensor(tuple(targets), dtype=torch.int64)
    if reference.shape[0] != target.numel():
        raise ValueError("target token count does not match logits")
    reference_logp = torch.log_softmax(reference.float(), dim=-1)
    candidate_logp = torch.log_softmax(candidate.float(), dim=-1)
    reference_probability = reference_logp.exp()
    kl = (reference_probability * (reference_logp - candidate_logp)).sum(dim=-1)
    positions = torch.arange(target.numel(), dtype=torch.int64)
    return {
        "kl_from_full_context": float(kl.mean().item()),
        "full_argmax_agreement": float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean().item()
        ),
        "next_token_nll": float((-candidate_logp[positions, target]).mean().item()),
        "teacher_forced_token_accuracy": float(
            (candidate.argmax(dim=-1) == target).float().mean().item()
        ),
    }


def _prediction(bundle, logits: torch.Tensor) -> tuple[str, tuple[int, ...]]:
    ids = tuple(int(value) for value in logits.argmax(dim=-1).tolist())
    return bundle.tokenizer.decode(ids, skip_special_tokens=True).strip(), ids


def _condition_metrics(
    *,
    bundle,
    reference_logits: torch.Tensor,
    uncompressed_logits: torch.Tensor | None = None,
    logits: torch.Tensor,
    targets: Sequence[int],
    reference_answer: str,
) -> dict[str, Any]:
    text, token_ids = _prediction(bundle, logits)
    result: dict[str, Any] = {
        "prediction_text": text,
        "prediction_token_ids": list(token_ids),
        "answer_exact_match": exact_match(reference_answer, text),
        "answer_token_f1": token_f1(reference_answer, text),
    }
    result.update(_distribution_metrics(reference_logits, logits, targets))
    if uncompressed_logits is not None:
        replay_metrics = _distribution_metrics(uncompressed_logits, logits, targets)
        result["kl_from_uncompressed_replay"] = replay_metrics[
            "kl_from_full_context"
        ]
        result["uncompressed_argmax_agreement"] = replay_metrics[
            "full_argmax_agreement"
        ]
    return result


def _evidence_count(session, positions: Mapping[int, Sequence[int]]) -> tuple[int, int]:
    present = sum(len(values) for values in positions.values())
    evidence = sum(
        session._is_evidence_position(int(position))
        for values in positions.values()
        for position in values
    )
    return present, evidence


def _aggregate(rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]):
    full_f1 = _mean([float(row["full_context"]["answer_token_f1"]) for row in rows])
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "full_context": {
            "answer_exact_match": _mean(
                [float(row["full_context"]["answer_exact_match"]) for row in rows]
            ),
            "answer_token_f1": full_f1,
        },
        "uncompressed_full_replay": {},
        "thresholds": {},
    }
    metric_names = (
        "answer_exact_match",
        "answer_token_f1",
        "kl_from_full_context",
        "full_argmax_agreement",
        "next_token_nll",
        "teacher_forced_token_accuracy",
        "retained_record_ratio",
        "retained_evidence_ratio",
        "latency_seconds",
        "kl_from_uncompressed_replay",
        "uncompressed_argmax_agreement",
    )
    replay = [row["uncompressed_full_replay"] for row in rows]
    for name in metric_names:
        values = [float(value[name]) for value in replay if name in value]
        if values:
            result["uncompressed_full_replay"][name] = _mean(values)
    for threshold in thresholds:
        key = format(float(threshold), ".12g")
        conditions = [row["thresholds"][key] for row in rows]
        summary = {}
        for name in metric_names:
            values = [float(value[name]) for value in conditions if name in value]
            if values:
                summary[name] = _mean(values)
        summary["answer_token_f1_delta_from_full"] = (
            summary["answer_token_f1"] - full_f1
        )
        result["thresholds"][key] = summary
    return result


def _choose_setting(summary: Mapping[str, Any], criteria: Mapping[str, Any]):
    max_kl = float(criteria.get("max_mean_kl_from_uncompressed_replay", 0.02))
    min_agreement = float(
        criteria.get("min_mean_uncompressed_argmax_agreement", 0.98)
    )
    max_replay_kl_from_full = float(
        criteria.get(
            "max_mean_uncompressed_replay_kl_from_full_context", float("inf")
        )
    )
    min_replay_agreement_with_full = float(
        criteria.get(
            "min_mean_uncompressed_replay_full_argmax_agreement", 0.0
        )
    )
    max_f1_drop = float(criteria.get("max_mean_answer_f1_drop_from_full", 0.02))
    replay = summary["uncompressed_full_replay"]
    replay_f1_drop = (
        float(replay["answer_token_f1"])
        - float(summary["full_context"]["answer_token_f1"])
    )
    replay_gate_checks = {
        "answer_f1": replay_f1_drop >= -max_f1_drop,
        "kl_from_full_context": (
            float(replay["kl_from_full_context"]) <= max_replay_kl_from_full
        ),
        "full_argmax_agreement": (
            float(replay["full_argmax_agreement"])
            >= min_replay_agreement_with_full
        ),
    }
    replay_gate = all(replay_gate_checks.values())
    eligible = []
    for threshold, values in summary["thresholds"].items():
        if (
            replay_gate
            and float(values["kl_from_uncompressed_replay"]) <= max_kl
            and float(values["uncompressed_argmax_agreement"]) >= min_agreement
            and float(values["answer_token_f1_delta_from_full"]) >= -max_f1_drop
        ):
            eligible.append((float(values["retained_record_ratio"]), float(threshold)))
    selected = min(eligible) if eligible else None
    return {
        "criteria": {
            "max_mean_kl_from_uncompressed_replay": max_kl,
            "min_mean_uncompressed_argmax_agreement": min_agreement,
            "max_mean_uncompressed_replay_kl_from_full_context": (
                max_replay_kl_from_full
            ),
            "min_mean_uncompressed_replay_full_argmax_agreement": (
                min_replay_agreement_with_full
            ),
            "max_mean_answer_f1_drop_from_full": max_f1_drop,
        },
        "uncompressed_replay_f1_delta_from_full": replay_f1_drop,
        "uncompressed_replay_gate_checks": replay_gate_checks,
        "uncompressed_replay_gate_passed": replay_gate,
        "selected_usage_threshold": None if selected is None else selected[1],
        "selected_retained_record_ratio": None if selected is None else selected[0],
        "status": (
            "accepted"
            if selected is not None
            else (
                "uncompressed_replay_not_equivalent"
                if not replay_gate
                else "no_setting_met_constraints"
            )
        ),
    }


def run_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(sorted(set(float(value) for value in config["usage_thresholds"])))
    if not thresholds or thresholds[0] < 0.0:
        raise ValueError("usage_thresholds must contain non-negative values")

    model_values = dict(config["model"])
    model = Gemma4ClusterRouterModel(**model_values)
    dataset = DynamicConvoMemDataset(**dict(config["data"]))
    rows: list[dict[str, Any]] = []
    samples_path = output_dir / "sample_metrics.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for example in dataset:
            print(f"[{example.sample_id}] memory replay calibration", flush=True)
            session = model.open(example)
            try:
                full = session.run_full_context()
                reference_logits = full.distribution_payload
                full_text = full.predicted_text
                full_metrics = {
                    "prediction_text": full_text,
                    "prediction_token_ids": list(full.predicted_token_ids),
                    "answer_exact_match": exact_match(example.reference_answer, full_text),
                    "answer_token_f1": token_f1(example.reference_answer, full_text),
                }
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                (
                    replay_logits,
                    record_ids,
                    positions,
                    usage,
                    usage_state,
                ) = session.collect_full_replay_usage()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                replay_latency = time.perf_counter() - started
                total_records, total_evidence = _evidence_count(session, positions)
                replay_metrics = _condition_metrics(
                    bundle=model.bundle,
                    reference_logits=reference_logits,
                    logits=replay_logits,
                    targets=session.target_token_ids,
                    reference_answer=example.reference_answer,
                )
                replay_metrics.update(
                    {
                        "retained_record_ratio": 1.0,
                        "retained_evidence_ratio": 1.0 if total_evidence else 0.0,
                        "retained_record_count": total_records,
                        "retained_evidence_count": total_evidence,
                        "latency_seconds": replay_latency,
                    }
                )
                threshold_rows: dict[str, Any] = {}
                for threshold in thresholds:
                    retained: dict[int, tuple[str, ...]] = {}
                    planned_evictions = 0
                    for layer, ids in record_ids.items():
                        plan = session.memories[layer].plan_recall_eviction(
                            ids,
                            usage[layer],
                            usage_threshold=threshold,
                        )
                        retained[layer] = plan["retained_record_ids"]
                        planned_evictions += len(plan["evicted_record_ids"])
                    layer_kv, aligned_ids, retained_positions = session.packed_record_plan(
                        retained
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    compressed_logits = session._restricted_logits(layer_kv)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - started
                    retained_count, retained_evidence = _evidence_count(
                        session, retained_positions
                    )
                    metrics = _condition_metrics(
                        bundle=model.bundle,
                        reference_logits=reference_logits,
                        uncompressed_logits=replay_logits,
                        logits=compressed_logits,
                        targets=session.target_token_ids,
                        reference_answer=example.reference_answer,
                    )
                    metrics.update(
                        {
                            "retained_record_ratio": retained_count / total_records,
                            "retained_evidence_ratio": (
                                retained_evidence / total_evidence
                                if total_evidence
                                else 0.0
                            ),
                            "retained_record_count": retained_count,
                            "retained_evidence_count": retained_evidence,
                            "planned_eviction_count": planned_evictions,
                            "latency_seconds": latency,
                            "records_by_layer": {
                                str(layer): len(ids)
                                for layer, ids in sorted(aligned_ids.items())
                            },
                        }
                    )
                    threshold_rows[format(threshold, ".12g")] = metrics
                row = {
                    "sample_id": example.sample_id,
                    "sequence_length": example.sequence_length,
                    "evidence_distance_tokens": example.evidence_distance_tokens,
                    "full_context": full_metrics,
                    "uncompressed_full_replay": replay_metrics,
                    "thresholds": threshold_rows,
                    "memory_before_eviction": {
                        str(layer): memory.snapshot()
                        for layer, memory in sorted(session.memories.items())
                    },
                    "usage_adapter": usage_state,
                }
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            finally:
                session.close()
    summary = _aggregate(rows, thresholds)
    summary["selection"] = _choose_setting(summary, dict(config.get("criteria", {})))
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "reference": "full_context_teacher_forced",
            "usage_pass": "uncompressed_full_memory_replay",
            "compression_scope": "within_recalled_cluster_only",
            "sweep": "non_mutating_counterfactual_from_shared_usage_pass",
            "router_selection": False,
        },
        "thresholds": list(thresholds),
        "model": model.descriptor,
        "dataset": dataset.descriptor,
        "summary": summary,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "config.resolved.json").write_text(
        json.dumps(dict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
