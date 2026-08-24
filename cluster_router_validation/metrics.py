from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import STATE_SCHEMA_VERSION


@dataclass(frozen=True)
class MetricConfig:
    distance_boundaries: tuple[int, ...] = (1_024, 2_048, 4_096, 8_192)
    length_boundaries: tuple[int, ...] = (4_096, 8_192, 16_384)
    kv_ratio_thresholds: tuple[float, ...] = (0.25, 0.5)
    allow_incomplete_state: bool = False

    def __post_init__(self) -> None:
        for name in ("distance_boundaries", "length_boundaries"):
            values = tuple(int(value) for value in getattr(self, name))
            if any(value <= 0 for value in values) or values != tuple(
                sorted(set(values))
            ):
                raise ValueError(f"{name} must be sorted unique positive integers")
            object.__setattr__(self, name, values)
        thresholds = tuple(float(value) for value in self.kv_ratio_thresholds)
        if any(not 0.0 < value <= 1.0 for value in thresholds):
            raise ValueError("kv_ratio_thresholds must lie in (0, 1]")
        object.__setattr__(self, "kv_ratio_thresholds", thresholds)


def _normalized_answer(text: str) -> str:
    words = re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
    words = [word for word in words if word not in {"a", "an", "the"}]
    return " ".join(words)


def _answer_tokens(text: str) -> list[str]:
    normalized = _normalized_answer(text)
    if not normalized:
        return []
    words = normalized.split()
    if len(words) == 1 and any("\u4e00" <= char <= "\u9fff" for char in words[0]):
        return list(words[0])
    return words


def exact_match(reference: str, prediction: str) -> float:
    return float(_normalized_answer(reference) == _normalized_answer(prediction))


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = _answer_tokens(reference)
    prediction_tokens = _answer_tokens(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _int_map(payload: Mapping[str, Any]) -> dict[int, int]:
    return {int(key): int(value) for key, value in payload.items()}


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(percentile * len(ordered))) - 1))
    return float(ordered[index])


def _sample_condition_metrics(
    sample: Mapping[str, Any], condition: Mapping[str, Any]
) -> dict[str, float | None]:
    prediction = condition["prediction"]
    metrics: dict[str, float | None] = {
        "answer_exact_match": exact_match(
            str(sample["reference_answer"]), str(prediction["text"])
        ),
        "answer_token_f1": token_f1(
            str(sample["reference_answer"]), str(prediction["text"])
        ),
    }
    distribution = condition.get("distribution")
    if distribution is not None:
        count = int(distribution["token_count"])
        metrics.update(
            {
                "next_token_nll": float(distribution["target_nll_sum"]) / count,
                "kl_from_full_context": max(
                    0.0,
                    (
                        float(distribution["reference_cross_entropy_sum"])
                        - float(distribution["reference_entropy_sum"])
                    )
                    / count,
                ),
                "full_argmax_agreement": (
                    int(distribution["argmax_agreement_count"]) / count
                ),
                "teacher_forced_token_accuracy": (
                    int(distribution["target_accuracy_count"]) / count
                ),
            }
        )

    selection = condition.get("selection")
    if selection is not None:
        selected_records = int(selection["selected_record_count"])
        selected_evidence_records = int(selection["selected_evidence_record_count"])
        total_evidence_records = int(selection["total_cluster_evidence_record_count"])
        selected_evidence_tokens = int(selection["selected_evidence_token_count"])
        total_evidence_tokens = int(selection["total_cluster_evidence_token_count"])
        selected_blocks = set(selection["selected_evidence_block_ids"])
        total_blocks = set(selection["total_evidence_block_ids"])
        metrics.update(
            {
                "selected_cluster_count": float(selection["selected_cluster_count"]),
                "selected_record_count": float(selected_records),
                "selected_record_token_count": float(
                    selection["selected_record_token_count"]
                ),
                "mean_selected_cluster_record_count": float(
                    selection["mean_selected_cluster_record_count"]
                ),
                "evidence_record_recall": _ratio(
                    selected_evidence_records, total_evidence_records
                ),
                "evidence_token_recall": _ratio(
                    selected_evidence_tokens, total_evidence_tokens
                ),
                "evidence_block_recall": _ratio(
                    len(selected_blocks.intersection(total_blocks)), len(total_blocks)
                ),
                "any_evidence_hit": float(
                    selected_evidence_records > 0 or bool(selected_blocks)
                ),
                "all_evidence_hit": (
                    None
                    if total_evidence_records <= 0 and not total_blocks
                    else float(
                        (
                            total_evidence_records <= 0
                            or selected_evidence_records >= total_evidence_records
                        )
                        and (not total_blocks or total_blocks.issubset(selected_blocks))
                    )
                ),
                "retrieval_precision": _ratio(
                    selected_evidence_records, selected_records
                ),
                "cluster_amplification": _ratio(
                    selected_records, selected_evidence_records
                ),
            }
        )
        selected_attention = selection.get("selected_teacher_attention_mass")
        total_attention = selection.get("total_teacher_attention_mass")
        metrics["teacher_attention_mass_coverage"] = (
            None
            if selected_attention is None or total_attention is None
            else _ratio(float(selected_attention), float(total_attention))
        )

    resources = condition["resources"]
    historical = _int_map(resources["historical_tokens_by_layer"])
    local = _int_map(resources["local_tokens_by_layer"])
    full = _int_map(resources["full_history_tokens_by_layer"])
    historical_total = sum(historical.values())
    local_total = sum(local.values())
    full_total = sum(full.values())
    layer_ids = sorted(set(full) | set(historical))
    retrieved_by_layer = [historical.get(layer, 0) for layer in layer_ids]
    metrics.update(
        {
            "historical_layer_token_kv_ratio": _ratio(
                historical_total, full_total
            ),
            "visible_layer_token_kv_ratio": _ratio(
                historical_total + local_total, full_total + local_total
            ),
            "mean_retrieved_tokens_per_layer": (
                0.0 if not layer_ids else historical_total / len(layer_ids)
            ),
            "max_retrieved_tokens_in_any_layer": float(
                max(historical.values(), default=0)
            ),
            "p95_retrieved_tokens_per_layer": _nearest_rank_percentile(
                retrieved_by_layer, 0.95
            ),
            "active_layer_fraction": _ratio(
                sum(count > 0 for count in historical.values()), len(full)
            ),
            "kv_bytes_visible": float(resources["kv_bytes_visible"]),
            "cuda_peak_allocated_bytes": float(
                resources["cuda_peak_allocated_bytes"]
            ),
            "cuda_peak_reserved_bytes": float(resources["cuda_peak_reserved_bytes"]),
            "cuda_incremental_peak_allocated_bytes": float(
                resources["cuda_incremental_peak_allocated_bytes"]
            ),
            "cuda_incremental_peak_reserved_bytes": float(
                resources["cuda_incremental_peak_reserved_bytes"]
            ),
            "attention_query_key_pairs": float(
                resources["attention_query_key_pairs"]
            ),
            "latency_seconds": float(resources["latency_seconds"]),
        }
    )
    return metrics


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conditions = sorted(
        {name for row in rows for name in row.get("conditions", {})}
    )
    result: dict[str, Any] = {"sample_count": len(rows), "conditions": {}}
    for condition in conditions:
        condition_rows = [
            row["conditions"][condition]
            for row in rows
            if condition in row.get("conditions", {})
        ]
        names = sorted({name for metric in condition_rows for name in metric})
        result["conditions"][condition] = {
            name: _mean(metric.get(name) for metric in condition_rows)
            for name in names
        }
        result["conditions"][condition]["sample_count"] = len(condition_rows)

    full = result["conditions"].get("full_context")
    local = result["conditions"].get("local_only")
    if full is not None and local is not None:
        for name, values in result["conditions"].items():
            if name in {"full_context", "local_only"}:
                continue
            for score_name in ("answer_exact_match", "answer_token_f1"):
                denominator = full[score_name] - local[score_name]
                values[f"normalized_quality_recovery/{score_name}"] = (
                    None
                    if abs(denominator) <= 1e-12
                    else (values[score_name] - local[score_name]) / denominator
                )
    evidence_only = result["conditions"].get("evidence_only")
    if evidence_only is not None and local is not None:
        for name, values in result["conditions"].items():
            if name == "evidence_only":
                continue
            for score_name in ("answer_exact_match", "answer_token_f1"):
                values[f"quality_gap_to_evidence_only/{score_name}"] = (
                    evidence_only[score_name] - values[score_name]
                )
                denominator = evidence_only[score_name] - local[score_name]
                values[
                    f"normalized_quality_recovery_to_evidence_only/{score_name}"
                ] = (
                    None
                    if abs(denominator) <= 1e-12
                    else (values[score_name] - local[score_name]) / denominator
                )
    return result


def _bucket(value: int, boundaries: Sequence[int]) -> str:
    lower = 0
    for upper in boundaries:
        if value < upper:
            return f"{lower}-{upper - 1}"
        lower = upper
    return f"{lower}+"


def _bucket_summaries(
    rows: Sequence[Mapping[str, Any]], config: MetricConfig
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "evidence_distance_tokens": {},
        "sequence_length": {},
        "evidence_placement_bin": {},
    }
    for row in rows:
        sample = row["sample"]
        labels = {
            "evidence_distance_tokens": _bucket(
                int(sample["evidence_distance_tokens"]), config.distance_boundaries
            ),
            "sequence_length": _bucket(
                int(sample["sequence_length"]), config.length_boundaries
            ),
            "evidence_placement_bin": str(
                sample.get("metadata", {}).get("evidence_placement_bin", "unknown")
            ),
        }
        for group_name, label in labels.items():
            groups[group_name].setdefault(label, []).append(row)
    return {
        group_name: {
            label: _aggregate(group_rows)
            for label, group_rows in sorted(labels.items())
        }
        for group_name, labels in groups.items()
    }


def _pareto_summary(summary: Mapping[str, Any], config: MetricConfig) -> dict[str, Any]:
    conditions = summary["conditions"]
    points = []
    for name, metrics in conditions.items():
        if not name.startswith("learned_router@"):
            continue
        ratio = metrics.get("historical_layer_token_kv_ratio")
        score = metrics.get("answer_token_f1")
        if ratio is not None and score is not None:
            points.append(
                {
                    "condition": name,
                    "budget": int(name.rsplit("@", 1)[1]),
                    "historical_layer_token_kv_ratio": ratio,
                    "answer_token_f1": score,
                    "answer_exact_match": metrics.get("answer_exact_match"),
                }
            )
    points.sort(key=lambda row: (row["historical_layer_token_kv_ratio"], row["budget"]))
    full_score = conditions.get("full_context", {}).get("answer_token_f1")
    result: dict[str, Any] = {"learned_router_points": points}
    for threshold in config.kv_ratio_thresholds:
        eligible = [
            point
            for point in points
            if point["historical_layer_token_kv_ratio"] <= threshold
        ]
        result[f"best_answer_token_f1_at_kv_ratio<={threshold:g}"] = (
            None if not eligible else max(point["answer_token_f1"] for point in eligible)
        )
        exact_values = [
            point["answer_exact_match"]
            for point in eligible
            if point["answer_exact_match"] is not None
        ]
        result[f"best_answer_exact_match_at_kv_ratio<={threshold:g}"] = (
            None if not exact_values else max(exact_values)
        )
    result["minimum_kv_ratio_for_95pct_full_context_f1"] = None
    if full_score is not None:
        eligible = [point for point in points if point["answer_token_f1"] >= 0.95 * full_score]
        if eligible:
            result["minimum_kv_ratio_for_95pct_full_context_f1"] = min(
                point["historical_layer_token_kv_ratio"] for point in eligible
            )
    return result


def _load_states(state_dir: Path, *, allow_incomplete: bool) -> tuple[dict, list[dict]]:
    manifest_path = state_dir / "run_manifest.json"
    samples_path = state_dir / "samples.jsonl"
    if not manifest_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError("state directory needs run_manifest.json and samples.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported validation state schema")
    if not allow_incomplete and manifest.get("status") != "complete":
        raise ValueError(
            f"validation state is not cleanly complete: {manifest.get('status')!r}"
        )
    rows = []
    seen: set[str] = set()
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
                raise ValueError("sample state schema does not match runner")
            sample_id = str(row["sample"]["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample state: {sample_id}")
            seen.add(sample_id)
            if "full_context" not in row["conditions"] or "local_only" not in row["conditions"]:
                raise ValueError("sample state is missing full/local baselines")
            rows.append(row)
    if not rows:
        raise ValueError("validation state contains no completed samples")
    return manifest, rows


def evaluate_validation_states(
    state_dir: str | Path,
    output_dir: str | Path,
    config: MetricConfig | None = None,
) -> dict[str, Any]:
    """Compute all quality/retrieval/resource metrics from collected state."""

    config = config or MetricConfig()
    state_dir = Path(state_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, state_rows = _load_states(
        state_dir, allow_incomplete=config.allow_incomplete_state
    )
    sample_metrics: list[dict[str, Any]] = []
    for row in state_rows:
        sample = row["sample"]
        sample_metrics.append(
            {
                "sample": sample,
                "conditions": {
                    name: _sample_condition_metrics(sample, condition)
                    for name, condition in row["conditions"].items()
                },
            }
        )
    summary = _aggregate(sample_metrics)
    result = {
        "metric_schema_version": 1,
        "source_state_schema_version": STATE_SCHEMA_VERSION,
        "source_identity_sha256": manifest["identity_sha256"],
        "metric_config": asdict(config),
        "summary": summary,
        "breakdowns": _bucket_summaries(sample_metrics, config),
        "quality_memory_pareto": _pareto_summary(summary, config),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in sample_metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "condition_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        names = sorted(
            {
                metric
                for condition in summary["conditions"].values()
                for metric in condition
            }
        )
        writer = csv.DictWriter(handle, fieldnames=["condition", *names])
        writer.writeheader()
        for condition, values in sorted(summary["conditions"].items()):
            writer.writerow({"condition": condition, **values})
    return result
