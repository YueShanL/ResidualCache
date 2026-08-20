from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable

import torch

from .aligned_builder import load_collected_sequences
from .collectors import RestrictedStudentCollector, StudentCollectionConfig
from .data import load_dataset
from .kv_store import KVBlockStore, merge_layer_kv
from .replay import (
    _condition_metrics,
    _full_context_logits,
    _kv_bytes,
    _restricted_logits,
)
from .retrieval import oracle_indices, recent_indices, score_retrieval_sample
from .trainer import load_checkpoint, resolve_device


@dataclass(frozen=True)
class TopNSweepConfig:
    budgets: tuple[int, ...] = (1, 2, 4, 8)
    maximum_samples: int | None = None
    router_device: str = "cpu"
    random_seed: int = 13
    bootstrap_iterations: int = 2000
    verify_query_summary: bool = True
    query_verification_atol: float = 1e-4

    def __post_init__(self) -> None:
        if not self.budgets or any(budget <= 0 for budget in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be sorted and unique")
        if self.maximum_samples is not None and self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        if self.query_verification_atol <= 0:
            raise ValueError("query_verification_atol must be positive")


def _random_indices(sample, top_n: int, seed: int) -> tuple[int, ...]:
    budget = min(top_n, len(sample.candidate_blocks))
    digest = hashlib.sha256(
        f"{seed}:{sample.sample_id}:{top_n}".encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(generator.sample(range(len(sample.candidate_blocks)), budget)))


def _oldest_indices(sample, top_n: int) -> tuple[int, ...]:
    budget = min(top_n, len(sample.candidate_blocks))
    return tuple(
        sorted(
            range(len(sample.candidate_blocks)),
            key=lambda index: sample.candidate_blocks[index].start_position,
        )[:budget]
    )


def _cuda_devices(bundle) -> tuple[torch.device, ...]:
    devices = {device for device in bundle.cache_layer_devices if device.type == "cuda"}
    if bundle.input_device.type == "cuda":
        devices.add(bundle.input_device)
    return tuple(sorted(devices, key=str))


def _measure_cuda_peak(bundle, operation: Callable[[], Any]) -> tuple[Any, dict[str, int]]:
    devices = _cuda_devices(bundle)
    if not devices:
        return operation(), {
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
            "cuda_incremental_peak_allocated_bytes": 0,
            "cuda_incremental_peak_reserved_bytes": 0,
        }
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    allocated_before = sum(torch.cuda.memory_allocated(device) for device in devices)
    reserved_before = sum(torch.cuda.memory_reserved(device) for device in devices)
    result = operation()
    for device in devices:
        torch.cuda.synchronize(device)
    peak_allocated = sum(torch.cuda.max_memory_allocated(device) for device in devices)
    peak_reserved = sum(torch.cuda.max_memory_reserved(device) for device in devices)
    return result, {
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_allocated_bytes": max(0, peak_allocated - allocated_before),
        "cuda_incremental_peak_reserved_bytes": max(0, peak_reserved - reserved_before),
    }


def _mean_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({name for row in rows for name in row["conditions"]})
    summary: dict[str, Any] = {"sample_count": len(rows), "conditions": {}}
    for condition in conditions:
        metrics = [row["conditions"][condition] for row in rows]
        numeric_names = sorted(
            {
                name
                for metric in metrics
                for name, value in metric.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        summary["conditions"][condition] = {
            name: sum(float(metric[name]) for metric in metrics) / len(metrics)
            for name in numeric_names
        }
    return summary


def _bootstrap_interval(
    values: list[float], *, iterations: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )
    return means[int(0.025 * iterations)], means[min(iterations - 1, int(0.975 * iterations))]


def paired_comparisons(
    rows: list[dict[str, Any]], *, budgets: Iterable[int], iterations: int, seed: int
) -> dict[str, Any]:
    directions = {
        "next_token_nll": "lower",
        "kl_from_full_context": "lower",
        "full_argmax_agreement": "higher",
        "teacher_forced_token_accuracy": "higher",
    }
    result: dict[str, Any] = {}
    for budget in budgets:
        learned = f"predicted_top_{budget}"
        for baseline_name in ("recent", "oldest", "random", "oracle"):
            baseline = f"{baseline_name}_top_{budget}"
            comparison_name = f"{learned}_vs_{baseline}"
            result[comparison_name] = {}
            for metric_name, direction in directions.items():
                differences = [
                    float(row["conditions"][learned][metric_name])
                    - float(row["conditions"][baseline][metric_name])
                    for row in rows
                ]
                lower, upper = _bootstrap_interval(
                    differences,
                    iterations=iterations,
                    seed=seed + budget * 1009 + len(result) * 37 + len(metric_name),
                )
                wins = sum(
                    difference < 0 if direction == "lower" else difference > 0
                    for difference in differences
                )
                ties = sum(difference == 0 for difference in differences)
                result[comparison_name][metric_name] = {
                    "direction": direction,
                    "mean_learned_minus_baseline": sum(differences) / len(differences),
                    "bootstrap_95_ci": [lower, upper],
                    "learned_win_rate": wins / len(differences),
                    "tie_rate": ties / len(differences),
                }
    return result


def evaluate_topn_sweep(
    bundle,
    collection_dir: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    config: TopNSweepConfig,
) -> dict[str, Any]:
    collection_dir = Path(collection_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, dataset_manifest = load_dataset(collection_dir / "dataset")
    if dataset_manifest["metadata"]["model_fingerprint"] != bundle.fingerprint:
        raise ValueError("collection and sweep model fingerprints do not match")
    records = {
        record.sequence_id: record for record in load_collected_sequences(collection_dir)
    }
    store = KVBlockStore(collection_dir / "kv_store")
    collection_manifest = json.loads(
        (collection_dir / "collection_manifest.json").read_text(encoding="utf-8")
    )
    teacher_prefill_chunk_size = collection_manifest["collection_config"].get(
        "teacher_prefill_chunk_size"
    )
    student = RestrictedStudentCollector(
        bundle,
        StudentCollectionConfig(
            **collection_manifest["collection_config"]["student"]
        ),
    )
    router, _, _, _, checkpoint = load_checkpoint(checkpoint_path)
    router_device = resolve_device(config.router_device)
    samples = dataset.samples[: config.maximum_samples] if config.maximum_samples else dataset.samples
    samples_path = output_dir / "samples.jsonl"
    state_path = output_dir / "sweep_state.json"
    expected_state = json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": bundle.fingerprint,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "checkpoint_epoch": checkpoint["epoch"],
                "sweep_config": asdict(config),
                "sample_ids": [sample.sample_id for sample in samples],
            },
            ensure_ascii=False,
        )
    )
    if state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state != expected_state:
            raise ValueError(
                "existing sweep_state.json does not match the requested evaluation"
            )
    else:
        temporary_state_path = state_path.with_suffix(".json.tmp")
        temporary_state_path.write_text(
            json.dumps(expected_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_state_path.replace(state_path)

    existing_rows: dict[str, dict[str, Any]] = {}
    if samples_path.exists():
        with samples_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row["sample_id"]
                if sample_id in existing_rows:
                    raise ValueError(
                        f"duplicate sample_id {sample_id!r} at {samples_path}:{line_number}"
                    )
                existing_rows[sample_id] = row
    expected_sample_ids = set(expected_state["sample_ids"])
    unexpected_sample_ids = set(existing_rows) - expected_sample_ids
    if unexpected_sample_ids:
        raise ValueError(
            f"samples.jsonl contains unexpected sample IDs: {sorted(unexpected_sample_ids)}"
        )
    rows: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, start=1):
        if sample.sample_id in existing_rows:
            rows.append(existing_rows[sample.sample_id])
            continue
        started = time.perf_counter()
        record = records[sample.sequence_id]
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
        local_kv_bytes = _kv_bytes(local_layer_kv)
        full_end = (
            sample.first_future_position_affected_by_retrieval
            + sample.future_horizon_length
        )
        (full_result, full_memory) = _measure_cuda_peak(
            bundle,
            lambda: _full_context_logits(
                bundle,
                record,
                sample,
                prefill_chunk_size=teacher_prefill_chunk_size,
            ),
        )
        full_logits, full_attention_pairs = full_result
        targets = torch.tensor(
            record.token_ids[
                sample.first_future_position_affected_by_retrieval
                + 1 : sample.first_future_position_affected_by_retrieval
                + sample.future_horizon_length
                + 1
            ],
            dtype=torch.long,
        )
        scores = score_retrieval_sample(router, sample, device=router_device)
        conditions: dict[str, dict[str, Any]] = {}
        full_metrics = _condition_metrics(full_logits, full_logits, targets)
        full_metrics.update(full_memory)
        bytes_per_local_token = local_kv_bytes / (
            sample.local_context_end - sample.local_context_start
        )
        full_metrics.update(
            {
                "selected_block_count": 0,
                "retrieved_kv_bytes": 0,
                "local_kv_bytes": 0,
                "total_visible_kv_bytes": int(bytes_per_local_token * full_end),
                "attention_query_key_pairs_per_layer": full_attention_pairs,
            }
        )
        conditions["full_context"] = full_metrics

        replay_cache: dict[tuple[str, ...], tuple[dict[str, Any], int]] = {}

        def restricted_for(block_ids: tuple[str, ...]) -> tuple[dict[str, Any], int]:
            if block_ids in replay_cache:
                return replay_cache[block_ids]
            blocks = store.load_many(block_ids)
            if any(block.model_fingerprint != bundle.fingerprint for block in blocks):
                raise RuntimeError("sweep KV payload model fingerprint mismatch")
            prefix = merge_layer_kv(blocks, device="cpu")
            (restricted_result, memory) = _measure_cuda_peak(
                bundle,
                lambda: _restricted_logits(
                    bundle, record, sample, local_layer_kv, prefix
                ),
            )
            logits, retrieved_bytes, attention_pairs = restricted_result
            metrics = _condition_metrics(logits, full_logits, targets)
            metrics.update(memory)
            metrics.update(
                {
                    "selected_block_count": len(block_ids),
                    "retrieved_kv_bytes": retrieved_bytes,
                    "local_kv_bytes": local_kv_bytes,
                    "total_visible_kv_bytes": local_kv_bytes + retrieved_bytes,
                    "attention_query_key_pairs_per_layer": attention_pairs,
                }
            )
            replay_cache[block_ids] = (metrics, attention_pairs)
            return replay_cache[block_ids]

        local_metrics, _ = restricted_for(())
        conditions["local_256"] = dict(local_metrics)
        selected: dict[str, list[str]] = {}
        learned_order = tuple(int(index) for index in scores.argsort(descending=True).cpu())
        for budget in config.budgets:
            methods = {
                "predicted": learned_order[: min(budget, len(learned_order))],
                "recent": recent_indices(sample, budget),
                "oldest": _oldest_indices(sample, budget),
                "random": _random_indices(sample, budget, config.random_seed),
                "oracle": oracle_indices(sample, budget),
            }
            for method, indices in methods.items():
                block_ids = tuple(
                    sample.candidate_blocks[index].block_id for index in indices
                )
                name = f"{method}_top_{budget}"
                selected[name] = list(block_ids)
                metrics, _ = restricted_for(block_ids)
                conditions[name] = dict(metrics)

        row = {
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "retrieval_position": sample.retrieval_position,
            "evidence_token_ranges": record.metadata.get("evidence_token_ranges"),
            "evidence_placement": record.metadata.get("evidence_placement"),
            "evidence_placement_bin": record.metadata.get("evidence_placement_bin"),
            "evidence_block_indices": record.metadata.get("evidence_block_indices"),
            "selected_block_ids": selected,
            "target_token_ids": targets.tolist(),
            "conditions": conditions,
            "wall_time_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        with samples_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            print(
                json.dumps(
                    {
                        "event": "topn_sweep_progress",
                        "completed": sample_index,
                        "total": len(samples),
                        "sample_id": sample.sample_id,
                        "wall_time_seconds": row["wall_time_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except (BrokenPipeError, OSError):
            # An HPC launcher or API wrapper may close stdout before the child
            # process exits. Durable results must not depend on the log pipe.
            pass
    summary = _mean_summary(rows)
    summary.update(
        {
            "schema_version": 1,
            "model_fingerprint": bundle.fingerprint,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "sweep_config": asdict(config),
            "paired_comparisons": paired_comparisons(
                rows,
                budgets=config.budgets,
                iterations=config.bootstrap_iterations,
                seed=config.random_seed,
            ),
            "memory_semantics": {
                "total_visible_kv_bytes": (
                    "exact local plus retrieved KV payload bytes for restricted conditions; "
                    "full-context value is the equivalent cache bytes through the horizon"
                ),
                "cuda_incremental_peak": (
                    "peak above allocated/reserved bytes immediately before each condition"
                ),
            },
        }
    )
    with (output_dir / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
