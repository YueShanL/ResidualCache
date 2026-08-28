from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from cluster_router_bridge import LearnableRouterEncoder
from cluster_router_experiment.convomem import DynamicConvoMemDataset
from cluster_router_experiment.streaming import (
    EvictedStreamingBlock,
    RollingContextCollector,
)
from cluster_router_validation.contracts import EvaluationExample
from learnable_index.model_adapter import load_frozen_gemma
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from residual_cache.gpu_local_cluster_memory import (
    GpuLocalClusterMemory,
    GpuLocalClusterMemoryConfig,
)

from .metrics import (
    bcubed_metrics,
    cluster_size_metrics,
    permutation_baseline,
    retrieval_fact_metrics,
)


_GRID_FIELDS = {
    "alpha",
    "tau_new",
    "count_exponent",
    "concentration_prior_mass",
    "maximum_concentration",
    "candidate_capacity",
    "locality_bits",
    "locality_probe_radius",
    "router_count_exponent",
    "router_concentration_prior_mass",
    "router_maximum_concentration",
}


@dataclass(frozen=True)
class CachedBlock:
    block_id: str
    logical_positions: tuple[int, ...]
    router_key: torch.Tensor
    router_block_size: int
    layer_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class FixedCache:
    sample_id: str
    target_fact_id: str
    label_by_position: Mapping[int, str]
    query_router_key: torch.Tensor
    blocks: tuple[CachedBlock, ...]
    forward_calls: int
    forwarded_tokens: int
    maximum_forward_context_length: int


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _stable_seed(*values: object) -> int:
    payload = ":".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _variant_id(index: int, values: Mapping[str, Any]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return f"v{index:03d}-{digest}"


def expand_parameter_grid(
    base: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    *,
    maximum_variants: int = 64,
) -> tuple[dict[str, Any], ...]:
    unknown = set(grid).difference(_GRID_FIELDS)
    if unknown:
        raise ValueError(f"unsupported clustering grid fields: {sorted(unknown)}")
    names = tuple(sorted(grid))
    axes: list[tuple[Any, ...]] = []
    for name in names:
        values = tuple(grid[name])
        if not values:
            raise ValueError(f"parameter grid axis {name!r} is empty")
        axes.append(values)
    combinations = list(itertools.product(*axes)) if axes else [()]
    if len(combinations) > int(maximum_variants):
        raise ValueError(
            f"parameter grid has {len(combinations)} variants; maximum is {maximum_variants}"
        )
    variants: list[dict[str, Any]] = []
    for index, combination in enumerate(combinations):
        values = dict(base)
        values.update(dict(zip(names, combination)))
        values["memory_budget_bytes"] = None
        values["eviction_enabled"] = False
        config = GpuLocalClusterMemoryConfig(**values)
        resolved = asdict(config)
        variants.append(
            {
                "variant_id": _variant_id(index, resolved),
                "grid_values": dict(zip(names, combination)),
                "memory_config": resolved,
            }
        )
    return tuple(variants)


def _memory_layers(bundle) -> tuple[int, ...]:
    config = bundle.text_config
    first_shared = int(config.num_hidden_layers) - int(
        getattr(config, "num_kv_shared_layers", 0)
    )
    result = tuple(
        index
        for index, layer_type in enumerate(config.layer_types[:first_shared])
        if layer_type == "full_attention"
    )
    if not result:
        raise ValueError("Gemma 4 exposes no physical full-attention memory layers")
    return result


def _fact_labels(row: Mapping[str, Any], positions: Sequence[int]) -> dict[int, str]:
    allowed = set(int(value) for value in positions)
    labels: dict[int, str] = {}
    for item in row.get("memory_fact_token_ranges", ()):
        fact_id = str(item["fact_id"])
        start = int(item["start"])
        end = int(item["end"])
        if start >= end:
            raise ValueError("memory fact range must be non-empty")
        for position in range(start, end):
            if position not in allowed:
                continue
            previous = labels.setdefault(position, fact_id)
            if previous != fact_id:
                raise ValueError("overlapping memory fact ranges have different labels")
    if len(set(labels.values())) < 2:
        raise ValueError("cluster calibration requires at least two labeled facts")
    return labels


def _capture_fixed_cache(
    *,
    bundle,
    router: LearnableRouterEncoder,
    example: EvaluationExample,
    memory_layers: Sequence[int],
    local_context_length: int,
    block_size: int,
    residual_layer: int,
    query_summary_length: int,
) -> FixedCache:
    if not isinstance(example.payload, Mapping):
        raise TypeError("ConvoMem calibration requires the original row payload")
    row = dict(example.payload)
    record = SequenceRecord(
        sequence_id=example.sample_id,
        token_ids=tuple(int(value) for value in row["token_ids"]),
        metadata={key: value for key, value in row.items() if key != "token_ids"},
    )
    plans = build_retrieval_plans(
        record,
        PlanConfig(
            local_context_length=int(local_context_length),
            block_size=int(block_size),
            future_horizon_length=max(1, len(example.reference_token_ids)),
            retrieval_interval=int(block_size),
            minimum_candidate_blocks=1,
            retrieval_point_policy="metadata",
        ),
    )
    if len(plans) != 1:
        raise ValueError(f"expected one answer retrieval point, found {len(plans)}")
    plan = plans[0]
    block_keys: dict[object, torch.Tensor] = {}
    blocks: list[CachedBlock] = []

    def on_block_ready(block, residual_summary: torch.Tensor) -> None:
        block_keys[block.block_id] = router.encode_block_tensor(residual_summary).detach()

    def on_evict(block: EvictedStreamingBlock) -> None:
        try:
            router_key = block_keys[block.block.block_id]
        except KeyError as error:
            raise RuntimeError("evicted block has no prepared router key") from error
        # Clone only the evicted slice. Retaining a view would pin the complete
        # rolling cache once per block and invalidate the fixed-cache memory bound.
        layer_kv = {
            int(layer): (
                block.layer_kv[int(layer)][0].clone(),
                block.layer_kv[int(layer)][1].clone(),
            )
            for layer in memory_layers
        }
        blocks.append(
            CachedBlock(
                block_id=str(block.block.block_id),
                logical_positions=tuple(int(value) for value in block.logical_positions),
                router_key=router_key.clone(),
                router_block_size=int(block.block.length),
                layer_kv=layer_kv,
            )
        )

    collector = RollingContextCollector(
        bundle,
        local_context_length=int(local_context_length),
        block_size=int(block_size),
        residual_layer=int(residual_layer),
        query_summary_length=int(query_summary_length),
    )
    result = collector.collect(
        record,
        plan,
        on_block_ready=on_block_ready,
        on_evict=on_evict,
    )
    all_positions = tuple(
        position for block in blocks for position in block.logical_positions
    )
    if len(all_positions) != len(set(all_positions)):
        raise RuntimeError("fixed cache contains duplicate logical positions")
    labels = _fact_labels(row, all_positions)
    target_fact_id = str(row["target_example_id"])
    if target_fact_id not in set(labels.values()):
        raise ValueError("target fact has no token outside the retained local context")
    return FixedCache(
        sample_id=example.sample_id,
        target_fact_id=target_fact_id,
        label_by_position=labels,
        query_router_key=router.encode_query_tensor(result.query_summary).detach(),
        blocks=tuple(blocks),
        forward_calls=int(result.forward_calls),
        forwarded_tokens=int(result.forwarded_tokens),
        maximum_forward_context_length=int(result.maximum_forward_context_length),
    )


def _evaluate_variant(
    cache: FixedCache,
    variant: Mapping[str, Any],
    *,
    top_ns: Sequence[int],
    permutation_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    config = GpuLocalClusterMemoryConfig(**dict(variant["memory_config"]))
    memories: dict[int, GpuLocalClusterMemory] = {}
    start = time.perf_counter()
    for block in cache.blocks:
        for layer, (key, value) in block.layer_kv.items():
            memory = memories.get(int(layer))
            if memory is None:
                memory = GpuLocalClusterMemory(
                    kv_heads=int(key.shape[1]),
                    head_dim=int(key.shape[3]),
                    router_dim=int(block.router_key.numel()),
                    device=key.device,
                    dtype=key.dtype,
                    config=config,
                )
                memories[int(layer)] = memory
            memory.ingest_block(
                key,
                value,
                router_key=block.router_key,
                block_id=block.block_id,
                logical_positions=block.logical_positions,
                router_block_size=block.router_block_size,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ingestion_seconds = time.perf_counter() - start
    rows: list[dict[str, Any]] = []
    for layer, memory in sorted(memories.items()):
        ranked = memory.router_clusters(cache.query_router_key)
        memberships = [tuple(int(value) for value in row.logical_positions) for row in ranked]
        cluster_by_record = {
            int(position): str(row.cluster_id)
            for row in ranked
            for position in row.logical_positions
        }
        fact = bcubed_metrics(cluster_by_record, cache.label_by_position)
        permutation = permutation_baseline(
            cluster_by_record,
            cache.label_by_position,
            trials=int(permutation_trials),
            seed=_stable_seed(seed, cache.sample_id, variant["variant_id"], layer),
        )
        fact["bcubed_f1_gain_over_permutation"] = (
            float(fact["bcubed_f1"]) - float(permutation["mean_bcubed_f1"])
        )
        fact["beats_permutation_p95"] = float(
            float(fact["bcubed_f1"]) > float(permutation["p95_bcubed_f1"])
        )
        retrieval = {
            f"top_{int(top_n)}": retrieval_fact_metrics(
                memberships,
                cache.label_by_position,
                target_fact_id=cache.target_fact_id,
                top_n=int(top_n),
            )
            for top_n in top_ns
        }
        snapshot = memory.snapshot()
        candidate_requests = int(snapshot["local_candidate_requests"])
        rows.append(
            {
                "sample_id": cache.sample_id,
                "variant_id": str(variant["variant_id"]),
                "grid_values": dict(variant["grid_values"]),
                "layer": int(layer),
                "structure": cluster_size_metrics(memberships),
                "fact_separation": fact,
                "permutation_baseline": permutation,
                "retrieval": retrieval,
                "assignment": {
                    "created_cluster_rate": (
                        float(snapshot["created_slots"])
                        / float(snapshot["ingested_records"])
                        if snapshot["ingested_records"]
                        else 0.0
                    ),
                    "existing_assignment_rate": (
                        float(snapshot["assigned_existing"])
                        / float(snapshot["ingested_records"])
                        if snapshot["ingested_records"]
                        else 0.0
                    ),
                    "mean_local_candidates": (
                        float(snapshot["local_candidate_slots_considered"])
                        / candidate_requests
                        if candidate_requests
                        else 0.0
                    ),
                    "maximum_local_candidates": float(
                        snapshot["maximum_candidate_slots_considered"]
                    ),
                },
                "ingestion_seconds_all_layers": ingestion_seconds,
                "memory_snapshot": snapshot,
            }
        )
    return rows


def _aggregate_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_ns: Sequence[int],
) -> dict[str, Any]:
    def values(section: str, metric: str) -> list[float]:
        return [float(row[section][metric]) for row in rows]

    structure_names = tuple(rows[0]["structure"])
    fact_names = tuple(rows[0]["fact_separation"])
    assignment_names = tuple(rows[0]["assignment"])
    result = {
        "sample_count": len({str(row["sample_id"]) for row in rows}),
        "layer_count": len({int(row["layer"]) for row in rows}),
        "condition_count": len(rows),
        "structure": {
            name: _mean(values("structure", name)) for name in structure_names
        },
        "fact_separation": {
            name: _mean(values("fact_separation", name)) for name in fact_names
        },
        "assignment": {
            name: _mean(values("assignment", name)) for name in assignment_names
        },
        "retrieval": {},
    }
    result["fact_separation"]["consistent_condition_fraction"] = _mean(
        values("fact_separation", "beats_permutation_p95")
    )
    for top_n in top_ns:
        key = f"top_{int(top_n)}"
        metric_names = tuple(rows[0]["retrieval"][key])
        result["retrieval"][key] = {
            name: _mean([float(row["retrieval"][key][name]) for row in rows])
            for name in metric_names
        }
    by_layer: dict[str, Any] = {}
    for layer in sorted({int(row["layer"]) for row in rows}):
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        by_layer[str(layer)] = {
            "bcubed_f1": _mean(values_from(layer_rows, "fact_separation", "bcubed_f1")),
            "bcubed_f1_gain_over_permutation": _mean(
                values_from(
                    layer_rows,
                    "fact_separation",
                    "bcubed_f1_gain_over_permutation",
                )
            ),
            "mean_cluster_size": _mean(
                values_from(layer_rows, "structure", "mean_cluster_size")
            ),
            "singleton_cluster_ratio": _mean(
                values_from(layer_rows, "structure", "singleton_cluster_ratio")
            ),
        }
    result["by_layer"] = by_layer
    return result


def values_from(
    rows: Sequence[Mapping[str, Any]], section: str, metric: str
) -> list[float]:
    return [float(row[section][metric]) for row in rows]


def _select_variant(
    summaries: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    min_b3 = float(criteria.get("minimum_mean_bcubed_f1", 0.20))
    min_gain = float(criteria.get("minimum_mean_b3_gain_over_permutation", 0.05))
    min_consistency = float(
        criteria.get("minimum_consistent_condition_fraction", 0.75)
    )
    max_singleton = float(criteria.get("maximum_singleton_cluster_ratio", 0.60))
    min_cluster_size = float(criteria.get("minimum_mean_cluster_size", 2.0))
    max_p95 = float(criteria.get("maximum_mean_p95_cluster_size", 256.0))
    min_samples = int(criteria.get("minimum_samples_for_structural_conclusion", 4))
    eligible: list[Mapping[str, Any]] = []
    for row in summaries:
        fact = row["summary"]["fact_separation"]
        structure = row["summary"]["structure"]
        if (
            float(fact["bcubed_f1"]) >= min_b3
            and float(fact["bcubed_f1_gain_over_permutation"]) >= min_gain
            and float(fact["consistent_condition_fraction"]) >= min_consistency
            and float(structure["singleton_cluster_ratio"]) <= max_singleton
            and float(structure["mean_cluster_size"]) >= min_cluster_size
            and float(structure["p95_cluster_size"]) <= max_p95
        ):
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(row["summary"]["fact_separation"]["bcubed_f1"]),
            float(row["summary"]["structure"]["singleton_cluster_ratio"]),
            str(row["variant_id"]),
        )
    )
    best = max(
        summaries,
        key=lambda row: (
            float(row["summary"]["fact_separation"]["bcubed_f1_gain_over_permutation"]),
            float(row["summary"]["fact_separation"]["consistent_condition_fraction"]),
        ),
    )
    sample_count = max(int(row["summary"]["sample_count"]) for row in summaries)
    if eligible:
        status = "calibrated"
    elif sample_count < min_samples:
        status = "insufficient_conditions"
    elif (
        float(best["summary"]["fact_separation"]["bcubed_f1"]) < min_b3
        or float(
            best["summary"]["fact_separation"][
                "bcubed_f1_gain_over_permutation"
            ]
        ) < min_gain
        or float(
            best["summary"]["fact_separation"]["consistent_condition_fraction"]
        )
        < min_consistency
    ):
        status = "possible_structural_limit"
    else:
        status = "grid_did_not_meet_allocation_constraints"
    selected = eligible[0] if eligible else None
    return {
        "status": status,
        "criteria": {
            "minimum_mean_bcubed_f1": min_b3,
            "minimum_mean_b3_gain_over_permutation": min_gain,
            "minimum_consistent_condition_fraction": min_consistency,
            "maximum_singleton_cluster_ratio": max_singleton,
            "minimum_mean_cluster_size": min_cluster_size,
            "maximum_mean_p95_cluster_size": max_p95,
            "minimum_samples_for_structural_conclusion": min_samples,
        },
        "selected_variant_id": None if selected is None else selected["variant_id"],
        "selected_grid_values": None if selected is None else selected["grid_values"],
        "best_observed_variant_id": best["variant_id"],
        "best_observed_grid_values": best["grid_values"],
        "best_observed_bcubed_f1": float(
            best["summary"]["fact_separation"]["bcubed_f1"]
        ),
        "best_observed_b3_gain_over_permutation": float(
            best["summary"]["fact_separation"]["bcubed_f1_gain_over_permutation"]
        ),
        "best_observed_consistent_condition_fraction": float(
            best["summary"]["fact_separation"]["consistent_condition_fraction"]
        ),
    }


def run_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    model_config = dict(config["model"])
    checkpoint = Path(str(model_config.pop("checkpoint_path"))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    block_size = int(model_config.pop("block_size", 64))
    local_context_length = int(model_config.pop("local_context_length", 512))
    residual_layer = int(model_config.pop("residual_layer", 40))
    query_summary_length = int(model_config.pop("query_summary_length", 16))
    router_device = str(model_config.pop("router_device", model_config.get("device", "cuda")))
    model_cache_dir = model_config.pop("model_cache_dir", None)
    if model_cache_dir is not None:
        if "cache_dir" in model_config:
            raise ValueError("model may define only one of model_cache_dir and cache_dir")
        model_config["cache_dir"] = model_cache_dir
    bundle = load_frozen_gemma(**model_config)
    router = LearnableRouterEncoder.from_checkpoint(checkpoint, device=router_device)
    layers = _memory_layers(bundle)
    dataset = DynamicConvoMemDataset(**dict(config["data"]))
    if dataset.block_size != block_size:
        raise ValueError("data.block_size must equal model.block_size")
    if dataset.local_context_length != local_context_length:
        raise ValueError(
            "data.local_context_length must equal model.local_context_length"
        )
    sweep = dict(config["sweep"])
    variants = expand_parameter_grid(
        dict(sweep.get("base_memory_config", {})),
        dict(sweep.get("grid", {})),
        maximum_variants=int(sweep.get("maximum_variants", 64)),
    )
    evaluation = dict(config.get("evaluation", {}))
    top_ns = tuple(sorted(set(int(value) for value in evaluation.get("top_ns", [4]))))
    if not top_ns or min(top_ns) <= 0:
        raise ValueError("evaluation.top_ns must contain positive values")
    permutation_trials = int(evaluation.get("permutation_trials", 32))
    if permutation_trials <= 0:
        raise ValueError("evaluation.permutation_trials must be positive")
    seed = int(config.get("seed", 13))
    all_rows: list[dict[str, Any]] = []
    cache_captures: list[dict[str, Any]] = []
    sample_path = output_dir / "sample_metrics.jsonl"
    with sample_path.open("w", encoding="utf-8") as stream:
        for sample_index, example in enumerate(dataset):
            print(
                f"[{example.sample_id}] capture fixed cache once "
                f"({sample_index + 1}/{dataset.sequence_count})",
                flush=True,
            )
            capture_started = time.perf_counter()
            cache = _capture_fixed_cache(
                bundle=bundle,
                router=router,
                example=example,
                memory_layers=layers,
                local_context_length=local_context_length,
                block_size=block_size,
                residual_layer=residual_layer,
                query_summary_length=query_summary_length,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cache_tensor_bytes = sum(
                int(block.router_key.numel() * block.router_key.element_size())
                + sum(
                    int(key.numel() * key.element_size())
                    + int(value.numel() * value.element_size())
                    for key, value in block.layer_kv.values()
                )
                for block in cache.blocks
            ) + int(
                cache.query_router_key.numel()
                * cache.query_router_key.element_size()
            )
            cache_captures.append(
                {
                    "sample_id": cache.sample_id,
                    "capture_count": 1,
                    "reused_by_variant_count": len(variants),
                    "block_count": len(cache.blocks),
                    "record_count_per_layer": sum(
                        len(block.logical_positions) for block in cache.blocks
                    ),
                    "labeled_record_count": len(cache.label_by_position),
                    "fact_count": len(set(cache.label_by_position.values())),
                    "forward_calls": cache.forward_calls,
                    "forwarded_tokens": cache.forwarded_tokens,
                    "maximum_forward_context_length": (
                        cache.maximum_forward_context_length
                    ),
                    "resident_tensor_bytes": cache_tensor_bytes,
                    "capture_seconds": time.perf_counter() - capture_started,
                }
            )
            for variant_index, variant in enumerate(variants):
                rows = _evaluate_variant(
                    cache,
                    variant,
                    top_ns=top_ns,
                    permutation_trials=permutation_trials,
                    seed=seed,
                )
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                all_rows.extend(rows)
                mean_gain = _mean(
                    [
                        float(row["fact_separation"]["bcubed_f1_gain_over_permutation"])
                        for row in rows
                    ]
                )
                print(
                    f"[{example.sample_id}] variant {variant_index + 1}/{len(variants)} "
                    f"{variant['variant_id']} mean_b3_gain={mean_gain:.4f}",
                    flush=True,
                )
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in all_rows if row["variant_id"] == variant["variant_id"]]
        summaries.append(
            {
                "variant_id": variant["variant_id"],
                "grid_values": dict(variant["grid_values"]),
                "memory_config": dict(variant["memory_config"]),
                "summary": _aggregate_variant(rows, top_ns=top_ns),
            }
        )
    selection = _select_variant(summaries, dict(config.get("criteria", {})))
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "cache_capture": (
                f"single_pass_rolling_{local_context_length}_to_"
                f"{local_context_length + block_size}"
            ),
            "cache_reuse": "one_fixed_gpu_kv_capture_per_sample_for_all_variants",
            "variant_initial_state": "empty_independent_memory",
            "record_order": "identical_logical_position_order_for_every_variant",
            "parameter_policy": "static_per_variant_no_online_tuning",
            "eviction": "disabled",
            "fact_labels_visible_to_memory": False,
            "raw_cache_persisted": False,
        },
        "model": {
            "fingerprint": bundle.fingerprint,
            "checkpoint": str(checkpoint),
            "memory_layers": list(layers),
            "block_size": block_size,
            "local_context_length": local_context_length,
            "residual_layer": residual_layer,
            "query_summary_length": query_summary_length,
        },
        "dataset": dataset.descriptor,
        "cache_captures": cache_captures,
        "variant_count": len(variants),
        "variants": summaries,
        "selection": selection,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)
    return result
