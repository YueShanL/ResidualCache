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
from cluster_router_experiment.block_streaming import (
    BlockAlignedRollingContextCollector,
    EvictedCompleteBlock,
)
from cluster_router_experiment.convomem import DynamicConvoMemDataset
from cluster_router_validation.contracts import EvaluationExample
from learnable_index.model_adapter import load_frozen_gemma
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from memory_cluster_calibration.metrics import (
    bcubed_metrics,
    cluster_size_metrics,
    permutation_baseline,
)
from residual_cache.gpu_block_cluster_memory import (
    GpuBlockClusterMemory,
    GpuBlockClusterMemoryConfig,
)

from .metrics import build_block_fact_labels, target_block_retrieval_metrics


_GRID_FIELDS = {
    "alpha",
    "tau_new",
    "count_exponent",
    "concentration_prior_mass",
    "maximum_concentration",
    "candidate_capacity",
    "locality_bits",
    "locality_probe_radius",
}


@dataclass(frozen=True)
class CachedBlock:
    block_id: str
    logical_positions: tuple[int, ...]
    router_key: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class FixedBlockCache:
    sample_id: str
    target_fact_id: str
    primary_label_by_block: Mapping[str, str]
    target_block_ids: tuple[str, ...]
    label_diagnostics: Mapping[str, float]
    query_router_key: torch.Tensor
    blocks: tuple[CachedBlock, ...]
    payload_layer: int
    equivalent_memory_layers: tuple[int, ...]
    forward_calls: int
    forwarded_tokens: int
    maximum_forward_context_length: int
    final_local_context_length: int


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
    block_size: int,
    maximum_variants: int = 64,
) -> tuple[dict[str, Any], ...]:
    unknown = set(grid).difference(_GRID_FIELDS)
    if unknown:
        raise ValueError(f"unsupported block-memory grid fields: {sorted(unknown)}")
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
            f"parameter grid has {len(combinations)} variants; maximum is "
            f"{maximum_variants}"
        )
    variants: list[dict[str, Any]] = []
    for index, combination in enumerate(combinations):
        values = dict(base)
        values.update(dict(zip(names, combination)))
        configured_block_size = int(values.get("block_size", block_size))
        if configured_block_size != int(block_size):
            raise ValueError("base memory block_size must equal model.block_size")
        values["block_size"] = int(block_size)
        values["memory_budget_bytes"] = None
        values["eviction_enabled"] = False
        resolved = asdict(GpuBlockClusterMemoryConfig(**values))
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


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"output_dir", "data", "model", "sweep"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"calibration config is missing fields: {sorted(missing)}")
    data = dict(config["data"])
    model = dict(config["model"])
    sweep = dict(config["sweep"])
    block_size = int(model.get("block_size", 64))
    local_context = int(model.get("local_context_length", 512))
    query_length = int(model.get("query_summary_length", 16))
    if min(block_size, local_context, query_length) <= 0:
        raise ValueError("model context and block lengths must be positive")
    if local_context % block_size:
        raise ValueError("model.local_context_length must be divisible by block_size")
    if query_length > local_context:
        raise ValueError("model.query_summary_length cannot exceed local context")
    if int(data.get("block_size", block_size)) != block_size:
        raise ValueError("data.block_size must equal model.block_size")
    if int(data.get("local_context_length", local_context)) != local_context:
        raise ValueError("data.local_context_length must equal model local context")
    if int(data.get("sequence_count", 0)) <= 0:
        raise ValueError("data.sequence_count must be positive")
    if int(data.get("sequence_length", 0)) <= local_context:
        raise ValueError("data.sequence_length must exceed the local context")
    checkpoint = Path(str(model["checkpoint_path"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    variants = expand_parameter_grid(
        dict(sweep.get("base_memory_config", {})),
        dict(sweep.get("grid", {})),
        block_size=block_size,
        maximum_variants=int(sweep.get("maximum_variants", 64)),
    )
    evaluation = dict(config.get("evaluation", {}))
    top_ns = tuple(sorted(set(int(value) for value in evaluation.get("top_ns", [4]))))
    if not top_ns or min(top_ns) <= 0:
        raise ValueError("evaluation.top_ns must contain positive values")
    if int(evaluation.get("permutation_trials", 32)) <= 0:
        raise ValueError("evaluation.permutation_trials must be positive")
    criteria = dict(config.get("criteria", {}))
    selection_top_n = int(criteria.get("selection_top_n", 4))
    if selection_top_n not in top_ns:
        raise ValueError("criteria.selection_top_n must be present in evaluation.top_ns")
    return {
        "checkpoint": checkpoint,
        "block_size": block_size,
        "local_context_length": local_context,
        "query_summary_length": query_length,
        "top_ns": top_ns,
        "variant_count": len(variants),
    }


def _capture_fixed_cache(
    *,
    bundle,
    router: LearnableRouterEncoder,
    example: EvaluationExample,
    memory_layers: Sequence[int],
    payload_layer: int,
    local_context_length: int,
    block_size: int,
    residual_layer: int,
    query_summary_length: int,
) -> FixedBlockCache:
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
    router_keys: dict[object, torch.Tensor] = {}
    blocks: list[CachedBlock] = []

    def on_block_ready(block, residual_summary: torch.Tensor) -> None:
        router_keys[block.block_id] = router.encode_block_tensor(
            residual_summary
        ).detach()

    def on_evict(block: EvictedCompleteBlock) -> None:
        if len(block.logical_positions) != int(block_size):
            raise RuntimeError("block calibration received an incomplete memory record")
        try:
            router_key = router_keys.pop(block.block.block_id)
        except KeyError as error:
            raise RuntimeError("evicted block has no prepared router key") from error
        key, value = block.layer_kv[int(payload_layer)]
        blocks.append(
            CachedBlock(
                block_id=str(block.block.block_id),
                logical_positions=tuple(int(value) for value in block.logical_positions),
                router_key=router_key.clone(),
                key=key.clone(),
                value=value.clone(),
            )
        )

    collector = BlockAlignedRollingContextCollector(
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
    block_positions = {
        block.block_id: block.logical_positions for block in blocks
    }
    labels = build_block_fact_labels(
        row.get("memory_fact_token_ranges", ()),
        block_positions,
        target_fact_id=str(row["target_example_id"]),
    )
    all_positions = tuple(
        position for block in blocks for position in block.logical_positions
    )
    if len(all_positions) != len(set(all_positions)):
        raise RuntimeError("fixed block cache contains duplicate logical positions")
    return FixedBlockCache(
        sample_id=example.sample_id,
        target_fact_id=str(row["target_example_id"]),
        primary_label_by_block=labels.primary_by_block,
        target_block_ids=labels.target_block_ids,
        label_diagnostics=labels.diagnostics,
        query_router_key=router.encode_query_tensor(result.query_summary).detach(),
        blocks=tuple(blocks),
        payload_layer=int(payload_layer),
        equivalent_memory_layers=tuple(int(value) for value in memory_layers),
        forward_calls=int(result.forward_calls),
        forwarded_tokens=int(result.forwarded_tokens),
        maximum_forward_context_length=int(result.maximum_forward_context_length),
        final_local_context_length=len(result.local_positions),
    )


def _evaluate_variant(
    cache: FixedBlockCache,
    variant: Mapping[str, Any],
    *,
    top_ns: Sequence[int],
    permutation_trials: int,
    seed: int,
) -> dict[str, Any]:
    if not cache.blocks:
        raise ValueError("fixed block cache contains no historical blocks")
    config = GpuBlockClusterMemoryConfig(**dict(variant["memory_config"]))
    first = cache.blocks[0]
    memory = GpuBlockClusterMemory(
        kv_heads=int(first.key.shape[1]),
        head_dim=int(first.key.shape[3]),
        router_dim=int(first.router_key.numel()),
        device=first.key.device,
        dtype=first.key.dtype,
        config=config,
    )
    started = time.perf_counter()
    for block in cache.blocks:
        memory.ingest_block(
            block.key,
            block.value,
            router_key=block.router_key,
            block_id=block.block_id,
            logical_positions=block.logical_positions,
        )
    if first.key.is_cuda:
        torch.cuda.synchronize(first.key.device)
    ingestion_seconds = time.perf_counter() - started

    ranked = memory.router_clusters(cache.query_router_key)
    memberships = [tuple(str(value) for value in row.block_ids) for row in ranked]
    cluster_by_block = {
        str(block_id): str(row.cluster_id)
        for row in ranked
        for block_id in row.block_ids
    }
    fact = bcubed_metrics(cluster_by_block, cache.primary_label_by_block)
    permutation = permutation_baseline(
        cluster_by_block,
        cache.primary_label_by_block,
        trials=int(permutation_trials),
        seed=_stable_seed(seed, cache.sample_id, variant["variant_id"]),
    )
    fact["bcubed_f1_gain_over_permutation"] = (
        float(fact["bcubed_f1"]) - float(permutation["mean_bcubed_f1"])
    )
    fact["beats_permutation_p95"] = float(
        float(fact["bcubed_f1"]) > float(permutation["p95_bcubed_f1"])
    )
    retrieval = {
        f"top_{int(top_n)}": target_block_retrieval_metrics(
            memberships,
            labeled_block_ids=tuple(cache.primary_label_by_block),
            target_block_ids=cache.target_block_ids,
            total_block_count=len(cache.blocks),
            top_n=int(top_n),
        )
        for top_n in top_ns
    }
    snapshot = memory.snapshot()
    candidate_requests = int(snapshot["local_candidate_requests"])
    return {
        "sample_id": cache.sample_id,
        "variant_id": str(variant["variant_id"]),
        "grid_values": dict(variant["grid_values"]),
        "payload_layer": cache.payload_layer,
        "classification_equivalent_layers": list(cache.equivalent_memory_layers),
        "structure": cluster_size_metrics(memberships),
        "block_label_quality": dict(cache.label_diagnostics),
        "fact_separation": fact,
        "permutation_baseline": permutation,
        "retrieval": retrieval,
        "assignment": {
            "created_cluster_rate": (
                float(snapshot["created_slots"]) / float(snapshot["ingested_records"])
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
        "ingestion_seconds": ingestion_seconds,
        "memory_snapshot": snapshot,
    }


def _values(
    rows: Sequence[Mapping[str, Any]], section: str, metric: str
) -> list[float]:
    return [float(row[section][metric]) for row in rows]


def _aggregate_variant(
    rows: Sequence[Mapping[str, Any]], *, top_ns: Sequence[int]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty calibration variant")
    result: dict[str, Any] = {
        "sample_count": len({str(row["sample_id"]) for row in rows}),
        "condition_count": len(rows),
    }
    for section in (
        "structure",
        "block_label_quality",
        "fact_separation",
        "assignment",
    ):
        result[section] = {
            name: _mean(_values(rows, section, name))
            for name in rows[0][section]
        }
    result["fact_separation"]["consistent_condition_fraction"] = _mean(
        _values(rows, "fact_separation", "beats_permutation_p95")
    )
    result["retrieval"] = {}
    for top_n in top_ns:
        key = f"top_{int(top_n)}"
        result["retrieval"][key] = {
            name: _mean(
                [float(row["retrieval"][key][name]) for row in rows]
            )
            for name in rows[0]["retrieval"][key]
        }
    return result


def _select_variant(
    summaries: Sequence[Mapping[str, Any]], criteria: Mapping[str, Any]
) -> dict[str, Any]:
    if not summaries:
        raise ValueError("calibration produced no variant summaries")
    min_b3 = float(criteria.get("minimum_mean_bcubed_f1", 0.20))
    min_gain = float(criteria.get("minimum_mean_b3_gain_over_permutation", 0.05))
    min_consistency = float(
        criteria.get("minimum_consistent_condition_fraction", 0.75)
    )
    max_singleton = float(criteria.get("maximum_singleton_cluster_ratio", 0.60))
    min_cluster_size = float(criteria.get("minimum_mean_cluster_size", 2.0))
    max_p95 = float(criteria.get("maximum_mean_p95_cluster_size", 32.0))
    selection_top_n = int(criteria.get("selection_top_n", 4))
    min_target_recall = float(criteria.get("minimum_target_fact_block_recall", 0.75))
    max_selected_ratio = float(criteria.get("maximum_selected_block_ratio", 0.75))
    min_samples = int(criteria.get("minimum_samples_for_structural_conclusion", 4))
    retrieval_key = f"top_{selection_top_n}"

    def passes_fact_and_retrieval(row: Mapping[str, Any]) -> bool:
        fact = row["summary"]["fact_separation"]
        retrieval = row["summary"]["retrieval"][retrieval_key]
        return (
            float(fact["bcubed_f1"]) >= min_b3
            and float(fact["bcubed_f1_gain_over_permutation"]) >= min_gain
            and float(fact["consistent_condition_fraction"]) >= min_consistency
            and float(retrieval["target_fact_block_recall"]) >= min_target_recall
        )

    eligible: list[Mapping[str, Any]] = []
    for row in summaries:
        structure = row["summary"]["structure"]
        retrieval = row["summary"]["retrieval"][retrieval_key]
        if (
            passes_fact_and_retrieval(row)
            and float(structure["singleton_cluster_ratio"]) <= max_singleton
            and float(structure["mean_cluster_size"]) >= min_cluster_size
            and float(structure["p95_cluster_size"]) <= max_p95
            and float(retrieval["selected_block_ratio"]) <= max_selected_ratio
        ):
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(
                row["summary"]["retrieval"][retrieval_key][
                    "target_fact_block_recall"
                ]
            ),
            -float(row["summary"]["fact_separation"]["bcubed_f1"]),
            float(
                row["summary"]["retrieval"][retrieval_key]["selected_block_ratio"]
            ),
            str(row["variant_id"]),
        )
    )
    best = max(
        summaries,
        key=lambda row: (
            float(row["summary"]["fact_separation"]["bcubed_f1_gain_over_permutation"]),
            float(
                row["summary"]["retrieval"][retrieval_key][
                    "target_fact_block_recall"
                ]
            ),
            -float(
                row["summary"]["retrieval"][retrieval_key]["selected_block_ratio"]
            ),
        ),
    )
    sample_count = max(int(row["summary"]["sample_count"]) for row in summaries)
    if eligible:
        status = "calibrated"
    elif sample_count < min_samples:
        status = "insufficient_conditions"
    elif not passes_fact_and_retrieval(best):
        status = "possible_structural_limit"
    else:
        status = "grid_did_not_meet_allocation_constraints"
    selected = eligible[0] if eligible else None
    best_retrieval = best["summary"]["retrieval"][retrieval_key]
    return {
        "status": status,
        "criteria": {
            "minimum_mean_bcubed_f1": min_b3,
            "minimum_mean_b3_gain_over_permutation": min_gain,
            "minimum_consistent_condition_fraction": min_consistency,
            "maximum_singleton_cluster_ratio": max_singleton,
            "minimum_mean_cluster_size": min_cluster_size,
            "maximum_mean_p95_cluster_size": max_p95,
            "selection_top_n": selection_top_n,
            "minimum_target_fact_block_recall": min_target_recall,
            "maximum_selected_block_ratio": max_selected_ratio,
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
        "best_observed_target_fact_block_recall": float(
            best_retrieval["target_fact_block_recall"]
        ),
        "best_observed_selected_block_ratio": float(
            best_retrieval["selected_block_ratio"]
        ),
    }


def run_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_config(config)
    output_dir = Path(str(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    model_config = dict(config["model"])
    checkpoint = Path(str(model_config.pop("checkpoint_path"))).resolve()
    block_size = int(model_config.pop("block_size", 64))
    local_context_length = int(model_config.pop("local_context_length", 512))
    residual_layer = int(model_config.pop("residual_layer", 40))
    query_summary_length = int(model_config.pop("query_summary_length", 16))
    requested_payload_layer = model_config.pop("payload_layer", None)
    router_device = str(
        model_config.pop("router_device", model_config.get("device", "cuda"))
    )
    model_cache_dir = model_config.pop("model_cache_dir", None)
    if model_cache_dir is not None:
        if "cache_dir" in model_config:
            raise ValueError("model may define only one of model_cache_dir and cache_dir")
        model_config["cache_dir"] = model_cache_dir
    bundle = load_frozen_gemma(**model_config)
    router = LearnableRouterEncoder.from_checkpoint(checkpoint, device=router_device)
    memory_layers = _memory_layers(bundle)
    payload_layer = (
        memory_layers[-1]
        if requested_payload_layer is None
        else int(requested_payload_layer)
    )
    if payload_layer not in memory_layers:
        raise ValueError(
            f"payload_layer {payload_layer} is not a physical full-attention layer"
        )
    dataset = DynamicConvoMemDataset(**dict(config["data"]))

    sweep = dict(config["sweep"])
    variants = expand_parameter_grid(
        dict(sweep.get("base_memory_config", {})),
        dict(sweep.get("grid", {})),
        block_size=block_size,
        maximum_variants=int(sweep.get("maximum_variants", 64)),
    )
    evaluation = dict(config.get("evaluation", {}))
    top_ns = tuple(validated["top_ns"])
    permutation_trials = int(evaluation.get("permutation_trials", 32))
    seed = int(config.get("seed", 13))
    all_rows: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    sample_path = output_dir / "sample_metrics.jsonl"
    with sample_path.open("w", encoding="utf-8") as stream:
        for sample_index, example in enumerate(dataset):
            print(
                f"[{example.sample_id}] capture block-aligned router cache once "
                f"({sample_index + 1}/{dataset.sequence_count})",
                flush=True,
            )
            capture_started = time.perf_counter()
            cache = _capture_fixed_cache(
                bundle=bundle,
                router=router,
                example=example,
                memory_layers=memory_layers,
                payload_layer=payload_layer,
                local_context_length=local_context_length,
                block_size=block_size,
                residual_layer=residual_layer,
                query_summary_length=query_summary_length,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tensor_bytes = sum(
                int(block.router_key.numel() * block.router_key.element_size())
                + int(block.key.numel() * block.key.element_size())
                + int(block.value.numel() * block.value.element_size())
                for block in cache.blocks
            ) + int(
                cache.query_router_key.numel()
                * cache.query_router_key.element_size()
            )
            captures.append(
                {
                    "sample_id": cache.sample_id,
                    "capture_count": 1,
                    "reused_by_variant_count": len(variants),
                    "block_record_count": len(cache.blocks),
                    "record_count_per_layer": len(cache.blocks),
                    "payload_layer": cache.payload_layer,
                    "classification_equivalent_layers": list(
                        cache.equivalent_memory_layers
                    ),
                    "labeled_block_count": len(cache.primary_label_by_block),
                    "target_block_count": len(cache.target_block_ids),
                    "label_diagnostics": dict(cache.label_diagnostics),
                    "forward_calls": cache.forward_calls,
                    "forwarded_tokens": cache.forwarded_tokens,
                    "maximum_forward_context_length": (
                        cache.maximum_forward_context_length
                    ),
                    "final_local_context_length": cache.final_local_context_length,
                    "resident_tensor_bytes": tensor_bytes,
                    "capture_seconds": time.perf_counter() - capture_started,
                }
            )
            for variant_index, variant in enumerate(variants):
                row = _evaluate_variant(
                    cache,
                    variant,
                    top_ns=top_ns,
                    permutation_trials=permutation_trials,
                    seed=seed,
                )
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                all_rows.append(row)
                print(
                    f"[{example.sample_id}] variant {variant_index + 1}/"
                    f"{len(variants)} {variant['variant_id']} "
                    f"b3_gain="
                    f"{row['fact_separation']['bcubed_f1_gain_over_permutation']:.4f}",
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
                f"single_pass_block_aligned_dynamic_{local_context_length}_to_"
                f"{local_context_length + block_size - 1}"
            ),
            "cache_reuse": "one_fixed_gpu_block_capture_per_sample_for_all_variants",
            "variant_initial_state": "empty_independent_block_memory",
            "record_unit": "complete_layer_local_block",
            "classification_feature": "learned_router_block_key",
            "kv_role": "opaque_replay_payload_not_used_for_assignment",
            "layer_equivalence": (
                "one_payload_layer_evaluated_because_router_key_assignment_is_"
                "identical_for_all_physical_memory_layers"
            ),
            "record_order": "identical_logical_position_order_for_every_variant",
            "parameter_policy": "static_per_variant_no_online_tuning",
            "eviction": "disabled",
            "fact_labels_visible_to_memory": False,
            "raw_cache_persisted": False,
        },
        "model": {
            "fingerprint": bundle.fingerprint,
            "checkpoint": str(checkpoint),
            "classification_equivalent_layers": list(memory_layers),
            "payload_layer": payload_layer,
            "block_size": block_size,
            "local_context_length": local_context_length,
            "dynamic_local_context_maximum": local_context_length + block_size - 1,
            "residual_layer": residual_layer,
            "query_summary_length": query_summary_length,
        },
        "dataset": dataset.descriptor,
        "cache_captures": captures,
        "variant_count": len(variants),
        "variants": summaries,
        "selection": selection,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(selection, sort_keys=True), flush=True)
    return result


__all__ = [
    "expand_parameter_grid",
    "run_calibration",
    "validate_config",
    "_select_variant",
]
