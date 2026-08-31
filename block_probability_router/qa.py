from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import torch

from learnable_index.aligned_builder import load_collected_sequences
from learnable_index.collectors import StudentCollectionConfig
from learnable_index.data import load_dataset
from learnable_index.model_adapter import (
    build_rolling_local_mask,
    cache_from_layer_kv,
    cache_suffix,
    forward_tokens,
    layer_kv_from_cache,
    load_frozen_gemma,
    new_full_dynamic_cache,
)
from learnable_index.trainer import resolve_device
from residual_cache.gemma4_memory_adapter import Gemma4StaticKVAdapter

from .model import minimum_cumulative_mass_mask
from .streaming_collection import (
    STUDENT_STATE_PROTOCOL,
    collect_streaming_student_state,
)
from .trainer import load_checkpoint


QA_SCHEMA_VERSION = 2


def _epsilon_label(value: float) -> str:
    return format(float(value), ".6g")


def _normalized_answer(text: str) -> str:
    words = re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)
    return " ".join(word for word in words if word not in {"a", "an", "the"})


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


def answer_contains(reference: str, prediction: str) -> float:
    normalized_reference = _normalized_answer(reference)
    normalized_prediction = _normalized_answer(prediction)
    return float(bool(normalized_reference) and normalized_reference in normalized_prediction)


@dataclass(frozen=True)
class QAEvaluationConfig:
    missing_mass_tolerances: tuple[float, ...]
    maximum_retrieval_blocks: int = -1
    maximum_samples: int | None = None
    maximum_new_tokens: int = 64
    prefill_chunk_size: int = 256
    router_device: str = "cpu"
    progress_every: int = 10
    bootstrap_iterations: int = 2_000
    seed: int = 13

    def __post_init__(self) -> None:
        tolerances = tuple(float(value) for value in self.missing_mass_tolerances)
        if not tolerances or tuple(sorted(set(tolerances))) != tolerances:
            raise ValueError("missing_mass_tolerances must be sorted and unique")
        if any(not 0.0 < value < 1.0 for value in tolerances):
            raise ValueError("missing-mass tolerances must lie in (0, 1)")
        object.__setattr__(self, "missing_mass_tolerances", tolerances)
        if self.maximum_retrieval_blocks != -1 and self.maximum_retrieval_blocks <= 0:
            raise ValueError("maximum_retrieval_blocks must be -1 or positive")
        if self.maximum_samples is not None and self.maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive when set")
        if self.maximum_new_tokens <= 0 or self.prefill_chunk_size <= 0:
            raise ValueError("generation lengths must be positive")
        if self.progress_every < 0:
            raise ValueError("progress_every must be non-negative")
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive")


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    stopped_on_eos: bool
    latency_seconds: float


def _decode(bundle, token_ids: Sequence[int]) -> str:
    return bundle.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _next_token(logits: torch.Tensor) -> int:
    return int(logits.detach().float().argmax(dim=-1).item())


def _synchronize_bundle(bundle) -> None:
    devices = {bundle.input_device, *bundle.cache_layer_devices}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@torch.inference_mode()
def _generate_full_context(
    bundle,
    prompt_token_ids: Sequence[int],
    *,
    maximum_new_tokens: int,
    prefill_chunk_size: int,
) -> GenerationResult:
    prompt = tuple(int(value) for value in prompt_token_ids)
    if not prompt:
        raise ValueError("generation prompt cannot be empty")
    eos_token_id = bundle.tokenizer.eos_token_id
    cache = new_full_dynamic_cache()
    logits = None
    _synchronize_bundle(bundle)
    start_time = time.perf_counter()
    for chunk_start in range(0, len(prompt), prefill_chunk_size):
        chunk_end = min(len(prompt), chunk_start + prefill_chunk_size)
        output = forward_tokens(
            bundle,
            prompt[chunk_start:chunk_end],
            range(chunk_start, chunk_end),
            past_key_values=cache,
            use_cache=True,
            logical_cache_position=True,
        )
        logits = output.logits[0, -1]
        cache = cache_from_layer_kv(layer_kv_from_cache(output.past_key_values))
    if logits is None:
        raise RuntimeError("full-context prefill produced no logits")

    generated: list[int] = []
    stopped = False
    logical_position = len(prompt)
    for _ in range(maximum_new_tokens):
        token_id = _next_token(logits)
        generated.append(token_id)
        if eos_token_id is not None and token_id == int(eos_token_id):
            stopped = True
            break
        output = forward_tokens(
            bundle,
            [token_id],
            [logical_position],
            past_key_values=cache,
            use_cache=True,
            logical_cache_position=True,
        )
        logits = output.logits[0, -1]
        cache = cache_from_layer_kv(layer_kv_from_cache(output.past_key_values))
        logical_position += 1
    _synchronize_bundle(bundle)
    latency = time.perf_counter() - start_time
    return GenerationResult(
        token_ids=tuple(generated),
        text=_decode(bundle, generated),
        stopped_on_eos=stopped,
        latency_seconds=latency,
    )


@torch.inference_mode()
def _generate_sparse_replay(
    bundle,
    *,
    local_layer_kv,
    local_positions: Sequence[int],
    initial_token_id: int,
    initial_logical_position: int,
    historical_layer_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    local_context_length: int,
    block_size: int,
    maximum_new_tokens: int,
) -> GenerationResult:
    positions = tuple(int(value) for value in local_positions)
    if local_context_length <= 0 or block_size <= 0:
        raise ValueError("local context length and block size must be positive")
    maximum_context_length = local_context_length + block_size
    if not local_context_length <= len(positions) < maximum_context_length:
        raise ValueError(
            "initial block-aligned cache must lie in [local, local + block)"
        )
    if positions[0] % block_size:
        raise ValueError("initial block-aligned cache must start on a block boundary")
    if any(right != left + 1 for left, right in zip(positions, positions[1:])):
        raise ValueError("initial local cache positions must be contiguous")
    if positions[-1] + 1 != initial_logical_position:
        raise ValueError("initial generation token must immediately follow the local cache")
    eos_token_id = bundle.tokenizer.eos_token_id
    cache = cache_from_layer_kv(local_layer_kv)
    current_token = int(initial_token_id)
    logical_position = int(initial_logical_position)
    generated: list[int] = []
    stopped = False
    _synchronize_bundle(bundle)
    start_time = time.perf_counter()
    adapter = (
        Gemma4StaticKVAdapter(bundle.model, dict(historical_layer_kv))
        if historical_layer_kv
        else nullcontext()
    )
    with adapter:
        for _ in range(maximum_new_tokens):
            mask = build_rolling_local_mask(
                bundle,
                past_positions=positions,
                query_positions=(logical_position,),
                local_context_length=maximum_context_length,
            )
            output = forward_tokens(
                bundle,
                [current_token],
                [logical_position],
                past_key_values=cache,
                attention_mask=mask,
                use_cache=True,
                logical_cache_position=True,
            )
            cache = cache_from_layer_kv(layer_kv_from_cache(output.past_key_values))
            positions = positions + (logical_position,)
            if len(positions) == maximum_context_length:
                cache = cache_suffix(cache, local_context_length)
                positions = positions[block_size:]
            elif len(positions) > maximum_context_length:
                raise RuntimeError("generation exceeded the block-aligned context bound")
            token_id = _next_token(output.logits[0, -1])
            generated.append(token_id)
            if eos_token_id is not None and token_id == int(eos_token_id):
                stopped = True
                break
            current_token = token_id
            logical_position += 1
    _synchronize_bundle(bundle)
    latency = time.perf_counter() - start_time
    return GenerationResult(
        token_ids=tuple(generated),
        text=_decode(bundle, generated),
        stopped_on_eos=stopped,
        latency_seconds=latency,
    )


def _evidence_only_prompt(record) -> tuple[int, ...]:
    row = record.metadata
    memory_start, memory_end = (int(value) for value in row["distractor_token_range"])
    target_start, target_end = (int(value) for value in row["target_memory_chunk_range"])
    answer_start = int(row["answer_start_position"])
    if not (
        0 <= memory_start <= target_start < target_end <= memory_end <= answer_start
        <= len(record.token_ids)
    ):
        raise ValueError("ConvoMem evidence and prompt ranges are inconsistent")
    return (
        record.token_ids[:memory_start]
        + record.token_ids[target_start:target_end]
        + record.token_ids[memory_end:answer_start]
    )


@torch.inference_mode()
def _router_selections(
    router,
    query_summary: torch.Tensor,
    block_summaries: torch.Tensor,
    config: QAEvaluationConfig,
    device: torch.device,
):
    # Streaming residuals inherit the frozen model dtype (normally bf16),
    # whereas router checkpoints are stored and loaded in fp32.  Match the
    # router parameter dtype explicitly so the real streaming QA path has the
    # same numerical contract as offline evaluation over persisted fp32 data.
    router_dtype = next(router.parameters()).dtype
    query = query_summary.unsqueeze(0).to(device=device, dtype=router_dtype)
    blocks = block_summaries.unsqueeze(0).to(device=device, dtype=router_dtype)
    candidate_mask = torch.ones(
        (1, block_summaries.shape[0]), dtype=torch.bool, device=device
    )
    output = router(query, blocks, candidate_mask)
    probabilities = output.probabilities[0].detach().float().cpu()
    selections: dict[float, tuple[int, ...]] = {}
    for tolerance in config.missing_mass_tolerances:
        selected = minimum_cumulative_mass_mask(
            output.probabilities,
            candidate_mask,
            tolerance,
            config.maximum_retrieval_blocks,
        )[0].detach().cpu()
        selections[tolerance] = tuple(
            index for index, keep in enumerate(selected.tolist()) if keep
        )
    return probabilities, selections


def _physical_full_attention_layers(bundle) -> tuple[int, ...]:
    physical_count = bundle.physical_cache_layer_count
    layer_types = tuple(bundle.text_config.layer_types[:physical_count])
    layers = tuple(
        index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
    )
    if not layers:
        raise ValueError("Gemma configuration exposes no physical full-attention layers")
    return layers


def _pack_selected_block_kv(
    block_kv: Mapping[int, Mapping[int, tuple[torch.Tensor, torch.Tensor]]],
    selected_indices: Sequence[int],
    full_attention_layers: Sequence[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if not selected_indices:
        return {}
    ordered = tuple(sorted(int(value) for value in selected_indices))
    return {
        layer: (
            torch.cat([block_kv[index][layer][0] for index in ordered], dim=2),
            torch.cat([block_kv[index][layer][1] for index in ordered], dim=2),
        )
        for layer in full_attention_layers
    }


def _generation_metrics(reference: str, result: GenerationResult) -> dict[str, Any]:
    return {
        "prediction": result.text,
        "predicted_token_ids": list(result.token_ids),
        "answer_exact_match": exact_match(reference, result.text),
        "answer_token_f1": token_f1(reference, result.text),
        "answer_contains": answer_contains(reference, result.text),
        "generated_token_count": len(result.token_ids),
        "stopped_on_eos": float(result.stopped_on_eos),
        "latency_seconds": result.latency_seconds,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _bootstrap_mean_interval(
    values: Sequence[float], *, iterations: int, seed: int
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty metric")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )
    return [
        float(means[int(0.025 * iterations)]),
        float(means[min(iterations - 1, int(0.975 * iterations))]),
    ]


def _aggregate(rows: Sequence[Mapping[str, Any]], config: QAEvaluationConfig) -> dict[str, Any]:
    condition_names = sorted({name for row in rows for name in row["conditions"]})
    conditions: dict[str, Any] = {}
    excluded = {"prediction", "predicted_token_ids"}
    for condition_index, name in enumerate(condition_names):
        metrics = [row["conditions"][name] for row in rows]
        numeric_names = sorted(
            {
                key
                for metric in metrics
                for key, value in metric.items()
                if key not in excluded
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        aggregate = {"sample_count": len(metrics)}
        for metric_index, metric_name in enumerate(numeric_names):
            values = [float(metric[metric_name]) for metric in metrics]
            aggregate[metric_name] = _mean(values)
            if metric_name in {"answer_exact_match", "answer_token_f1", "answer_contains"}:
                aggregate[f"{metric_name}_95ci"] = _bootstrap_mean_interval(
                    values,
                    iterations=config.bootstrap_iterations,
                    seed=config.seed + condition_index * 1009 + metric_index,
                )
        conditions[name] = aggregate

    comparisons: dict[str, Any] = {}
    for name in condition_names:
        if not name.startswith("router_epsilon_"):
            continue
        comparison: dict[str, Any] = {}
        for baseline in ("full_context", "evidence_only", "local_only"):
            for metric_name in ("answer_exact_match", "answer_token_f1"):
                deltas = [
                    float(row["conditions"][name][metric_name])
                    - float(row["conditions"][baseline][metric_name])
                    for row in rows
                ]
                key = f"delta_vs_{baseline}/{metric_name}"
                comparison[key] = _mean(deltas)
                comparison[f"{key}_95ci"] = _bootstrap_mean_interval(
                    deltas,
                    iterations=config.bootstrap_iterations,
                    seed=config.seed + len(comparisons) * 2027,
                )
        full_f1 = [float(row["conditions"]["full_context"]["answer_token_f1"]) for row in rows]
        router_f1 = [float(row["conditions"][name]["answer_token_f1"]) for row in rows]
        comparison["f1_wins_vs_full_context"] = sum(
            candidate > baseline for candidate, baseline in zip(router_f1, full_f1)
        )
        comparison["f1_ties_vs_full_context"] = sum(
            candidate == baseline for candidate, baseline in zip(router_f1, full_f1)
        )
        comparison["f1_losses_vs_full_context"] = sum(
            candidate < baseline for candidate, baseline in zip(router_f1, full_f1)
        )
        comparisons[name] = comparison
    return {"sample_count": len(rows), "conditions": conditions, "paired_comparisons": comparisons}


def _evidence_block_metrics(record, sample, selected_indices: Sequence[int]) -> dict[str, float]:
    evidence_indices = {int(value) for value in record.metadata.get("evidence_block_indices", ())}
    candidate_block_indices = {
        index: int(block.start_position) // int(block.length)
        for index, block in enumerate(sample.candidate_blocks)
    }
    selected_evidence = {
        candidate_block_indices[index]
        for index in selected_indices
        if candidate_block_indices[index] in evidence_indices
    }
    eligible_evidence = evidence_indices.intersection(candidate_block_indices.values())
    return {
        "evidence_block_recall": (
            0.0 if not eligible_evidence else len(selected_evidence) / len(eligible_evidence)
        ),
        "any_evidence_hit": float(bool(selected_evidence)),
        "all_evidence_hit": float(bool(eligible_evidence) and selected_evidence == eligible_evidence),
    }


def _verify_streaming_sample_alignment(sample, state) -> None:
    """Reject collections whose router inputs were not captured by this stream."""

    query = state.query_summary.detach().float().cpu()
    blocks = state.block_summaries.detach().float().cpu()
    if query.shape != sample.query_summary.shape or not torch.allclose(
        query, sample.query_summary.float(), rtol=1e-4, atol=1e-4
    ):
        raise RuntimeError(
            "stored query summary does not match block-aligned streaming replay"
        )
    if blocks.shape != sample.block_summaries.shape or not torch.allclose(
        blocks, sample.block_summaries.float(), rtol=1e-4, atol=1e-4
    ):
        raise RuntimeError(
            "stored block summaries do not match block-aligned streaming replay"
        )


def evaluate_generation_qa(
    *,
    collection_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples_output_path: str | Path,
    model_name: str,
    model_device: str,
    dtype: str,
    local_files_only: bool,
    model_cache_dir: str | None,
    config: QAEvaluationConfig,
) -> dict[str, Any]:
    collection_dir = Path(collection_dir)
    output_path = Path(output_path)
    samples_output_path = Path(samples_output_path)
    dataset, dataset_manifest = load_dataset(collection_dir / "dataset")
    records = {record.sequence_id: record for record in load_collected_sequences(collection_dir)}
    collection_manifest = json.loads(
        (collection_dir / "collection_manifest.json").read_text(encoding="utf-8")
    )
    bundle = load_frozen_gemma(
        model_name,
        device=model_device,
        dtype=dtype,
        local_files_only=local_files_only,
        cache_dir=model_cache_dir,
    )
    if dataset_manifest["metadata"]["model_fingerprint"] != bundle.fingerprint:
        raise ValueError("QA model fingerprint does not match the aligned collection")
    if collection_manifest.get("student_state_protocol") != STUDENT_STATE_PROTOCOL:
        raise ValueError(
            "QA requires a block-aligned streaming probability-router collection"
        )
    if (
        dataset_manifest.get("metadata", {}).get("student_state_protocol")
        != STUDENT_STATE_PROTOCOL
    ):
        raise ValueError("dataset student-state protocol does not match QA replay")
    collection_config = collection_manifest["collection_config"]
    student_config = StudentCollectionConfig(**collection_config["student"])
    block_size = int(collection_config["plan"]["block_size"])
    if student_config.local_context_length % block_size:
        raise ValueError("block-aligned local context must be divisible by block size")
    native_sliding_window = int(bundle.text_config.sliding_window)
    if student_config.local_context_length != native_sliding_window:
        raise ValueError(
            "probability-router replay local context must equal Gemma's native "
            f"sliding window ({native_sliding_window})"
        )
    full_attention_layers = _physical_full_attention_layers(bundle)
    router, router_config, _loss_config, _train_config, checkpoint = load_checkpoint(
        checkpoint_path
    )
    checkpoint_protocol = checkpoint.get("student_state_protocol")
    if router_config.residual_dim != dataset.residual_dim:
        raise ValueError("router and aligned dataset residual dimensions do not match")
    router_device = resolve_device(config.router_device)
    router.to(router_device).eval()

    selected_samples = dataset.samples
    if config.maximum_samples is not None:
        selected_samples = selected_samples[: config.maximum_samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples_output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with samples_output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_number, sample in enumerate(selected_samples, start=1):
            record = records[sample.sequence_id]
            answer_start = int(record.metadata["answer_start_position"])
            if sample.first_future_position_affected_by_retrieval != answer_start - 1:
                raise ValueError("QA requires the answer-aligned retrieval schedule")
            reference = str(record.metadata["answer"])
            streaming_state = collect_streaming_student_state(
                bundle,
                record,
                sample,
                student_config,
                block_size=block_size,
                capture_layers=full_attention_layers,
            )
            _verify_streaming_sample_alignment(sample, streaming_state)
            probabilities, selections = _router_selections(
                router,
                streaming_state.query_summary,
                streaming_state.block_summaries,
                config,
                router_device,
            )
            conditions: dict[str, dict[str, Any]] = {}

            full_result = _generate_full_context(
                bundle,
                record.token_ids[:answer_start],
                maximum_new_tokens=config.maximum_new_tokens,
                prefill_chunk_size=config.prefill_chunk_size,
            )
            conditions["full_context"] = _generation_metrics(reference, full_result)
            evidence_result = _generate_full_context(
                bundle,
                _evidence_only_prompt(record),
                maximum_new_tokens=config.maximum_new_tokens,
                prefill_chunk_size=config.prefill_chunk_size,
            )
            conditions["evidence_only"] = _generation_metrics(reference, evidence_result)

            local_layer_kv = streaming_state.local_layer_kv
            local_positions = streaming_state.local_positions
            initial_position = sample.first_future_position_affected_by_retrieval
            initial_token = record.token_ids[initial_position]
            local_result = _generate_sparse_replay(
                bundle,
                local_layer_kv=local_layer_kv,
                local_positions=local_positions,
                initial_token_id=initial_token,
                initial_logical_position=initial_position,
                historical_layer_kv={},
                local_context_length=student_config.local_context_length,
                block_size=block_size,
                maximum_new_tokens=config.maximum_new_tokens,
            )
            conditions["local_only"] = _generation_metrics(reference, local_result)

            block_kv = streaming_state.block_layer_kv
            candidate_count = len(sample.candidate_blocks)
            physical_count = bundle.physical_cache_layer_count
            prompt_length = answer_start
            for tolerance in config.missing_mass_tolerances:
                indices = selections[tolerance]
                historical = _pack_selected_block_kv(
                    block_kv, indices, full_attention_layers
                )
                result = _generate_sparse_replay(
                    bundle,
                    local_layer_kv=local_layer_kv,
                    local_positions=local_positions,
                    initial_token_id=initial_token,
                    initial_logical_position=initial_position,
                    historical_layer_kv=historical,
                    local_context_length=student_config.local_context_length,
                    block_size=block_size,
                    maximum_new_tokens=config.maximum_new_tokens,
                )
                metric = _generation_metrics(reference, result)
                selected_mass = float(probabilities[list(indices)].sum()) if indices else 0.0
                teacher_mass = float(
                    sample.conditional_teacher_distribution[list(indices)].sum()
                ) if indices else 0.0
                selected_tokens = sum(sample.candidate_blocks[index].length for index in indices)
                visible_layer_tokens = (
                    len(local_positions) * physical_count
                    + selected_tokens * len(full_attention_layers)
                )
                metric.update(
                    {
                        "selected_block_count": len(indices),
                        "selected_block_fraction": len(indices) / candidate_count,
                        "selected_historical_tokens": selected_tokens,
                        "predicted_probability_mass": selected_mass,
                        "teacher_mass_recall": teacher_mass,
                        "visible_layer_token_kv_ratio": (
                            visible_layer_tokens / (prompt_length * physical_count)
                        ),
                        **_evidence_block_metrics(record, sample, indices),
                    }
                )
                conditions[f"router_epsilon_{_epsilon_label(tolerance)}"] = metric
                del historical

            row = {
                "qa_schema_version": QA_SCHEMA_VERSION,
                "sample_id": sample.sample_id,
                "sequence_id": sample.sequence_id,
                "reference_answer": reference,
                "answer_start_position": answer_start,
                "candidate_block_count": candidate_count,
                "evidence_placement_bin": record.metadata.get("evidence_placement_bin"),
                "evidence_distance_tokens": record.metadata.get(
                    "evidence_to_answer_distance_tokens"
                ),
                "streaming_state": {
                    "protocol": STUDENT_STATE_PROTOCOL,
                    "local_context_start": local_positions[0],
                    "local_context_end": local_positions[-1] + 1,
                    "local_context_length": len(local_positions),
                    "forward_calls": streaming_state.forward_calls,
                    "forwarded_tokens": streaming_state.forwarded_tokens,
                    "evicted_blocks": streaming_state.evicted_blocks,
                    "evicted_tokens": streaming_state.evicted_tokens,
                    "maximum_forward_context_length": (
                        streaming_state.maximum_forward_context_length
                    ),
                },
                "conditions": conditions,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if config.progress_every and (
                sample_number == 1
                or sample_number % config.progress_every == 0
                or sample_number == len(selected_samples)
            ):
                print(
                    json.dumps(
                        {
                            "event": "qa_progress",
                            "completed": sample_number,
                            "total": len(selected_samples),
                            "sample_id": sample.sample_id,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    result = {
        "qa_schema_version": QA_SCHEMA_VERSION,
        "evaluation_kind": "greedy_autoregressive_long_context_qa",
        "teacher_forcing": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "student_state_protocol": {
            "evaluation": STUDENT_STATE_PROTOCOL,
            "checkpoint": checkpoint_protocol,
            "matched": checkpoint_protocol == STUDENT_STATE_PROTOCOL,
        },
        "model_fingerprint": bundle.fingerprint,
        "qa_config": asdict(config),
        "replay_semantics": {
            "student_state_protocol": STUDENT_STATE_PROTOCOL,
            "historical_unit": f"complete_{block_size}_token_candidate_block",
            "block_capture": "single_causal_pass_at_atomic_block_eviction",
            "router_inputs": "same_stream_query_and_completed_block_states",
            "selected_kv_layers": "physical_full_attention_only",
            "sliding_attention_layers": (
                f"native_block_aligned_{student_config.local_context_length}_to_"
                f"{student_config.local_context_length + block_size - 1}_token_window"
            ),
            "logical_positions": "original_sequence_positions",
            "retrieval_time": "immediately_before_first_answer token prediction",
            "generation_cache_policy": (
                "grow_to_local_plus_block_then_atomically_evict_one_complete_block"
            ),
        },
        "summary": _aggregate(rows, config),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _parse_tolerances(value: str) -> tuple[float, ...]:
    values = tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autoregressive QA evaluation for a block-probability router"
    )
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--router-device", default="cpu")
    parser.add_argument("--missing-mass-tolerances", required=True)
    parser.add_argument("--max-block", type=int, default=-1)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = QAEvaluationConfig(
        missing_mass_tolerances=_parse_tolerances(
            arguments.missing_mass_tolerances
        ),
        maximum_retrieval_blocks=arguments.max_block,
        maximum_samples=arguments.maximum_samples,
        maximum_new_tokens=arguments.maximum_new_tokens,
        prefill_chunk_size=arguments.prefill_chunk_size,
        router_device=arguments.router_device,
        progress_every=arguments.progress_every,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.seed,
    )
    result = evaluate_generation_qa(
        collection_dir=arguments.collection_dir,
        checkpoint_path=arguments.checkpoint,
        output_path=arguments.output,
        samples_output_path=arguments.samples_output,
        model_name=arguments.model_name,
        model_device=arguments.model_device,
        dtype=arguments.dtype,
        local_files_only=not arguments.allow_network,
        model_cache_dir=arguments.model_cache_dir,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
