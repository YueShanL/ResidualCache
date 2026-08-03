from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from residual_cache.data_process import DataConfig, build_dataset
from residual_cache.residual_collect import (
    _hidden,
    _q_proj,
    _stack_layer_rows,
    _torch_dtype,
    _transformer_layers,
)


INDEX_METHODS = ("momentum_dtw", "mean_state_cosine", "mean_q_cosine")
STATE_TARGETS = ("pre_attn", "block_output")


@dataclass(frozen=True)
class PromptFreeCollectConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    dtype: str = "auto"
    device: str = "auto"
    layers: str = "all"
    state_target: str = "block_output"
    trajectory_points: int = 24
    projection_dim: int = 64
    projection_seed: int = 13


@dataclass(frozen=True)
class PromptFreeAnalysisConfig:
    collect_dir: Path
    output_dir: Path
    top_k: int = 10
    dtw_window: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class PromptFreePipelineConfig:
    output_dir: Path
    model_name: str | None = None
    max_facts: int = 120
    seed: int = 13
    max_context_turns: int = 48
    convomem_root: Path | None = None
    chat_template: bool = False
    dataset_path: Path | None = None
    reuse_collection: Path | None = None
    dtype: str = "auto"
    device: str = "auto"
    layers: str = "all"
    state_target: str = "block_output"
    trajectory_points: int = 24
    projection_dim: int = 64
    projection_seed: int = 13
    top_k: int = 10
    dtw_window: int | None = None
    analysis_device: str = "auto"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_layer_spec(spec: str, layer_count: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(layer_count))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range {part!r}.")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("At least one layer must be selected.")
    invalid = sorted(index for index in selected if index < 0 or index >= layer_count)
    if invalid:
        raise ValueError(f"Layer indices out of range for {layer_count} layers: {invalid}")
    return sorted(selected)


def _load_model(config: PromptFreeCollectConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Prompt-free collection requires torch and transformers from the current environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": _torch_dtype(torch, config.dtype)}
    if config.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if config.device != "auto":
        model.to(config.device)
    model.eval()
    return torch, tokenizer, model


def _encode_user_span(tokenizer, user_input: str, *, raw_prompt: bool) -> tuple[str, list[int], list[int]]:
    """Encode one user input and return positions that overlap only its content.

    A model chat template may add BOS, role, and generation-marker tokens. Those
    tokens are fed to the model but deliberately excluded from all three index
    representations.
    """

    content = user_input.strip()
    if not content:
        raise ValueError("Cannot build an index from an empty user input.")
    if raw_prompt:
        formatted = content
        content_start = 0
    else:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Dataset requests a chat template, but the tokenizer has none.")
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        content_start = formatted.find(content)
        if content_start < 0:
            raise RuntimeError("Could not locate the original user content in the rendered chat template.")
    content_end = content_start + len(content)

    try:
        encoded = tokenizer(
            formatted,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = list(encoded.offset_mapping)
    except (NotImplementedError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            "Exact user-span indexing requires a fast tokenizer with offset_mapping support."
        ) from exc

    input_ids = list(encoded.input_ids)
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and end > content_start and start < content_end
    ]
    if not positions:
        raise RuntimeError("The tokenizer produced no tokens overlapping the user content.")
    return formatted, input_ids, positions


def _capture_user_states(
    torch,
    model,
    input_ids,
    positions: list[int],
    layer_indices: list[int],
    state_target: str,
) -> tuple[object, object]:
    if state_target not in STATE_TARGETS:
        raise ValueError(f"Unknown state target {state_target!r}; choose one of {STATE_TARGETS}.")
    layers = _transformer_layers(model)
    position_tensor = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
    selected = set(layer_indices)
    states: dict[int, object] = {}
    queries: dict[int, object] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        if layer_index not in selected:
            continue

        if state_target == "pre_attn":
            def save_state(_module, inputs, *, index=layer_index):
                states[index] = inputs[0][0, position_tensor, :].detach().float().cpu()

            handles.append(layer.register_forward_pre_hook(save_state))
        else:
            def save_state(_module, _inputs, output, *, index=layer_index):
                states[index] = _hidden(output)[0, position_tensor, :].detach().float().cpu()

            handles.append(layer.register_forward_hook(save_state))

        q_proj = _q_proj(layer)
        if q_proj is None:
            raise RuntimeError(f"Could not find self-attention q_proj at layer {layer_index}.")

        def save_q(_module, _inputs, output, *, index=layer_index):
            queries[index] = _hidden(output)[0, position_tensor, :].detach().float().cpu()

        handles.append(q_proj.register_forward_hook(save_q))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing_state = [index for index in layer_indices if index not in states]
    missing_q = [index for index in layer_indices if index not in queries]
    if missing_state or missing_q:
        raise RuntimeError(
            f"Incomplete hook capture: missing state layers={missing_state}, q layers={missing_q}."
        )
    return (
        _stack_layer_rows(torch, [states[index] for index in layer_indices]),
        _stack_layer_rows(torch, [queries[index] for index in layer_indices]),
    )


def _temporal_resample(torch, trajectory, points: int):
    if points < 1:
        raise ValueError("trajectory_points must be positive.")
    if trajectory.ndim != 3 or trajectory.shape[1] < 1:
        raise ValueError("Expected trajectory shape [layers, time, width] with at least one time step.")
    if trajectory.shape[1] == points:
        return trajectory.float().contiguous()
    values = trajectory.float().permute(0, 2, 1)
    values = torch.nn.functional.interpolate(values, size=points, mode="linear", align_corners=False)
    return values.permute(0, 2, 1).contiguous()


def _random_projection(torch, width: int, projection_dim: int, seed: int):
    if projection_dim < 0:
        raise ValueError("projection_dim must be non-negative.")
    if projection_dim == 0 or projection_dim >= width:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(width) + 97 * int(projection_dim))
    signs = torch.randint(0, 2, (width, projection_dim), generator=generator, dtype=torch.int8)
    return (signs.float().mul_(2).sub_(1)) / math.sqrt(projection_dim)


def _momentum_index(torch, states, points: int, projection):
    if states.shape[1] < 2:
        momentum = torch.zeros(
            (states.shape[0], 1, states.shape[2]),
            dtype=torch.float32,
        )
    else:
        momentum = states[:, 1:, :].float() - states[:, :-1, :].float()
    momentum = _temporal_resample(torch, momentum, points)
    if projection is not None:
        momentum = momentum @ projection
    return momentum.contiguous()


def _validate_prompt_free_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("The dataset contains no rows.")
    bad_suite = [row.get("prompt_id") for row in rows if row.get("suite") != "convomem_query_to_fact"]
    if bad_suite:
        raise ValueError(
            "Prompt-free KNN expects paired convomem_query_to_fact rows; "
            f"found incompatible rows such as {bad_suite[:3]}."
        )
    prompted = [row.get("prompt_id") for row in rows if row.get("knowledge_prompt")]
    if prompted:
        raise ValueError(
            "Knowledge-instructed rows are not prompt-free. Rebuild without "
            f"--convomem-knowledge-prompt; examples: {prompted[:3]}."
        )
    conditions = {row.get("condition_id") for row in rows}
    if not {"fact_reference", "question_query"} <= conditions:
        raise ValueError("Dataset must contain both fact_reference and question_query rows.")


def collect_prompt_free_indices(config: PromptFreeCollectConfig) -> Path:
    rows = _read_jsonl(config.dataset_path)
    _validate_prompt_free_rows(rows)
    if config.trajectory_points < 1:
        raise ValueError("trajectory_points must be positive.")
    if config.projection_dim < 0:
        raise ValueError("projection_dim must be non-negative.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = config.output_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    serialized = {
        **asdict(config),
        "dataset_path": str(config.dataset_path),
        "output_dir": str(config.output_dir),
        "index_methods": INDEX_METHODS,
        "momentum_definition": "first difference h[t] - h[t-1] over user-content tokens",
        "dtw_local_cost": "cosine distance after deterministic random projection",
    }
    (config.output_dir / "collect_config.json").write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    torch, tokenizer, model = _load_model(config)
    all_layers = _transformer_layers(model)
    layer_indices = _parse_layer_spec(config.layers, len(all_layers))
    metadata = []
    projection = None
    projection_width = None
    try:
        for row_index, row in enumerate(rows):
            formatted, token_ids, user_positions = _encode_user_span(
                tokenizer,
                row["prompt"],
                raw_prompt=bool(row.get("raw_prompt")),
            )
            device = next(model.parameters()).device
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            states, queries = _capture_user_states(
                torch,
                model,
                input_ids,
                user_positions,
                layer_indices,
                config.state_target,
            )
            if projection is None:
                projection_width = int(states.shape[-1])
                projection = _random_projection(
                    torch,
                    projection_width,
                    config.projection_dim,
                    config.projection_seed,
                )
            elif states.shape[-1] != projection_width:
                raise RuntimeError(
                    f"Hidden width changed from {projection_width} to {states.shape[-1]} across rows."
                )

            momentum = _momentum_index(
                torch,
                states,
                config.trajectory_points,
                projection,
            )
            tensor_path = tensor_dir / f"{row_index:05d}_{row['prompt_id']}.pt"
            torch.save(
                {
                    "layer_indices": layer_indices,
                    "state_target": config.state_target,
                    "state_mean": states.mean(dim=1).contiguous(),
                    "q_mean": queries.mean(dim=1).contiguous(),
                    "momentum": momentum,
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "user_positions": torch.tensor(user_positions, dtype=torch.long),
                },
                tensor_path,
            )
            metadata.append(
                {
                    **row,
                    "tensor_path": str(tensor_path.relative_to(config.output_dir)),
                    "input_tokens": len(token_ids),
                    "user_tokens": len(user_positions),
                    "template_tokens_excluded": len(token_ids) - len(user_positions),
                    "formatted_input": formatted,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "metadata.jsonl", metadata)
    manifest = {
        "rows": len(metadata),
        "layers": layer_indices,
        "state_target": config.state_target,
        "trajectory_points": config.trajectory_points,
        "original_momentum_width": projection_width,
        "projected_momentum_width": (
            (
                projection_width
                if config.projection_dim == 0
                else min(config.projection_dim, projection_width)
            )
            if projection_width is not None
            else None
        ),
        "projection_seed": config.projection_seed,
        "prompt_free": True,
        "generated_tokens_used": False,
    }
    (config.output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config.output_dir


def _analysis_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _cosine_distance_matrix(torch, left, right):
    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    return (1.0 - left @ right.T).clamp_min(0.0)


def _batched_dtw_distance_matrix(torch, left, right, window: int | None = None):
    """Compute all query/reference DTW distances for fixed-length trajectories."""

    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("DTW inputs must have shape [items, time, width].")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("DTW feature widths must match.")
    left_steps, right_steps = int(left.shape[1]), int(right.shape[1])
    if left_steps < 1 or right_steps < 1:
        raise ValueError("DTW trajectories cannot be empty.")
    if window is not None and window < 0:
        raise ValueError("dtw_window must be non-negative or omitted.")
    requested_window = max(left_steps, right_steps) if window is None else window
    effective_window = max(requested_window, abs(left_steps - right_steps))

    left = torch.nn.functional.normalize(left.float(), dim=-1)
    right = torch.nn.functional.normalize(right.float(), dim=-1)
    local_cost = (1.0 - torch.einsum("qtd,rsd->qrts", left, right)).clamp_min(0.0)
    query_count, ref_count = int(left.shape[0]), int(right.shape[0])
    infinity = torch.tensor(float("inf"), device=left.device, dtype=local_cost.dtype)
    previous = torch.full(
        (query_count, ref_count, right_steps + 1),
        infinity,
        device=left.device,
        dtype=local_cost.dtype,
    )
    previous[:, :, 0] = 0.0
    for i in range(1, left_steps + 1):
        current = torch.full_like(previous, infinity)
        start = max(1, i - effective_window)
        end = min(right_steps, i + effective_window)
        for j in range(start, end + 1):
            best_predecessor = torch.minimum(
                torch.minimum(current[:, :, j - 1], previous[:, :, j]),
                previous[:, :, j - 1],
            )
            current[:, :, j] = local_cost[:, :, i - 1, j - 1] + best_predecessor
        previous = current
    # Collection resamples every trajectory to one length. Dividing by the
    # common number of steps leaves rankings unchanged and makes values legible.
    return previous[:, :, right_steps] / max(left_steps, right_steps)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_distances(
    distances,
    query_rows: list[dict],
    ref_rows: list[dict],
    top_k: int,
) -> tuple[dict, list[dict]]:
    if distances.shape != (len(query_rows), len(ref_rows)):
        raise ValueError("Distance matrix shape does not match query/reference metadata.")
    k = min(max(1, top_k), len(ref_rows))
    hits_1 = hits_k = 0
    reciprocal_ranks = []
    same_distances = []
    other_distances = []
    neighbors = []
    distance_rows = distances.detach().float().cpu()
    for query_index, query in enumerate(query_rows):
        order = sorted(
            range(len(ref_rows)),
            key=lambda ref_index: (float(distance_rows[query_index, ref_index]), ref_index),
        )
        correct_positions = [
            rank
            for rank, ref_index in enumerate(order)
            if ref_rows[ref_index]["target_fact_id"] == query["target_fact_id"]
        ]
        if not correct_positions:
            raise ValueError(f"No matching fact_reference for query {query.get('prompt_id')!r}.")
        first_correct = correct_positions[0]
        hits_1 += int(first_correct == 0)
        hits_k += int(first_correct < k)
        reciprocal_ranks.append(1.0 / (first_correct + 1))
        for ref_index, ref in enumerate(ref_rows):
            value = float(distance_rows[query_index, ref_index])
            if ref["target_fact_id"] == query["target_fact_id"]:
                same_distances.append(value)
            else:
                other_distances.append(value)
        neighbors.append(
            {
                "query_prompt_id": query["prompt_id"],
                "target_fact_id": query["target_fact_id"],
                "correct_rank": first_correct + 1,
                "top_neighbors": [
                    {
                        "rank": rank + 1,
                        "prompt_id": ref_rows[ref_index]["prompt_id"],
                        "target_fact_id": ref_rows[ref_index]["target_fact_id"],
                        "distance": float(distance_rows[query_index, ref_index]),
                        "correct": (
                            ref_rows[ref_index]["target_fact_id"] == query["target_fact_id"]
                        ),
                    }
                    for rank, ref_index in enumerate(order[:k])
                ],
            }
        )
    query_count = len(query_rows)
    report = {
        "queries": query_count,
        "fact_references": len(ref_rows),
        "k": k,
        "top1_accuracy": hits_1 / query_count,
        "top_k_recall": hits_k / query_count,
        "mean_reciprocal_rank": _mean(reciprocal_ranks),
        "mean_same_fact_distance": _mean(same_distances),
        "mean_other_fact_distance": _mean(other_distances),
        "distance_margin": _mean(other_distances) - _mean(same_distances),
        "chance_top1": 1.0 / len(ref_rows),
        "chance_top_k": k / len(ref_rows),
    }
    return report, neighbors


def _load_collected(torch, collect_dir: Path, metadata: list[dict]):
    loaded = []
    expected_layers = None
    for row in metadata:
        tensors = torch.load(collect_dir / row["tensor_path"], map_location="cpu")
        layer_indices = [int(index) for index in tensors["layer_indices"]]
        if expected_layers is None:
            expected_layers = layer_indices
        elif layer_indices != expected_layers:
            raise RuntimeError("Collected rows do not share the same layer selection.")
        loaded.append((row, tensors))
    return expected_layers or [], loaded


def analyze_prompt_free_indices(config: PromptFreeAnalysisConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Prompt-free analysis requires torch.") from exc

    metadata = _read_jsonl(config.collect_dir / "metadata.jsonl")
    _validate_prompt_free_rows(metadata)
    layer_indices, loaded = _load_collected(torch, config.collect_dir, metadata)
    refs = [(row, tensors) for row, tensors in loaded if row["condition_id"] == "fact_reference"]
    queries = [(row, tensors) for row, tensors in loaded if row["condition_id"] == "question_query"]
    if not refs or not queries:
        raise ValueError("Collection must include fact_reference and question_query rows.")
    ref_rows = [row for row, _tensors in refs]
    query_rows = [row for row, _tensors in queries]
    ref_targets = {row["target_fact_id"] for row in ref_rows}
    missing = sorted({row["target_fact_id"] for row in query_rows} - ref_targets)
    if missing:
        raise ValueError(f"Queries have no reference entry for target facts: {missing[:5]}")

    device = _analysis_device(torch, config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "collect_dir": str(config.collect_dir),
                "output_dir": str(config.output_dir),
                "resolved_device": str(device),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reports = []
    neighbor_rows = []
    with torch.no_grad():
        for stored_layer, actual_layer in enumerate(layer_indices):
            momentum_queries = torch.stack(
                [tensors["momentum"][stored_layer] for _row, tensors in queries]
            ).to(device)
            momentum_refs = torch.stack(
                [tensors["momentum"][stored_layer] for _row, tensors in refs]
            ).to(device)
            state_queries = torch.stack(
                [tensors["state_mean"][stored_layer] for _row, tensors in queries]
            ).to(device)
            state_refs = torch.stack(
                [tensors["state_mean"][stored_layer] for _row, tensors in refs]
            ).to(device)
            q_queries = torch.stack(
                [tensors["q_mean"][stored_layer] for _row, tensors in queries]
            ).to(device)
            q_refs = torch.stack(
                [tensors["q_mean"][stored_layer] for _row, tensors in refs]
            ).to(device)

            method_distances = {
                "momentum_dtw": _batched_dtw_distance_matrix(
                    torch,
                    momentum_queries,
                    momentum_refs,
                    config.dtw_window,
                ),
                "mean_state_cosine": _cosine_distance_matrix(torch, state_queries, state_refs),
                "mean_q_cosine": _cosine_distance_matrix(torch, q_queries, q_refs),
            }
            for method, distances in method_distances.items():
                report, neighbors = _evaluate_distances(
                    distances,
                    query_rows,
                    ref_rows,
                    config.top_k,
                )
                reports.append({"method": method, "layer": actual_layer, **report})
                neighbor_rows.extend(
                    {"method": method, "layer": actual_layer, **neighbor}
                    for neighbor in neighbors
                )
            del method_distances
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "layer_report.jsonl", reports)
    _write_jsonl(config.output_dir / "neighbors.jsonl", neighbor_rows)
    best = {}
    for method in INDEX_METHODS:
        candidates = [row for row in reports if row["method"] == method]
        winner = max(
            candidates,
            key=lambda row: (
                row["top1_accuracy"],
                row["top_k_recall"],
                row["mean_reciprocal_rank"],
                row["distance_margin"],
            ),
        )
        best[method] = winner
    manifest_path = config.collect_dir / "index_manifest.json"
    collection_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    summary = {
        "prompt_free": True,
        "training_free": True,
        "generated_tokens_used": False,
        "query_count": len(query_rows),
        "reference_count": len(ref_rows),
        "layers": layer_indices,
        "top_k_requested": config.top_k,
        "dtw_window": config.dtw_window,
        "analysis_device": str(device),
        "collection_manifest": collection_manifest,
        "best_by_method": best,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config.output_dir


def run_prompt_free_pipeline(config: PromptFreePipelineConfig) -> Path:
    if config.reuse_collection is not None:
        collect_dir = config.reuse_collection
    else:
        if not config.model_name:
            raise ValueError("--model-name is required unless --reuse-collection is used.")
        dataset_path = config.dataset_path or config.output_dir / "data" / "prompt_free_pairs.jsonl"
        if config.dataset_path is None:
            build_dataset(
                DataConfig(
                    output_path=dataset_path,
                    max_facts=config.max_facts,
                    seed=config.seed,
                    source="convomem",
                    max_context_turns=config.max_context_turns,
                    convomem_root=config.convomem_root,
                    convomem_chat_template=config.chat_template,
                    convomem_knowledge_prompt=False,
                )
            )
        collect_dir = config.output_dir / "collect"
        collect_prompt_free_indices(
            PromptFreeCollectConfig(
                model_name=config.model_name,
                dataset_path=dataset_path,
                output_dir=collect_dir,
                dtype=config.dtype,
                device=config.device,
                layers=config.layers,
                state_target=config.state_target,
                trajectory_points=config.trajectory_points,
                projection_dim=config.projection_dim,
                projection_seed=config.projection_seed,
            )
        )
    return analyze_prompt_free_indices(
        PromptFreeAnalysisConfig(
            collect_dir=collect_dir,
            output_dir=config.output_dir / "analysis",
            top_k=config.top_k,
            dtw_window=config.dtw_window,
            device=config.analysis_device,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prompt-free, training-free user-input KNN retrieval with momentum DTW "
            "and mean-state/mean-q baselines."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--reuse-collection", type=Path)
    parser.add_argument("--max-facts", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-context-turns", type=int, default=48)
    parser.add_argument("--convomem-root", type=Path)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection such as 'all', '8', or '8,16,24-31'.",
    )
    parser.add_argument("--state-target", choices=STATE_TARGETS, default="block_output")
    parser.add_argument("--trajectory-points", type=int, default=24)
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
        help="Signed random-projection width; 0 disables feature projection.",
    )
    parser.add_argument("--projection-seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtw-window", type=int)
    parser.add_argument(
        "--analysis-device",
        default="auto",
        help="DTW analysis device: auto, cpu, cuda, or an explicit torch device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_prompt_free_pipeline(
        PromptFreePipelineConfig(
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_facts=args.max_facts,
            seed=args.seed,
            max_context_turns=args.max_context_turns,
            convomem_root=args.convomem_root,
            chat_template=args.chat_template,
            dataset_path=args.dataset_path,
            reuse_collection=args.reuse_collection,
            dtype=args.dtype,
            device=args.device,
            layers=args.layers,
            state_target=args.state_target,
            trajectory_points=args.trajectory_points,
            projection_dim=args.projection_dim,
            projection_seed=args.projection_seed,
            top_k=args.top_k,
            dtw_window=args.dtw_window,
            analysis_device=args.analysis_device,
        )
    )
    print(f"Wrote prompt-free index analysis to {output}")


if __name__ == "__main__":
    main()
