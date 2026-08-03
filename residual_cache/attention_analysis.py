from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean

from .answer_block_cache import (
    _cache_seq_len,
    _cat_block_caches,
    _encode_chat_prompt,
    _first_line_end_token_index,
    _forward,
    _generation_stop_token_ids,
    _make_prefix_causal_mask,
    _query_prompt,
    _token_f1,
    _transformer_layers,
)
from .residual_collect import CollectConfig, _load_model


@dataclass(frozen=True)
class AttentionAnalysisConfig:
    replay_dir: Path
    output_dir: Path
    limit: int | None = 4
    layers: tuple[int, ...] | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _add_range(regions: dict[str, list[tuple[int, int]]], name: str, start: int, end: int) -> None:
    if end > start:
        regions[name].append((start, end))


def _prefix_region_layout(
    tokenizer,
    selected_metadata: list[dict],
    *,
    filter_fact_line: bool,
    target_fact_id: str,
) -> tuple[dict[str, list[tuple[int, int]]], int]:
    regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cursor = 0
    for rank, metadata in enumerate(selected_metadata):
        token_ids = list(metadata["token_ids"])
        suffix_tokens = int(metadata["suffix_tokens"])
        if not 0 <= suffix_tokens <= len(token_ids):
            raise ValueError(
                f"Block {metadata['block_id']} has invalid suffix_tokens={suffix_tokens}."
            )
        generated_ids = token_ids[suffix_tokens:]
        line_end = _first_line_end_token_index(tokenizer, generated_ids)
        body_start = suffix_tokens + line_end + 1

        if filter_fact_line:
            effective_length = suffix_tokens + len(token_ids) - body_start
            suffix_range = (cursor, cursor + suffix_tokens)
            fact_range = (cursor + suffix_tokens, cursor + suffix_tokens)
            body_range = (cursor + suffix_tokens, cursor + effective_length)
        else:
            effective_length = len(token_ids)
            suffix_range = (cursor, cursor + suffix_tokens)
            fact_range = (cursor + suffix_tokens, cursor + body_start)
            body_range = (cursor + body_start, cursor + effective_length)

        block_range = (cursor, cursor + effective_length)
        block_kind = (
            "gold"
            if metadata["target_fact_id"] == target_fact_id
            else "distractor"
        )
        _add_range(regions, "history", *block_range)
        _add_range(regions, "history_suffix", *suffix_range)
        _add_range(regions, "history_fact", *fact_range)
        _add_range(regions, "history_body", *body_range)
        _add_range(regions, f"rank_{rank}", *block_range)
        _add_range(regions, f"rank_{rank}_body", *body_range)
        _add_range(regions, f"{block_kind}_block", *block_range)
        _add_range(regions, f"{block_kind}_body", *body_range)
        cursor += effective_length
    return dict(regions), cursor


class _AttentionRecorder:
    def __init__(
        self,
        *,
        torch,
        model,
        layer_indexes: tuple[int, ...],
        prefix_regions: dict[str, list[tuple[int, int]]],
        prefix_len: int,
        prompt_len: int,
    ):
        self.torch = torch
        self.model = model
        self.layer_indexes = layer_indexes
        self.prefix_regions = prefix_regions
        self.prefix_len = prefix_len
        self.prompt_len = prompt_len
        self.events: dict[int, list[dict]] = {
            index: [] for index in layer_indexes
        }
        self.handles = []

    def __enter__(self):
        layers = _transformer_layers(self.model)
        for layer_index in self.layer_indexes:
            attention = getattr(layers[layer_index], "self_attn", None)
            if attention is None:
                raise RuntimeError(f"Layer {layer_index} has no self_attn module.")

            def save_attention(
                _module,
                _inputs,
                output,
                *,
                captured_layer=layer_index,
            ):
                if (
                    not isinstance(output, tuple)
                    or len(output) < 2
                    or output[1] is None
                ):
                    raise RuntimeError(
                        "Attention backend did not return weights. "
                        "Run attention analysis with eager attention."
                    )
                weights = output[1]
                if weights.ndim != 4:
                    raise RuntimeError(
                        f"Layer {captured_layer} returned attention with "
                        f"shape {tuple(weights.shape)}."
                    )
                last_query = weights[0, :, -1, :].detach().float().cpu()
                key_len = int(last_query.shape[-1])
                dynamic_regions = dict(self.prefix_regions)
                prompt_end = min(
                    self.prefix_len + self.prompt_len,
                    key_len,
                )
                dynamic_regions["query_prompt"] = [
                    (self.prefix_len, prompt_end)
                ]
                dynamic_regions["generated_so_far"] = (
                    [(prompt_end, key_len)]
                    if key_len > prompt_end
                    else []
                )
                event_regions = {}
                for name, ranges in dynamic_regions.items():
                    valid_ranges = [
                        (max(0, start), min(key_len, end))
                        for start, end in ranges
                        if min(key_len, end) > max(0, start)
                    ]
                    length = sum(
                        end - start for start, end in valid_ranges
                    )
                    mass = self.torch.zeros(last_query.shape[0])
                    for start, end in valid_ranges:
                        mass += last_query[:, start:end].sum(dim=-1)
                    event_regions[name] = {
                        "length": length,
                        "head_mass": mass.tolist(),
                    }
                self.events[captured_layer].append(
                    {"key_len": key_len, "regions": event_regions}
                )

            self.handles.append(
                attention.register_forward_hook(save_attention)
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _force_eager_attention(model) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "eager"
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = "eager"
    for layer in _transformer_layers(model):
        attention = getattr(layer, "self_attn", None)
        attention_config = getattr(attention, "config", None)
        if attention_config is not None:
            attention_config._attn_implementation = "eager"


def _phase_target_indexes(
    tokenizer,
    token_ids: list[int],
) -> dict[str, list[int]]:
    stop_ids = _generation_stop_token_ids(tokenizer)
    non_stop = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id not in stop_ids
    ]
    try:
        line_end = _first_line_end_token_index(tokenizer, token_ids)
    except RuntimeError:
        line_end = len(token_ids) - 1
    fact_line = [index for index in non_stop if index <= line_end]
    answer_body = [index for index in non_stop if index > line_end]
    return {
        "all_output": non_stop,
        "fact_line": fact_line,
        "answer_body": answer_body,
        "first_body_token": answer_body[:1],
    }


def _summarize_events(
    events: list[dict],
    event_indexes: list[int],
) -> dict:
    selected = [
        events[index]
        for index in event_indexes
        if index < len(events)
    ]
    if not selected:
        return {"events": 0, "regions": {}}
    region_names = sorted(
        set().union(
            *(event["regions"].keys() for event in selected)
        )
    )
    summary = {}
    for region_name in region_names:
        event_head_mass = []
        event_head_enrichment = []
        lengths = []
        for event in selected:
            region = event["regions"].get(region_name)
            if region is None:
                continue
            head_mass = region["head_mass"]
            length = int(region["length"])
            lengths.append(length)
            event_head_mass.append(head_mass)
            if length:
                uniform_mass = length / event["key_len"]
                event_head_enrichment.append(
                    [mass / uniform_mass for mass in head_mass]
                )
        if not event_head_mass:
            continue
        head_count = len(event_head_mass[0])
        mean_by_head = [
            mean(masses[head] for masses in event_head_mass)
            for head in range(head_count)
        ]
        enrichment_by_head = (
            [
                mean(values[head] for values in event_head_enrichment)
                for head in range(head_count)
            ]
            if event_head_enrichment
            else [0.0] * head_count
        )
        summary[region_name] = {
            "mean_tokens": mean(lengths),
            "mean_mass": mean(mean_by_head),
            "max_head_mass": max(mean_by_head),
            "mean_enrichment": mean(enrichment_by_head),
            "max_head_enrichment": max(enrichment_by_head),
            "heads_over_2x_uniform": sum(
                value >= 2.0 for value in enrichment_by_head
            ),
            "per_head_mass": mean_by_head,
        }
    return {"events": len(selected), "regions": summary}


def _trace_saved_replay(
    *,
    torch,
    tokenizer,
    model,
    prompt_ids: list[int],
    generated_ids: list[int],
    start_position: int,
    prefix_past,
    custom_prefix_mask: bool,
    recorder: _AttentionRecorder,
) -> dict:
    device = next(model.parameters()).device
    prefix_len = _cache_seq_len(prefix_past)
    prompt_input = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )
    prompt_mask = None
    if custom_prefix_mask and prefix_len:
        prompt_mask = _make_prefix_causal_mask(
            torch,
            model=model,
            q_len=len(prompt_ids),
            prefix_len=prefix_len,
            current_kv_len=len(prompt_ids),
            current_query_start=0,
        )
    prediction_matches = []
    with recorder:
        prompt_out, _ = _forward(
            torch,
            model,
            prompt_input,
            start_position,
            past_key_values=prefix_past,
            attention_mask=prompt_mask,
        )
        past = prompt_out.past_key_values
        if generated_ids:
            predicted = int(
                prompt_out.logits[:, -1, :].argmax(dim=-1).item()
            )
            prediction_matches.append(predicted == generated_ids[0])

        for target_index in range(1, len(generated_ids)):
            previous_token = generated_ids[target_index - 1]
            token_input = torch.tensor(
                [[previous_token]],
                dtype=torch.long,
                device=device,
            )
            token_mask = None
            if custom_prefix_mask and prefix_len:
                token_mask = _make_prefix_causal_mask(
                    torch,
                    model=model,
                    q_len=1,
                    prefix_len=prefix_len,
                    current_kv_len=len(prompt_ids) + target_index,
                    current_query_start=len(prompt_ids) + target_index - 1,
                )
            token_out, _ = _forward(
                torch,
                model,
                token_input,
                start_position + len(prompt_ids) + target_index - 1,
                past_key_values=past,
                attention_mask=token_mask,
            )
            past = token_out.past_key_values
            predicted = int(
                token_out.logits[:, -1, :].argmax(dim=-1).item()
            )
            prediction_matches.append(
                predicted == generated_ids[target_index]
            )

    expected_events = len(generated_ids)
    for layer_index, events in recorder.events.items():
        if len(events) != expected_events:
            raise RuntimeError(
                f"Layer {layer_index} captured {len(events)} events; "
                f"expected {expected_events}."
            )
    return {
        "prediction_matches": prediction_matches,
        "prediction_match_rate": (
            sum(prediction_matches) / len(prediction_matches)
            if prediction_matches
            else None
        ),
    }


def _layer_type(model, layer_index: int) -> str:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    return layer_types[layer_index] if layer_types else "unknown"


def _aggregate(
    rows: list[dict],
    layer_indexes: tuple[int, ...],
) -> dict:
    layer_summary = {}
    for layer_index in layer_indexes:
        layer_rows = [
            row["layers"][str(layer_index)]
            for row in rows
        ]
        non_empty = [
            layer_row
            for layer_row in layer_rows
            if layer_row["answer_body"]["events"]
        ]
        region_names = sorted(
            set().union(
                *(
                    layer_row["answer_body"]["regions"].keys()
                    for layer_row in non_empty
                )
            )
            if non_empty
            else set()
        )
        regions = {}
        for region_name in region_names:
            metrics = [
                layer_row["answer_body"]["regions"][region_name]
                for layer_row in layer_rows
                if region_name
                in layer_row["answer_body"]["regions"]
            ]
            regions[region_name] = {
                key: mean(metric[key] for metric in metrics)
                for key in (
                    "mean_tokens",
                    "mean_mass",
                    "max_head_mass",
                    "mean_enrichment",
                    "max_head_enrichment",
                    "heads_over_2x_uniform",
                )
            }
        layer_summary[str(layer_index)] = {
            "layer_type": layer_rows[0]["layer_type"],
            "cases_with_body": len(non_empty),
            "regions": regions,
        }
    match_rates = [
        row["prediction_match_rate"]
        for row in rows
        if row["prediction_match_rate"] is not None
    ]
    return {
        "cases": len(rows),
        "mean_prediction_match_rate": (
            mean(match_rates) if match_rates else None
        ),
        "mean_gold_answer_body_f1": mean(
            row["gold_answer_body_f1"] for row in rows
        ),
        "top_hit_rate": mean(float(row["top_hit"]) for row in rows),
        "layers": layer_summary,
    }


def run_attention_analysis(
    config: AttentionAnalysisConfig,
) -> Path:
    replay_config = _read_json(config.replay_dir / "config.json")
    test_rows = _read_jsonl(
        config.replay_dir / "test_results.jsonl"
    )
    episode_rows = _read_jsonl(config.replay_dir / "episodes.jsonl")
    if config.limit is not None:
        test_rows = test_rows[: config.limit]
    if not test_rows:
        raise ValueError("No replay rows selected for attention analysis.")

    history_dir = Path(replay_config["reuse_history_dir"])
    if not history_dir.is_absolute():
        history_dir = (config.replay_dir / history_dir).resolve()
    history_metadata = _read_jsonl(
        history_dir / "history_blocks.jsonl"
    )
    metadata_by_id = {
        row["block_id"]: row for row in history_metadata
    }
    episodes_by_index = {
        index: row for index, row in enumerate(episode_rows)
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    saved_config = {
        **asdict(config),
        "replay_dir": str(config.replay_dir),
        "output_dir": str(config.output_dir),
        "layers": (
            list(config.layers)
            if config.layers is not None
            else None
        ),
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(saved_config, indent=2),
        encoding="utf-8",
    )

    torch, tokenizer, model = _load_model(
        CollectConfig(
            model_name=replay_config["model_name"],
            dataset_path=Path("unused.jsonl"),
            output_dir=config.output_dir,
            max_new_tokens=int(replay_config["max_new_tokens"]),
            dtype=replay_config["dtype"],
            device=replay_config["device"],
        )
    )
    _force_eager_attention(model)
    layer_count = len(_transformer_layers(model))
    layer_indexes = (
        tuple(range(layer_count))
        if config.layers is None
        else config.layers
    )
    if any(
        index < 0 or index >= layer_count
        for index in layer_indexes
    ):
        raise ValueError(
            f"Attention layer outside [0, {layer_count - 1}]."
        )

    filter_fact_line = bool(
        replay_config.get("filter_fact_line_from_blocks")
    )
    output_rows = []
    try:
        for test_row in test_rows:
            episode_index = int(test_row["episode_index"])
            episode = episodes_by_index[episode_index]
            selected_metadata = [
                metadata_by_id[block_id]
                for block_id in test_row["top_block_ids"]
            ]
            block_paths = [
                history_dir / metadata["block_path"]
                for metadata in selected_metadata
            ]
            prefix_past = _cat_block_caches(
                torch,
                block_paths,
                next(model.parameters()).device,
                tokenizer=tokenizer,
                block_metadata=selected_metadata,
                filter_fact_line=filter_fact_line,
            )
            regions, expected_prefix_len = _prefix_region_layout(
                tokenizer,
                selected_metadata,
                filter_fact_line=filter_fact_line,
                target_fact_id=test_row["target_fact_id"],
            )
            prefix_len = _cache_seq_len(prefix_past)
            if prefix_len != expected_prefix_len:
                raise RuntimeError(
                    "Prefix layout/cache mismatch: "
                    f"layout={expected_prefix_len}, "
                    f"cache={prefix_len}."
                )
            prompt = _encode_chat_prompt(
                tokenizer,
                _query_prompt(episode),
            )
            generated_ids = list(test_row["replay_token_ids"])
            recorder = _AttentionRecorder(
                torch=torch,
                model=model,
                layer_indexes=layer_indexes,
                prefix_regions=regions,
                prefix_len=prefix_len,
                prompt_len=len(prompt.ids),
            )
            trace = _trace_saved_replay(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                prompt_ids=prompt.ids,
                generated_ids=generated_ids,
                start_position=int(test_row["test_start_position"]),
                prefix_past=prefix_past,
                custom_prefix_mask=bool(
                    replay_config["custom_positioned_replay"]
                ),
                recorder=recorder,
            )
            phases = _phase_target_indexes(
                tokenizer,
                generated_ids,
            )
            layer_rows = {}
            for layer_index in layer_indexes:
                layer_rows[str(layer_index)] = {
                    "layer_type": _layer_type(
                        model,
                        layer_index,
                    ),
                    **{
                        phase: _summarize_events(
                            recorder.events[layer_index],
                            event_indexes,
                        )
                        for phase, event_indexes in phases.items()
                    },
                }
            replay_body = (
                test_row["replay_text"].split("\n", 1)[1]
                if "\n" in test_row["replay_text"]
                else ""
            )
            output_rows.append(
                {
                    "episode_index": episode_index,
                    "target_fact_id": test_row["target_fact_id"],
                    "top_hit": bool(test_row["top_hit"]),
                    "top_block_ids": test_row["top_block_ids"],
                    "filter_fact_line_from_blocks": (
                        filter_fact_line
                    ),
                    "prefix_tokens": prefix_len,
                    "query_prompt_tokens": len(prompt.ids),
                    "replay_generated_tokens": len(
                        generated_ids
                    ),
                    "answer_body_events": len(
                        phases["answer_body"]
                    ),
                    "gold_answer_body_f1": _token_f1(
                        replay_body,
                        test_row["gold_answer"],
                    ),
                    **trace,
                    "layers": layer_rows,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(
        config.output_dir / "attention_traces.jsonl",
        output_rows,
    )
    summary = _aggregate(output_rows, layer_indexes)
    summary.update(
        {
            "replay_dir": str(config.replay_dir),
            "filter_fact_line_from_blocks": filter_fact_line,
            "analyzed_layers": list(layer_indexes),
            "attention_backend": "eager",
            "trajectory": "teacher_forced_saved_replay",
        }
    )
    (
        config.output_dir / "attention_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return config.output_dir


def _parse_layers(value: str) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    indexes = tuple(
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    )
    if not indexes:
        raise argparse.ArgumentTypeError(
            "layers must be 'all' or a comma-separated list."
        )
    return indexes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace per-layer attention over recalled answer-block "
            "regions."
        )
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        default=None,
    )
    args = parser.parse_args()
    output_dir = run_attention_analysis(
        AttentionAnalysisConfig(
            replay_dir=args.replay_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            layers=args.layers,
        )
    )
    print(output_dir)


if __name__ == "__main__":
    main()
