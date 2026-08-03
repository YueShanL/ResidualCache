from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import inspect
import json
import math
from pathlib import Path
from typing import Iterable

from residual_cache.data_process import (
    append_convomem_index_instruction,
    build_convomem_rows,
    refresh_convomem_knowledge_instruction,
)
from residual_cache.residual_collect import (
    CollectConfig,
    GENERATED_PREFIX,
    _generation_stop_token_ids,
    _hidden,
    _load_model,
    _transformer_layers,
)

INDEX_PREFIX_MAX_NEW_TOKENS = 8
DUAL_PIPELINE_MODE = "dual_main_natural_index_instruction_v1"


@dataclass(frozen=True)
class AnswerBlockConfig:
    model_name: str
    output_dir: Path
    max_facts: int = 24
    seed: int = 13
    block_size: int = 8
    top_n_blocks: int = 8
    max_new_tokens: int = 24
    index_layer: int = 40
    dtype: str = "auto"
    device: str = "auto"
    max_context_turns: int = 48
    convomem_root: Path | None = None
    semantic_model: str | None = None
    custom_positioned_replay: bool = False
    test_position_override: int | None = None
    filter_fact_line_from_blocks: bool = False


@dataclass(frozen=True)
class ChatPromptTokens:
    ids: list[int]
    anchor_index: int
    suffix_start: int


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _clone_cache(cache):
    return copy.deepcopy(cache) if cache is not None else None


def _dual_branch_prompts(row: dict) -> tuple[str, str]:
    main_content = row["history_prompt"]
    return (
        main_content,
        append_convomem_index_instruction(main_content),
    )


def _dual_query_prompts(row: dict) -> tuple[str, str]:
    main_content = row["query_prompt"]
    return (
        main_content,
        append_convomem_index_instruction(main_content),
    )


def _encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("Prompt tokenized to nothing.")
    return ids


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    for index in range(len(values) - len(needle), -1, -1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def _encode_chat_prompt(tokenizer, user_content: str) -> ChatPromptTokens:
    content = user_content.strip()
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"{content}\nAnswer:"

    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("Prompt tokenized to nothing.")

    anchor_index = -1
    content_start = rendered.rfind(content)
    if content_start >= 0:
        content_end = content_start + len(content)
        try:
            encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            for index, (start, end) in enumerate(encoded.offset_mapping):
                if end > content_start and start < content_end and end <= content_end:
                    anchor_index = index
        except (NotImplementedError, TypeError, ValueError):
            anchor_index = -1

    if anchor_index < 0:
        content_ids = tokenizer(content, add_special_tokens=False).input_ids
        content_token_start = _find_subsequence(ids, content_ids)
        if content_token_start >= 0:
            anchor_index = content_token_start + len(content_ids) - 1
    if anchor_index < 0:
        anchor_index = len(ids) - 1

    return ChatPromptTokens(ids=ids, anchor_index=anchor_index, suffix_start=anchor_index + 1)


def _forward(
    torch,
    model,
    input_ids,
    start_position: int,
    past_key_values=None,
    capture_layer: int | None = None,
    capture_token_index: int = -1,
    attention_mask=None,
):
    captured = {}
    handle = None
    if capture_layer is not None:
        layers = _transformer_layers(model)

        def save_output(_module, _inputs, output):
            hidden = _hidden(output)
            token_index = capture_token_index if capture_token_index >= 0 else hidden.shape[1] + capture_token_index
            captured["index"] = hidden[0, token_index, :].detach().float().cpu()

        handle = layers[capture_layer].register_forward_hook(save_output)

    positions = torch.arange(start_position, start_position + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    kwargs = {
        "input_ids": input_ids,
        "position_ids": positions,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if "cache_position" in inspect.signature(model.forward).parameters:
        past_len = _cache_seq_len(past_key_values)
        kwargs["cache_position"] = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
    try:
        with torch.no_grad():
            output = model(**kwargs)
    finally:
        if handle is not None:
            handle.remove()
    return output, captured.get("index")


def _legacy_cache(cache):
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def _cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(_legacy_cache(cache)[0][0].shape[2])


def _cache_from_legacy(cache):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return cache
    return DynamicCache(cache)


def _slice_cache(cache, start: int, end: int):
    if start < 0 or end < start:
        raise ValueError(f"Invalid cache slice [{start}:{end}].")

    expected_length = end - start
    cache_layers = getattr(cache, "layers", None)
    sliced = []
    for layer_index, layer_cache in enumerate(_legacy_cache(cache)):
        key, value = layer_cache[:2]
        physical_length = int(key.shape[2])
        if int(value.shape[2]) != physical_length:
            raise RuntimeError(
                f"Cache layer {layer_index} has mismatched K/V lengths: "
                f"K={physical_length}, V={int(value.shape[2])}."
            )

        logical_length = physical_length
        if cache_layers is not None:
            logical_length = int(cache_layers[layer_index].get_seq_length())
        physical_offset = logical_length - physical_length
        physical_start = start - physical_offset
        physical_end = end - physical_offset
        if physical_start < 0 or physical_end > physical_length:
            raise RuntimeError(
                f"Cache slice [{start}:{end}] is unavailable in layer {layer_index}: "
                f"logical_length={logical_length}, physical_range="
                f"[{physical_offset}:{physical_offset + physical_length}]."
            )

        sliced_key = key[:, :, physical_start:physical_end, :].detach().cpu()
        sliced_value = value[:, :, physical_start:physical_end, :].detach().cpu()
        if int(sliced_key.shape[2]) != expected_length or int(sliced_value.shape[2]) != expected_length:
            raise RuntimeError(
                f"Cache layer {layer_index} produced an invalid slice length: "
                f"expected={expected_length}, K={int(sliced_key.shape[2])}, "
                f"V={int(sliced_value.shape[2])}."
            )
        sliced.append((sliced_key, sliced_value))
    return tuple(sliced)


def _validate_legacy_cache(cache, *, expected_length: int | None = None, context: str) -> int:
    layer_lengths = []
    for layer_index, layer_cache in enumerate(cache):
        key, value = layer_cache[:2]
        key_length = int(key.shape[2])
        value_length = int(value.shape[2])
        if key_length != value_length:
            raise RuntimeError(
                f"{context} layer {layer_index} has mismatched K/V lengths: "
                f"K={key_length}, V={value_length}."
            )
        layer_lengths.append(key_length)

    if not layer_lengths:
        raise RuntimeError(f"{context} contains no cache layers.")
    if len(set(layer_lengths)) != 1:
        raise RuntimeError(f"{context} has inconsistent layer lengths: {layer_lengths}.")

    actual_length = layer_lengths[0]
    if actual_length <= 0:
        raise RuntimeError(f"{context} has no cached tokens.")
    if expected_length is not None and actual_length != expected_length:
        raise RuntimeError(
            f"{context} has length {actual_length}, expected {expected_length}."
        )
    return actual_length


def _first_line_end_token_index(tokenizer, token_ids: list[int]) -> int:
    for index in range(len(token_ids)):
        decoded = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True)
        if "\n" in decoded:
            return index
    raise RuntimeError(f"Generated block has no complete first line: {_decode(tokenizer, token_ids)!r}.")


def _filter_fact_line_cache(torch, tokenizer, cache, token_ids: list[int], suffix_tokens: int):
    if not 0 <= suffix_tokens <= len(token_ids):
        raise ValueError(f"suffix_tokens={suffix_tokens} is outside block length {len(token_ids)}.")
    generated_ids = token_ids[suffix_tokens:]
    line_end = _first_line_end_token_index(tokenizer, generated_ids)
    fact_text = tokenizer.decode(generated_ids[: line_end + 1], skip_special_tokens=True).strip()
    if not fact_text.startswith(GENERATED_PREFIX):
        raise RuntimeError(f"Expected generated FACT line, found {fact_text!r}.")

    body_start = suffix_tokens + line_end + 1
    keep_ranges = []
    if suffix_tokens:
        keep_ranges.append((0, suffix_tokens))
    if body_start < len(token_ids):
        keep_ranges.append((body_start, len(token_ids)))
    if not keep_ranges:
        raise RuntimeError("Filtering the FACT line removed the entire history block.")

    filtered = []
    for layer_cache in _legacy_cache(cache):
        key, value = layer_cache[:2]
        filtered_key = torch.cat([key[:, :, start:end, :] for start, end in keep_ranges], dim=2)
        filtered_value = torch.cat([value[:, :, start:end, :] for start, end in keep_ranges], dim=2)
        filtered.append((filtered_key, filtered_value))
    filtered = tuple(filtered)
    expected_length = sum(end - start for start, end in keep_ranges)
    _validate_legacy_cache(filtered, expected_length=expected_length, context="FACT-filtered history block")
    return filtered


def _cat_block_caches(
    torch,
    block_paths: list[Path],
    device,
    *,
    tokenizer=None,
    block_metadata: list[dict] | None = None,
    filter_fact_line: bool = False,
):
    if not block_paths:
        return None
    if filter_fact_line and (tokenizer is None or block_metadata is None):
        raise ValueError("FACT-line filtering requires tokenizer and block metadata.")
    if block_metadata is not None and len(block_metadata) != len(block_paths):
        raise ValueError("Block metadata/path counts do not match.")
    blocks = []
    for block_index, path in enumerate(block_paths):
        payload = torch.load(path, map_location="cpu")
        block = payload["past_key_values"]
        _validate_legacy_cache(block, context=f"History block {path}")
        if filter_fact_line:
            metadata = block_metadata[block_index]
            block = _filter_fact_line_cache(
                torch,
                tokenizer,
                block,
                list(payload["token_ids"]),
                int(metadata["suffix_tokens"]),
            )
        blocks.append(block)
    layer_count = len(blocks[0])
    if any(len(block) != layer_count for block in blocks):
        raise RuntimeError("History blocks have inconsistent layer counts.")
    merged = []
    for layer_index in range(layer_count):
        keys = [block[layer_index][0].to(device) for block in blocks]
        values = [block[layer_index][1].to(device) for block in blocks]
        merged.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    merged = tuple(merged)
    _validate_legacy_cache(merged, context="Merged history cache")
    return _cache_from_legacy(merged)


def _attention_mask_key(torch, model) -> str:
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        return layer_types[0]
    text_config = getattr(config, "text_config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return layer_types[0]
    return "full_attention"


def _make_prefix_causal_mask(
    torch,
    *,
    model,
    q_len: int,
    prefix_len: int,
    current_kv_len: int,
    current_query_start: int,
):
    """Build an additive 4D mask for [sparse prefix KV] + [current AR KV]."""
    parameter = next(model.parameters())
    dtype = parameter.dtype
    device = parameter.device
    kv_len = prefix_len + current_kv_len
    mask = torch.zeros((1, 1, q_len, kv_len), dtype=dtype, device=device)
    if current_kv_len:
        key_positions = torch.arange(current_kv_len, device=device)
        query_positions = torch.arange(current_query_start, current_query_start + q_len, device=device)
        disallowed = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        if disallowed.any():
            mask[:, :, :, prefix_len:] = mask[:, :, :, prefix_len:].masked_fill(disallowed.view(1, 1, q_len, current_kv_len), torch.finfo(dtype).min)
    key = _attention_mask_key(torch, model)
    return {"full_attention": mask, "sliding_attention": mask, key: mask}


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _decode_raw(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False).strip()


def _generate(
    torch,
    tokenizer,
    model,
    prompt_ids: list[int],
    *,
    start_position: int,
    max_new_tokens: int,
    index_layer: int,
    prefix_past=None,
    capture_token_index: int = -1,
    capture_generated_prefix: str | None = None,
    custom_prefix_mask: bool = False,
    capture_cache_prompt_start: int | None = None,
    capture_cache_generated_tokens: int | None = None,
):
    if (capture_cache_prompt_start is None) != (capture_cache_generated_tokens is None):
        raise ValueError(
            "capture_cache_prompt_start and capture_cache_generated_tokens must be provided together."
        )
    if capture_cache_prompt_start is not None and not 0 <= capture_cache_prompt_start <= len(prompt_ids):
        raise ValueError(
            f"capture_cache_prompt_start={capture_cache_prompt_start} is outside the prompt."
        )
    if capture_cache_generated_tokens is not None and capture_cache_generated_tokens < 0:
        raise ValueError("capture_cache_generated_tokens must be non-negative.")

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prefix_len = _cache_seq_len(prefix_past)
    prompt_attention_mask = None
    if custom_prefix_mask and prefix_len:
        prompt_attention_mask = _make_prefix_causal_mask(
            torch,
            model=model,
            q_len=len(prompt_ids),
            prefix_len=prefix_len,
            current_kv_len=len(prompt_ids),
            current_query_start=0,
        )
    prompt_out, index_vector = _forward(
        torch,
        model,
        input_ids,
        start_position,
        past_key_values=prefix_past,
        capture_layer=index_layer if capture_generated_prefix is None else None,
        capture_token_index=capture_token_index,
        attention_mask=prompt_attention_mask,
    )
    past = prompt_out.past_key_values
    next_token = int(prompt_out.logits[:, -1, :].argmax(dim=-1).item())
    generated = []
    prompt_cache_start = prefix_len
    prompt_cache_len = len(prompt_ids)
    generated_start = prompt_cache_start + prompt_cache_len
    captured_cache = None
    captured_generated_tokens = None
    index_generated_token_index = None
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    stop_reason = "length" if max_new_tokens else "max_new_tokens_zero"

    if capture_cache_generated_tokens == 0:
        captured_cache = _slice_cache(
            past,
            prompt_cache_start + capture_cache_prompt_start,
            generated_start,
        )
        captured_generated_tokens = 0

    for step in range(max_new_tokens):
        generated.append(next_token)
        capture_generated_index = (
            capture_generated_prefix is not None
            and index_vector is None
            and _decode(tokenizer, generated).lstrip().startswith(capture_generated_prefix)
        )
        token_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        token_position = start_position + prompt_cache_len + step
        token_attention_mask = None
        if custom_prefix_mask and prefix_len:
            token_attention_mask = _make_prefix_causal_mask(
                torch,
                model=model,
                q_len=1,
                prefix_len=prefix_len,
                current_kv_len=prompt_cache_len + step + 1,
                current_query_start=prompt_cache_len + step,
            )
        token_out, generated_index = _forward(
            torch,
            model,
            token_input,
            token_position,
            past_key_values=past,
            capture_layer=index_layer if capture_generated_index else None,
            attention_mask=token_attention_mask,
        )
        if generated_index is not None:
            index_vector = generated_index
            index_generated_token_index = len(generated) - 1
        past = token_out.past_key_values
        if (
            captured_cache is None
            and capture_cache_generated_tokens is not None
            and (
                len(generated) >= capture_cache_generated_tokens
                or next_token in stop_token_ids
            )
        ):
            captured_cache = _slice_cache(
                past,
                prompt_cache_start + capture_cache_prompt_start,
                generated_start + len(generated),
            )
            captured_generated_tokens = len(generated)
        if next_token in stop_token_ids:
            stop_reason = "eos"
            break
        next_token = int(token_out.logits[:, -1, :].argmax(dim=-1).item())

    if captured_cache is None and capture_cache_generated_tokens is not None:
        captured_cache = _slice_cache(
            past,
            prompt_cache_start + capture_cache_prompt_start,
            generated_start + len(generated),
        )
        captured_generated_tokens = len(generated)

    if capture_generated_prefix is not None and index_vector is None:
        raise RuntimeError(
            f"Model did not generate required index prefix {capture_generated_prefix!r}; "
            f"output was {_decode(tokenizer, generated)!r}."
        )

    generated_end = generated_start + len(generated)
    return {
        "index_vector": index_vector,
        "index_generated_prefix": capture_generated_prefix,
        "index_generated_token_index": index_generated_token_index,
        "index_generated_prefix_text": (
            _decode(tokenizer, generated[: index_generated_token_index + 1])
            if index_generated_token_index is not None
            else None
        ),
        "generated_ids": generated,
        "generated_text": _decode(tokenizer, generated),
        "generated_raw_text": _decode_raw(tokenizer, generated),
        "past_key_values": past,
        "captured_cache": captured_cache,
        "captured_generated_tokens": captured_generated_tokens,
        "generated_cache_range": (generated_start, generated_end),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "eos_hit": any(token_id in stop_token_ids for token_id in generated),
        "eos_token_ids": sorted(stop_token_ids),
        "stop_reason": stop_reason,
    }


def _answer_block_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("condition_id") == "question_query"]


def _query_prompt(row: dict) -> str:
    if row.get("knowledge_prompt") and row.get("query_prompt"):
        return refresh_convomem_knowledge_instruction(
            row["query_prompt"]
        )
    return (
        "Answer the memory question directly and concisely. "
        "Do not repeat the question or invent unavailable details.\n\n"
        f"User profile: {row['entity'].split('_', 1)[-1].replace('_', ' ')}\n"
        f"Question: {row['specific_question']}"
    )


def _history_prompt(row: dict) -> str:
    history_prompt = row.get("history_prompt") or row["prompt"]
    if row.get("knowledge_prompt"):
        return refresh_convomem_knowledge_instruction(
            history_prompt
        )
    return (
        "Answer the memory question using only the conversation evidence. "
        "Return only the answer. Do not repeat the evidence or the question.\n\n"
        f"{history_prompt.rstrip()}"
    )


def _token_f1(left: str, right: str) -> float:
    left_tokens = left.lower().split()
    right_tokens = right.lower().split()
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = 0
    remaining = right_tokens.copy()
    for token in left_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _answer_block_end(generated_ids: list[int], block_size: int, eos_token_id: int | None) -> int:
    limit = min(len(generated_ids), block_size)
    if eos_token_id is None:
        return limit
    for index, token_id in enumerate(generated_ids[:limit]):
        if token_id == eos_token_id:
            return index + 1
    return limit


def _unique_token_ratio(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _history_quality(history_text: str, gold_answer: str) -> dict:
    has_prompt_marker = "Question:" in history_text or "Answer:" in history_text
    unique_ratio = _unique_token_ratio(history_text)
    gold_token_f1 = _token_f1(history_text, gold_answer)
    return {
        "history_has_prompt_marker": has_prompt_marker,
        "history_unique_token_ratio": unique_ratio,
        "history_gold_token_f1": gold_token_f1,
        "history_valid_payload": (not has_prompt_marker) and unique_ratio >= 0.25 and gold_token_f1 > 0.0,
    }


def _load_semantic_model(model_name: str | None):
    if not model_name:
        return None
    return _EncoderSemanticTokenEmbedder(model_name)


def _semantic_similarity(model, left: str, right: str) -> float | None:
    if model is None:
        return None
    embeddings = model.encode([left, right], normalize_embeddings=True)
    return float((embeddings[0] * embeddings[1]).sum())


class _EncoderSemanticTokenEmbedder:
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Semantic evaluation requires transformers.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if bool(getattr(self.model.config, "is_decoder", False)):
            raise ValueError(f"Semantic model must be encoder-only, got decoder model: {model_name}")
        architectures = getattr(self.model.config, "architectures", None) or []
        if any("CausalLM" in architecture for architecture in architectures):
            raise ValueError(f"Semantic model must not be decoder-only/CausalLM: {model_name}")
        if self.tokenizer.cls_token_id is None:
            raise ValueError(f"Semantic model tokenizer must expose a CLS/semantic token: {model_name}")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        if normalize_embeddings:
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _history_summary(block_metadata: list[dict]) -> dict:
    return {
        "history_valid_payload_rate": _mean([float(row["history_valid_payload"]) for row in block_metadata]),
        "history_prompt_marker_rate": _mean([float(row["history_has_prompt_marker"]) for row in block_metadata]),
        "history_unique_token_ratio": _mean([row["history_unique_token_ratio"] for row in block_metadata]),
        "history_gold_token_f1": _mean([row["history_gold_token_f1"] for row in block_metadata]),
    }


def _generation_fields(prefix: str, result: dict) -> dict:
    return {
        f"{prefix}_text": result["generated_text"],
        f"{prefix}_raw_text": result["generated_raw_text"],
        f"{prefix}_token_ids": result["generated_ids"],
        f"{prefix}_generated_tokens": result["generated_tokens"],
        f"{prefix}_eos_hit": result["eos_hit"],
        f"{prefix}_stop_reason": result["stop_reason"],
    }


def _write_generation_report(path: Path, rows: list[dict]) -> None:
    lines = ["# Actual Model Generations", ""]
    for row in rows:
        lines.extend(
            [
                f"## Episode {row['episode_index']}",
                "",
                f"- target_fact_id: `{row['target_fact_id']}`",
                f"- top_hit: `{row['top_hit']}`",
                f"- top_block_ids: `{', '.join(row['top_block_ids']) if row['top_block_ids'] else '<none>'}`",
                f"- baseline_stop: `{row['baseline_stop_reason']}`, tokens: `{row['baseline_generated_tokens']}`, eos: `{row['baseline_eos_hit']}`",
                f"- replay_stop: `{row['replay_stop_reason']}`, tokens: `{row['replay_generated_tokens']}`, eos: `{row['replay_eos_hit']}`",
                "",
                "### Baseline Actual Output",
                "",
                row["baseline_raw_text"] or "<empty>",
                "",
                "### Replay Actual Output",
                "",
                row["replay_raw_text"] or "<empty>",
                "",
                "### Query Index-only Output",
                "",
                row["query_index_generated_raw_text"] or "<empty>",
                "",
                "### History Output",
                "",
                row["history_generated_text"] or "<empty>",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_replay_tests(
    torch,
    tokenizer,
    model,
    *,
    rows: list[dict],
    block_metadata: list[dict],
    index_tensor,
    history_dir: Path,
    output_dir: Path,
    test_start_position: int,
    top_n_blocks: int,
    max_new_tokens: int,
    index_layer: int,
    semantic_model_name: str | None,
    custom_positioned_replay: bool = False,
    filter_fact_line_from_blocks: bool = False,
) -> dict:
    if filter_fact_line_from_blocks:
        raise ValueError(
            "Dual-pipeline main blocks contain natural model "
            "output and cannot use FACT-line filtering."
        )
    semantic_model = _load_semantic_model(semantic_model_name)
    test_rows = []
    for episode_index, row in enumerate(rows):
        main_query_content, index_query_content = (
            _dual_query_prompts(row)
        )
        main_prompt_tokens = _encode_chat_prompt(
            tokenizer,
            main_query_content,
        )
        index_prompt_tokens = _encode_chat_prompt(
            tokenizer,
            index_query_content,
        )
        search = _generate(
            torch,
            tokenizer,
            model,
            index_prompt_tokens.ids,
            start_position=test_start_position,
            max_new_tokens=INDEX_PREFIX_MAX_NEW_TOKENS,
            index_layer=index_layer,
            capture_generated_prefix=GENERATED_PREFIX,
        )
        query = torch.nn.functional.normalize(search["index_vector"].float(), dim=0)
        index_norm = torch.nn.functional.normalize(index_tensor.float(), dim=1)
        scores = index_norm @ query
        top_indices = scores.topk(min(top_n_blocks, len(block_metadata))).indices.tolist()
        selected = sorted((block_metadata[index] for index in top_indices), key=lambda item: item["positions"][0])
        prefix_past = _cat_block_caches(
            torch,
            [history_dir / item["block_path"] for item in selected],
            next(model.parameters()).device,
            tokenizer=tokenizer,
            block_metadata=selected,
            filter_fact_line=False,
        )
        baseline = _generate(
            torch,
            tokenizer,
            model,
            main_prompt_tokens.ids,
            start_position=test_start_position,
            max_new_tokens=max_new_tokens,
            index_layer=index_layer,
            capture_token_index=main_prompt_tokens.anchor_index,
        )
        replay = _generate(
            torch,
            tokenizer,
            model,
            main_prompt_tokens.ids,
            start_position=test_start_position,
            max_new_tokens=max_new_tokens,
            index_layer=index_layer,
            prefix_past=prefix_past,
            capture_token_index=main_prompt_tokens.anchor_index,
            custom_prefix_mask=custom_positioned_replay,
        )
        expected_history = next(
            (item["history_generated_text"] for item in block_metadata if item["episode_index"] == episode_index),
            "",
        )
        top_hit = any(item["episode_index"] == episode_index for item in selected)
        test_rows.append(
            {
                "episode_index": episode_index,
                "target_fact_id": row["target_fact_id"],
                "test_start_position": test_start_position,
                "top_hit": top_hit,
                "top_block_ids": [item["block_id"] for item in selected],
                "main_query_content": main_query_content,
                "index_query_content": index_query_content,
                "query_index_generated_prefix": search["index_generated_prefix_text"],
                "query_index_generated_token_index": search["index_generated_token_index"],
                "query_index_generated_text": search["generated_text"],
                "query_index_generated_raw_text": search["generated_raw_text"],
                "filter_fact_line_from_blocks": False,
                **_generation_fields("baseline", baseline),
                **_generation_fields("replay", replay),
                "history_generated_text": expected_history,
                "gold_answer": row["answer"],
                "baseline_history_similarity": SequenceMatcher(None, baseline["generated_text"], expected_history).ratio(),
                "replay_history_similarity": SequenceMatcher(None, replay["generated_text"], expected_history).ratio(),
                "baseline_token_f1": _token_f1(baseline["generated_text"], expected_history),
                "replay_token_f1": _token_f1(replay["generated_text"], expected_history),
                "baseline_semantic_similarity": _semantic_similarity(semantic_model, baseline["generated_text"], expected_history),
                "replay_semantic_similarity": _semantic_similarity(semantic_model, replay["generated_text"], expected_history),
            }
        )

    _write_jsonl(output_dir / "test_results.jsonl", test_rows)
    _write_generation_report(output_dir / "generated_outputs.md", test_rows)
    summary = {
        "top_hit_rate": _mean([float(row["top_hit"]) for row in test_rows]),
        "baseline_history_similarity": _mean([row["baseline_history_similarity"] for row in test_rows]),
        "replay_history_similarity": _mean([row["replay_history_similarity"] for row in test_rows]),
        "baseline_token_f1": _mean([row["baseline_token_f1"] for row in test_rows]),
        "replay_token_f1": _mean([row["replay_token_f1"] for row in test_rows]),
        "baseline_eos_rate": _mean([float(row["baseline_eos_hit"]) for row in test_rows]),
        "replay_eos_rate": _mean([float(row["replay_eos_hit"]) for row in test_rows]),
        "baseline_generated_tokens": _mean([row["baseline_generated_tokens"] for row in test_rows]),
        "replay_generated_tokens": _mean([row["replay_generated_tokens"] for row in test_rows]),
        "custom_positioned_replay": custom_positioned_replay,
        "filter_fact_line_from_blocks": False,
        "pipeline_mode": DUAL_PIPELINE_MODE,
    }
    semantic_values = [row["replay_semantic_similarity"] for row in test_rows if row["replay_semantic_similarity"] is not None]
    if semantic_values:
        summary["replay_semantic_similarity"] = _mean(semantic_values)
        summary["baseline_semantic_similarity"] = _mean(
            [row["baseline_semantic_similarity"] for row in test_rows if row["baseline_semantic_similarity"] is not None]
        )
    return summary


def run_answer_block_experiment(config: AnswerBlockConfig) -> Path:
    if config.block_size <= 0:
        raise ValueError("block_size must be positive.")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive when building history.")
    if config.filter_fact_line_from_blocks:
        raise ValueError(
            "--filter-fact-line-from-blocks is incompatible "
            "with dual-pipeline natural main blocks."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    block_dir = config.output_dir / "history_blocks"
    block_dir.mkdir(exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "pipeline_mode": DUAL_PIPELINE_MODE,
                "output_dir": str(config.output_dir),
                "convomem_root": (
                    str(config.convomem_root)
                    if config.convomem_root
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Answer-block experiment requires torch.") from exc

    torch_module, tokenizer, model = _load_model(
        CollectConfig(
            model_name=config.model_name,
            dataset_path=Path("unused.jsonl"),
            output_dir=config.output_dir,
            max_new_tokens=config.max_new_tokens,
            dtype=config.dtype,
            device=config.device,
        )
    )
    torch = torch_module
    rows = _answer_block_rows(
        build_convomem_rows(
            config.convomem_root,
            max_facts=config.max_facts,
            seed=config.seed,
            max_context_turns=config.max_context_turns,
            use_chat_template=True,
            use_knowledge_prompt=False,
        )
    )
    _write_jsonl(config.output_dir / "episodes.jsonl", rows)

    final_position = 0
    history_past = None
    block_metadata = []
    index_vectors = []
    try:
        for episode_index, row in enumerate(rows):
            main_content, index_content = _dual_branch_prompts(
                row
            )
            main_prompt_tokens = _encode_chat_prompt(
                tokenizer,
                main_content,
            )
            index_prompt_tokens = _encode_chat_prompt(
                tokenizer,
                index_content,
            )
            history_cache_len = _cache_seq_len(history_past)
            if history_cache_len != final_position:
                raise RuntimeError(f"History cache/position mismatch: cache={history_cache_len}, position={final_position}")
            index_history_past = _clone_cache(history_past)
            main_result = _generate(
                torch,
                tokenizer,
                model,
                main_prompt_tokens.ids,
                start_position=final_position,
                max_new_tokens=config.max_new_tokens,
                index_layer=config.index_layer,
                prefix_past=history_past,
                capture_cache_prompt_start=len(
                    main_prompt_tokens.ids
                ),
                capture_cache_generated_tokens=min(config.block_size, config.max_new_tokens),
            )
            index_result = _generate(
                torch,
                tokenizer,
                model,
                index_prompt_tokens.ids,
                start_position=final_position,
                max_new_tokens=INDEX_PREFIX_MAX_NEW_TOKENS,
                index_layer=config.index_layer,
                prefix_past=index_history_past,
                capture_generated_prefix=GENERATED_PREFIX,
            )
            generated_ids = main_result["generated_ids"]
            block_end = _answer_block_end(generated_ids, config.block_size, tokenizer.eos_token_id)
            if block_end:
                quality = _history_quality(main_result["generated_text"], row["answer"])
                block_path = block_dir / f"{episode_index:05d}_000.pt"
                block_token_ids = generated_ids[:block_end]
                block_cache = main_result["captured_cache"]
                if block_cache is None or main_result["captured_generated_tokens"] != block_end:
                    raise RuntimeError(
                        f"History block {episode_index} cache capture mismatch: "
                        f"captured={main_result['captured_generated_tokens']}, expected={block_end}."
                    )
                _validate_legacy_cache(
                    block_cache,
                    expected_length=len(block_token_ids),
                    context=f"History block {episode_index}",
                )
                first_generated_position = (
                    final_position
                    + main_result["prompt_tokens"]
                )
                positions = list(
                    range(
                        first_generated_position,
                        first_generated_position + block_end,
                    )
                )
                torch.save({"past_key_values": block_cache, "positions": positions, "token_ids": block_token_ids}, block_path)
                block_metadata.append(
                    {
                        "block_id": f"{episode_index:05d}_000",
                        "episode_index": episode_index,
                        "target_fact_id": row["target_fact_id"],
                        "block_path": str(block_path.relative_to(config.output_dir)),
                        "positions": positions,
                        "token_ids": block_token_ids,
                        "token_text": _decode(tokenizer, block_token_ids),
                        "anchor_position": (
                            final_position
                            + index_result["prompt_tokens"]
                            + index_result["index_generated_token_index"]
                        ),
                        "index_generated_prefix": index_result["index_generated_prefix_text"],
                        "index_generated_token_index": index_result["index_generated_token_index"],
                        "index_prompt_tokens": index_result["prompt_tokens"],
                        "index_generated_text": index_result["generated_text"],
                        "index_generated_raw_text": index_result["generated_raw_text"],
                        "index_generated_token_ids": index_result["generated_ids"],
                        "main_content": main_content,
                        "index_content": index_content,
                        "block_starts_at_first_generated_token": True,
                        "suffix_tokens": 0,
                        "answer_tokens": block_end,
                        "history_start_position": final_position,
                        "history_cache_start": history_cache_len,
                        "history_generated_text": main_result["generated_text"],
                        "history_generated_raw_text": main_result["generated_raw_text"],
                        "history_generated_token_ids": main_result["generated_ids"],
                        "history_generated_tokens": main_result["generated_tokens"],
                        "history_eos_hit": main_result["eos_hit"],
                        "history_stop_reason": main_result["stop_reason"],
                        "gold_answer": row["answer"],
                        "specific_question": row["specific_question"],
                        **quality,
                    }
                )
                index_vectors.append(index_result["index_vector"])
            index_result.pop("past_key_values", None)
            del index_history_past
            history_past = main_result["past_key_values"]
            final_position += (
                main_result["prompt_tokens"]
                + len(generated_ids)
            )

        if not block_metadata:
            raise RuntimeError("No answer blocks were recorded.")

        if _cache_seq_len(history_past) != final_position:
            raise RuntimeError(f"Final history cache/position mismatch: cache={_cache_seq_len(history_past)}, position={final_position}")
        test_start_position = config.test_position_override if config.test_position_override is not None else final_position
        index_tensor = torch.stack(index_vectors)
        torch.save(index_tensor, config.output_dir / "index_vectors.pt")
        _write_jsonl(config.output_dir / "history_blocks.jsonl", block_metadata)

        summary = {
            "pipeline_mode": DUAL_PIPELINE_MODE,
            "episodes": len(rows),
            "history_blocks": len(block_metadata),
            "final_position": final_position,
            "test_start_position": test_start_position,
            "test_position_override": config.test_position_override,
            **_history_summary(block_metadata),
        }
        summary.update(
            _run_replay_tests(
                torch,
                tokenizer,
                model,
                rows=rows,
                block_metadata=block_metadata,
                index_tensor=index_tensor,
                history_dir=config.output_dir,
                output_dir=config.output_dir,
                test_start_position=test_start_position,
                top_n_blocks=config.top_n_blocks,
                max_new_tokens=config.max_new_tokens,
                index_layer=config.index_layer,
                semantic_model_name=config.semantic_model,
                custom_positioned_replay=config.custom_positioned_replay,
                filter_fact_line_from_blocks=False,
            )
        )
        (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return config.output_dir


def run_answer_block_replay_only(config: AnswerBlockConfig, history_dir: Path) -> Path:
    history_summary = json.loads(
        (history_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        history_summary.get("pipeline_mode")
        != DUAL_PIPELINE_MODE
    ):
        raise ValueError(
            "History directory is not a compatible dual-pipeline "
            f"cache: {history_dir}."
        )
    if config.filter_fact_line_from_blocks:
        raise ValueError(
            "--filter-fact-line-from-blocks is incompatible "
            "with dual-pipeline natural main blocks."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "pipeline_mode": DUAL_PIPELINE_MODE,
                "output_dir": str(config.output_dir),
                "convomem_root": str(config.convomem_root) if config.convomem_root else None,
                "reuse_history_dir": str(history_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Answer-block replay test requires torch.") from exc

    torch_module, tokenizer, model = _load_model(
        CollectConfig(
            model_name=config.model_name,
            dataset_path=Path("unused.jsonl"),
            output_dir=config.output_dir,
            max_new_tokens=config.max_new_tokens,
            dtype=config.dtype,
            device=config.device,
        )
    )
    torch = torch_module
    try:
        rows = _read_jsonl(history_dir / "episodes.jsonl")
        block_metadata = _read_jsonl(history_dir / "history_blocks.jsonl")
        index_tensor = torch.load(history_dir / "index_vectors.pt", map_location="cpu")
        history_test_start_position = int(history_summary.get("test_start_position", history_summary["final_position"]))
        test_start_position = config.test_position_override if config.test_position_override is not None else history_test_start_position

        _write_jsonl(config.output_dir / "episodes.jsonl", rows)
        _write_jsonl(config.output_dir / "history_blocks.jsonl", block_metadata)

        summary = {
            "episodes": len(rows),
            "pipeline_mode": DUAL_PIPELINE_MODE,
            "history_blocks": len(block_metadata),
            "final_position": int(history_summary["final_position"]),
            "test_start_position": test_start_position,
            "history_test_start_position": history_test_start_position,
            "test_position_override": config.test_position_override,
            "reuse_history_dir": str(history_dir),
            **_history_summary(block_metadata),
        }
        summary.update(
            _run_replay_tests(
                torch,
                tokenizer,
                model,
                rows=rows,
                block_metadata=block_metadata,
                index_tensor=index_tensor,
                history_dir=history_dir,
                output_dir=config.output_dir,
                test_start_position=test_start_position,
                top_n_blocks=config.top_n_blocks,
                max_new_tokens=config.max_new_tokens,
                index_layer=config.index_layer,
                semantic_model_name=config.semantic_model,
                custom_positioned_replay=config.custom_positioned_replay,
                filter_fact_line_from_blocks=False,
            )
        )
        (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return config.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer-trajectory block cache experiment.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-facts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--top-n-blocks", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--index-layer", type=int, default=40)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-context-turns", type=int, default=48)
    parser.add_argument("--convomem-root", type=Path)
    parser.add_argument("--semantic-model")
    parser.add_argument("--reuse-history-dir", type=Path)
    parser.add_argument("--custom-positioned-replay", action="store_true")
    parser.add_argument("--filter-fact-line-from-blocks", action="store_true")
    parser.add_argument("--test-position-override", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnswerBlockConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_facts=args.max_facts,
        seed=args.seed,
        block_size=args.block_size,
        top_n_blocks=args.top_n_blocks,
        max_new_tokens=args.max_new_tokens,
        index_layer=args.index_layer,
        dtype=args.dtype,
        device=args.device,
        max_context_turns=args.max_context_turns,
        convomem_root=args.convomem_root,
        semantic_model=args.semantic_model,
        custom_positioned_replay=args.custom_positioned_replay,
        test_position_override=args.test_position_override,
        filter_fact_line_from_blocks=args.filter_fact_line_from_blocks,
    )
    if args.reuse_history_dir:
        path = run_answer_block_replay_only(config, args.reuse_history_dir)
    else:
        path = run_answer_block_experiment(config)
    print(f"Wrote answer-block experiment to {path}")


if __name__ == "__main__":
    main()
