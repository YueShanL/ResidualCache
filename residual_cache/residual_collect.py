from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


TARGETS = ("pre_attn", "q", "block_output")
POSITION_FAMILIES = ("final_prompt", "answer")
POSITION_FAMILY_CHOICES = ("final_prompt", "answer", "both", "generated_prefix")
GENERATED_PREFIX = "FACT:"


@dataclass(frozen=True)
class CollectConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    max_new_tokens: int = 8
    dtype: str = "auto"
    device: str = "auto"
    limit: int | None = None
    position_family: str = "final_prompt"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _torch_dtype(torch, dtype: str):
    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {dtype!r}") from exc


def _load_model(config: CollectConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Collection requires torch and transformers from the current environment.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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


def _generation_stop_token_ids(tokenizer) -> set[int]:
    token_ids = set()
    for attribute in ("eos_token_id", "eot_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    return token_ids


def _chat_prompt(tokenizer, prompt: str, *, raw_prompt: bool = False) -> str:
    if raw_prompt:
        return prompt.strip()
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt.rstrip() + "\n"


def _encode_pair(tokenizer, row: dict):
    prompt_text = _chat_prompt(tokenizer, row["prompt"], raw_prompt=bool(row.get("raw_prompt")))
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(row["answer"], add_special_tokens=False).input_ids
    if not answer_ids:
        raise ValueError(f"Answer tokenized to nothing for {row['prompt_id']}")
    return prompt_text, prompt_ids, answer_ids, prompt_ids + answer_ids


def _transformer_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        module = base
        try:
            for part in path.split("."):
                module = getattr(module, part)
            return list(module)
        except AttributeError:
            continue
    raise RuntimeError("Could not find decoder layers. Pass support for this architecture before collection.")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _stack_layer_rows(torch, rows: list[object]):
    width = max(row.shape[-1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[-1] == width:
            padded.append(row)
        else:
            padded.append(torch.nn.functional.pad(row, (0, width - row.shape[-1])))
    return torch.stack(padded).contiguous()


def _q_proj(layer):
    attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    return getattr(attention, "q_proj", None) if attention is not None else None


def _capture_residuals(torch, model, input_ids, positions: list[int]) -> dict[str, object]:
    layers = _transformer_layers(model)
    device_positions = torch.tensor(positions, dtype=torch.long, device=input_ids.device)
    captures = {name: [] for name in TARGETS}
    handles = []

    for layer_index, layer in enumerate(layers):
        def save_layer_input(_module, inputs, *, index=layer_index):
            captures["pre_attn"].append(inputs[0][0, device_positions, :].detach().cpu())

        def save_q(_module, _inputs, output):
            captures["q"].append(_hidden(output)[0, device_positions, :].detach().cpu())

        def save_layer_output(_module, _inputs, output):
            captures["block_output"].append(_hidden(output)[0, device_positions, :].detach().cpu())

        handles.append(layer.register_forward_pre_hook(save_layer_input))
        q_proj = _q_proj(layer)
        if q_proj is None:
            raise RuntimeError("Could not find self-attention q_proj for q comparison.")
        handles.append(q_proj.register_forward_hook(save_q))
        handles.append(layer.register_forward_hook(save_layer_output))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = len(layers)
    for name, rows in captures.items():
        if len(rows) != expected:
            raise RuntimeError(f"Captured {len(rows)} {name} layers, expected {expected}.")
    return {name: _stack_layer_rows(torch, rows) for name, rows in captures.items()}


def _generate_tokens(torch, tokenizer, model, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop_token_ids = sorted(_generation_stop_token_ids(tokenizer))
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids or None,
        )
    return generated[0, input_ids.shape[1] :].tolist()


def _generate_answer(torch, tokenizer, model, prompt_ids: list[int], max_new_tokens: int) -> str:
    generated_ids = _generate_tokens(torch, tokenizer, model, prompt_ids, max_new_tokens)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _generated_prefix_anchor_index(tokenizer, generated_ids: list[int], prefix: str = GENERATED_PREFIX) -> int:
    for index in range(len(generated_ids)):
        text = tokenizer.decode(generated_ids[: index + 1], skip_special_tokens=True)
        if text.lstrip().startswith(prefix):
            return index
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    raise RuntimeError(f"Model did not generate required prefix {prefix!r}; output was {decoded!r}.")


def _is_correct(pred_text: str, answer: str) -> bool:
    return pred_text.strip().lower().splitlines()[0:1] == [answer.strip().lower()]


def _positions(prompt_len: int, full_len: int, position_family: str) -> tuple[list[int], list[int]]:
    final_prompt = [prompt_len - 1]
    if position_family == "final_prompt":
        return final_prompt, []
    answer = list(range(prompt_len, full_len))
    if position_family == "answer":
        return answer, list(range(len(answer)))
    if position_family == "both":
        return final_prompt + answer, list(range(1, 1 + len(answer)))
    raise ValueError(f"Unknown position_family {position_family!r}; use final_prompt, answer, both, or generated_prefix.")


def collect(config: CollectConfig) -> None:
    rows = _read_jsonl(config.dataset_path)
    if config.limit is not None:
        rows = rows[: config.limit]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = config.output_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    (config.output_dir / "collect_config.json").write_text(
        json.dumps(
            {**asdict(config), "dataset_path": str(config.dataset_path), "output_dir": str(config.output_dir)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    torch, tokenizer, model = _load_model(config)
    metadata = []
    try:
        for row_index, row in enumerate(rows):
            _prompt_text, prompt_ids, answer_ids, full_ids = _encode_pair(tokenizer, row)
            anchor_position_family = "final_prompt"
            generated_prefix_index = None
            generated_prefix_text = None
            if config.position_family == "generated_prefix":
                generated_ids = _generate_tokens(torch, tokenizer, model, prompt_ids, config.max_new_tokens)
                generated_prefix_index = _generated_prefix_anchor_index(tokenizer, generated_ids)
                prefix_ids = generated_ids[: generated_prefix_index + 1]
                full_ids = prompt_ids + prefix_ids
                answer_ids = generated_ids
                positions = [len(full_ids) - 1]
                answer_indices = []
                anchor_position_family = "generated_prefix"
                generated_prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            else:
                positions, answer_indices = _positions(len(prompt_ids), len(full_ids), config.position_family)
                pred_text = _generate_answer(torch, tokenizer, model, prompt_ids, config.max_new_tokens)
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=next(model.parameters()).device)
            tensors = _capture_residuals(torch, model, input_ids, positions)
            tensor_path = tensor_dir / f"{row_index:05d}_{row['prompt_id']}.pt"
            torch.save(
                {
                    "targets": TARGETS,
                    "position_families": POSITION_FAMILIES,
                    "positions": positions,
                    "final_prompt_index": 0,
                    "anchor_position_family": anchor_position_family,
                    "answer_indices": answer_indices,
                    "states": tensors,
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                    "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
                },
                tensor_path,
            )
            metadata.append(
                {
                    **row,
                    "tensor_path": str(tensor_path.relative_to(config.output_dir)),
                    "model_answer": pred_text,
                    "generated_prefix": GENERATED_PREFIX if generated_prefix_index is not None else None,
                    "generated_prefix_token_index": generated_prefix_index,
                    "generated_prefix_text": generated_prefix_text,
                    "correct": _is_correct(pred_text, row["answer"]),
                    "prompt_tokens": len(prompt_ids),
                    "answer_tokens": len(answer_ids),
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "metadata.jsonl", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect residual states for ResidualCache pre-research.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--position-family", choices=POSITION_FAMILY_CHOICES, default="final_prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        CollectConfig(
            args.model_name,
            args.dataset_path,
            args.output_dir,
            args.max_new_tokens,
            args.dtype,
            args.device,
            args.limit,
            args.position_family,
        )
    )
    print(f"Wrote residual collection to {args.output_dir}")


if __name__ == "__main__":
    main()
