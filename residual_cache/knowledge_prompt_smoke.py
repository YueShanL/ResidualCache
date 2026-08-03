from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_cache.data_process import build_convomem_rows
from residual_cache.residual_collect import (
    CollectConfig,
    _chat_prompt,
    _generation_stop_token_ids,
    _load_model,
    _write_jsonl,
)


def _paired_rows(rows: list[dict], pair_count: int) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["target_fact_id"], {})[row["condition_id"]] = row

    selected = []
    for fact_id in sorted(grouped):
        pair = grouped[fact_id]
        if {"fact_reference", "question_query"} <= pair.keys():
            query_row = pair["question_query"]
            history_row = {
                **query_row,
                "prompt_id": query_row["prompt_id"].replace("question_query", "history_answer"),
                "condition_id": "history_answer",
                "prompt": query_row["history_prompt"],
            }
            selected.extend((pair["fact_reference"], query_row, history_row))
        if len(selected) >= pair_count * 3:
            break
    if len(selected) != pair_count * 3:
        raise RuntimeError(f"Requested {pair_count} pairs but found only {len(selected) // 3}.")
    return selected


def run_smoke(
    *,
    model_name: str,
    output_dir: Path,
    pair_count: int,
    seed: int,
    max_context_turns: int,
    max_new_tokens: int,
    dtype: str,
    device: str,
    convomem_root: Path | None,
) -> Path:
    candidate_rows = build_convomem_rows(
        convomem_root,
        max_facts=max(pair_count * 3, pair_count),
        seed=seed,
        max_context_turns=max_context_turns,
        use_chat_template=True,
        use_knowledge_prompt=True,
    )
    rows = _paired_rows(candidate_rows, pair_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch, tokenizer, model = _load_model(
        CollectConfig(
            model_name=model_name,
            dataset_path=Path("unused"),
            output_dir=output_dir,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            device=device,
        )
    )
    outputs = []
    try:
        model_device = next(model.parameters()).device
        stop_token_ids = sorted(_generation_stop_token_ids(tokenizer))
        for row in rows:
            prompt_text = _chat_prompt(tokenizer, row["prompt"])
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model_device)
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
            generated_ids = generated[0, input_ids.shape[1] :].tolist()
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            outputs.append(
                {
                    "prompt_id": row["prompt_id"],
                    "target_fact_id": row["target_fact_id"],
                    "condition_id": row["condition_id"],
                    "answer": row["answer"],
                    "prompt": row["prompt"],
                    "chat_prompt": prompt_text,
                    "prompt_token_count": len(prompt_ids),
                    "last_prompt_token_id": prompt_ids[-1],
                    "last_prompt_token_text": tokenizer.decode([prompt_ids[-1]]),
                    "generated_token_ids": generated_ids,
                    "generated_token_count": len(generated_ids),
                    "model_output": output_text,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = output_dir / "outputs.jsonl"
    _write_jsonl(output_path, outputs)
    (output_dir / "outputs.txt").write_text(
        "\n\n".join(
            (
                f"=== {row['prompt_id']} / {row['condition_id']} ===\n"
                f"EXPECTED ANSWER (inspection only): {row['answer']}\n"
                f"MODEL OUTPUT:\n{row['model_output']}"
            )
            for row in outputs
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "pair_count": pair_count,
                "seed": seed,
                "max_context_turns": max_context_turns,
                "max_new_tokens": max_new_tokens,
                "dtype": dtype,
                "device": device,
                "convomem_root": str(convomem_root) if convomem_root else None,
                "anchor": "last_prompt_token_before_first_generated_token",
                "metrics": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test paired ConvoMem knowledge-key generations.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-context-turns", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--convomem-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_smoke(
        model_name=args.model_name,
        output_dir=args.output_dir,
        pair_count=args.pair_count,
        seed=args.seed,
        max_context_turns=args.max_context_turns,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        convomem_root=args.convomem_root,
    )
    print(f"Wrote smoke outputs to {output_path}")


if __name__ == "__main__":
    main()
