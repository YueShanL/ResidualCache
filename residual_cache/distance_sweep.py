from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .answer_block_cache import (
    _decode,
    _decode_raw,
    _encode_chat_prompt,
    _first_line_end_token_index,
    _forward,
    _generate,
    _generation_stop_token_ids,
    _history_prompt,
    _query_prompt,
    _token_f1,
)
from .continuation_equivalence import (
    build_continuous_turn_tokens,
    validate_continuation_equivalence,
)
from .attention_analysis import _force_eager_attention
from .data_process import build_convomem_rows
from .residual_collect import CollectConfig, _load_model


@dataclass(frozen=True)
class DistanceSweepConfig:
    replay_dir: Path
    output_dir: Path
    case_index: int = 3
    gaps: tuple[int, ...] = (
        0,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    )
    joint_shifts: tuple[int, ...] = (0, 8192, 32768, 65536)
    control_gap: int = 512
    block_size: int = 64
    max_new_tokens: int = 64
    index_layer: int = 40
    attention_backend: str = "eager"
    use_knowledge_instruction: bool = False
    allow_position_extrapolation: bool = False


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


def _parse_nonnegative_ints(value: str) -> tuple[int, ...]:
    if value.strip().lower() in {"", "none"}:
        return ()
    try:
        values = tuple(
            int(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated integer positions."
        ) from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "Positions must be non-negative integers."
        )
    return values


def _condition_plan(
    gaps: tuple[int, ...],
    joint_shifts: tuple[int, ...],
    control_gap: int,
) -> list[dict]:
    if control_gap < 0:
        raise ValueError("control_gap must be non-negative.")
    conditions = [
        {
            "condition": "distance_sweep",
            "history_start_position": 0,
            "requested_gap": gap,
        }
        for gap in gaps
    ]
    conditions.extend(
        {
            "condition": "joint_shift_control",
            "history_start_position": shift,
            "requested_gap": control_gap,
        }
        for shift in joint_shifts
    )
    return conditions


def _answer_body(
    text: str,
    *,
    fact_prefixed: bool,
) -> str:
    if not fact_prefixed:
        return text
    return text.split("\n", 1)[1] if "\n" in text else ""


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


def _exact_gold_in_body(body: str, gold_answer: str) -> bool:
    gold = _normalize_text(gold_answer)
    candidate = _normalize_text(body)
    return bool(gold) and gold in candidate


def _retained_block_positions(
    tokenizer,
    *,
    history_start_position: int,
    prompt_ids: list[int],
    suffix_start: int,
    generated_ids: list[int],
    fact_prefixed: bool,
) -> tuple[list[int], list[int]]:
    suffix_ids = prompt_ids[suffix_start:]
    token_ids = suffix_ids + generated_ids
    generated_offset = len(suffix_ids)
    if fact_prefixed:
        line_end = _first_line_end_token_index(
            tokenizer,
            token_ids[generated_offset:],
        )
        body_start = generated_offset + line_end + 1
    else:
        body_start = generated_offset
    block_start_position = history_start_position + suffix_start
    all_positions = list(
        range(
            block_start_position,
            block_start_position + len(token_ids),
        )
    )
    retained_positions = (
        all_positions[:generated_offset]
        + all_positions[body_start:]
    )
    body_positions = all_positions[body_start:]
    return retained_positions, body_positions


def _max_position_embeddings(model) -> int | None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    value = getattr(text_config, "max_position_embeddings", None)
    return int(value) if value is not None else None


def _validate_position_plan(
    *,
    planned_end: int,
    position_limit: int | None,
    allow_extrapolation: bool,
) -> bool:
    extrapolated = (
        position_limit is not None
        and planned_end > position_limit
    )
    if extrapolated and not allow_extrapolation:
        raise ValueError(
            f"Condition would reach position {planned_end}, "
            f"past model limit {position_limit}. Pass "
            "--allow-position-extrapolation to run it "
            "intentionally."
        )
    return extrapolated


def _continue_from_prefill(
    *,
    torch,
    tokenizer,
    model,
    prefill_output,
    generated_start_position: int,
    max_new_tokens: int,
) -> dict:
    device = next(model.parameters()).device
    past = prefill_output.past_key_values
    next_token = int(
        prefill_output.logits[:, -1, :]
        .argmax(dim=-1)
        .item()
    )
    generated = []
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    stop_reason = (
        "length"
        if max_new_tokens
        else "max_new_tokens_zero"
    )
    for step in range(max_new_tokens):
        generated.append(next_token)
        token_input = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=device,
        )
        token_output, _ = _forward(
            torch,
            model,
            token_input,
            generated_start_position + step,
            past_key_values=past,
        )
        past = token_output.past_key_values
        if next_token in stop_token_ids:
            stop_reason = "eos"
            break
        next_token = int(
            token_output.logits[:, -1, :]
            .argmax(dim=-1)
            .item()
        )
    return {
        "generated_ids": generated,
        "generated_text": _decode(tokenizer, generated),
        "generated_raw_text": _decode_raw(
            tokenizer,
            generated,
        ),
        "generated_tokens": len(generated),
        "stop_reason": stop_reason,
        "past_key_values": past,
    }


def _generated_token_comparison(
    continuous_ids: list[int],
    cache_split_ids: list[int],
) -> dict:
    common_prefix_tokens = 0
    for continuous_id, cache_split_id in zip(
        continuous_ids,
        cache_split_ids,
    ):
        if continuous_id != cache_split_id:
            break
        common_prefix_tokens += 1
    first_mismatch = (
        None
        if continuous_ids == cache_split_ids
        else common_prefix_tokens
    )
    return {
        "exact": continuous_ids == cache_split_ids,
        "continuous_tokens": len(continuous_ids),
        "cache_split_tokens": len(cache_split_ids),
        "common_prefix_tokens": common_prefix_tokens,
        "first_mismatch": first_mismatch,
        "continuous_token_at_mismatch": (
            continuous_ids[first_mismatch]
            if first_mismatch is not None
            and first_mismatch < len(continuous_ids)
            else None
        ),
        "cache_split_token_at_mismatch": (
            cache_split_ids[first_mismatch]
            if first_mismatch is not None
            and first_mismatch < len(cache_split_ids)
            else None
        ),
    }


def _run_condition(
    *,
    torch,
    tokenizer,
    model,
    row: dict,
    condition: dict,
    config: DistanceSweepConfig,
) -> dict:
    history_start = int(condition["history_start_position"])
    gold_answer = row["answer"]
    if config.use_knowledge_instruction:
        history_user_content = _history_prompt(row)
        query_user_content = _query_prompt(row)
    else:
        history_user_content = row["history_prompt"]
        query_user_content = row["query_prompt"]
    history_prompt = _encode_chat_prompt(
        tokenizer,
        history_user_content,
    )
    history = _generate(
        torch,
        tokenizer,
        model,
        history_prompt.ids,
        start_position=history_start,
        max_new_tokens=config.max_new_tokens,
        index_layer=config.index_layer,
        capture_generated_prefix=(
            "FACT:"
            if config.use_knowledge_instruction
            else None
        ),
    )
    continuous_tokens = build_continuous_turn_tokens(
        tokenizer,
        history_user_content=history_user_content,
        history_prompt_ids=history_prompt.ids,
        history_generated_ids=history["generated_ids"],
        query_user_content=query_user_content,
    )
    retained_positions, body_positions = _retained_block_positions(
        tokenizer,
        history_start_position=history_start,
        prompt_ids=history_prompt.ids,
        suffix_start=history_prompt.suffix_start,
        generated_ids=history["generated_ids"],
        fact_prefixed=config.use_knowledge_instruction,
    )

    history_next_position = (
        history_start
        + len(continuous_tokens.history_ids)
    )
    query_start = (
        history_next_position
        + int(condition["requested_gap"])
    )
    position_limit = _max_position_embeddings(model)
    planned_end = (
        query_start
        + len(continuous_tokens.continuation_ids)
        + config.max_new_tokens
    )
    position_extrapolated = _validate_position_plan(
        planned_end=planned_end,
        position_limit=position_limit,
        allow_extrapolation=(
            config.allow_position_extrapolation
        ),
    )

    equivalence_report = None
    continuous_recall = None
    if int(condition["requested_gap"]) == 0:
        (
            equivalence_report,
            query_prefill,
            continuous_prefill,
        ) = (
            validate_continuation_equivalence(
                torch,
                model,
                continuous_tokens,
                start_position=history_start,
                return_continuous_output=True,
            )
        )
        equivalence_path = (
            config.output_dir
            / (
                "gap0_equivalence_"
                f"history_start_{history_start}.json"
            )
        )
        equivalence_path.write_text(
            json.dumps(equivalence_report, indent=2),
            encoding="utf-8",
        )
        if not equivalence_report["structural_passed"]:
            raise RuntimeError(
                "gap=0 token/position/mask equivalence failed; "
                f"see {equivalence_path}."
            )
        continuous_recall = _continue_from_prefill(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            prefill_output=continuous_prefill,
            generated_start_position=(
                history_start + len(continuous_tokens.full_ids)
            ),
            max_new_tokens=config.max_new_tokens,
        )
    else:
        device = next(model.parameters()).device
        history_input = torch.tensor(
            [continuous_tokens.history_ids],
            dtype=torch.long,
            device=device,
        )
        history_prefill, _ = _forward(
            torch,
            model,
            history_input,
            history_start,
        )
        continuation_input = torch.tensor(
            [continuous_tokens.continuation_ids],
            dtype=torch.long,
            device=device,
        )
        query_prefill, _ = _forward(
            torch,
            model,
            continuation_input,
            query_start,
            past_key_values=history_prefill.past_key_values,
        )
    recall = _continue_from_prefill(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        prefill_output=query_prefill,
        generated_start_position=(
            query_start
            + len(continuous_tokens.continuation_ids)
        ),
        max_new_tokens=config.max_new_tokens,
    )
    output_comparison = (
        _generated_token_comparison(
            continuous_recall["generated_ids"],
            recall["generated_ids"],
        )
        if continuous_recall is not None
        else None
    )
    history_body = _answer_body(
        history["generated_text"],
        fact_prefixed=config.use_knowledge_instruction,
    )
    recall_body = _answer_body(
        recall["generated_text"],
        fact_prefixed=config.use_knowledge_instruction,
    )
    first_recall_body_index = (
        _first_line_end_token_index(
            tokenizer,
            recall["generated_ids"],
        )
        + 1
        if config.use_knowledge_instruction
        else 0
    )
    first_recall_body_query_position = (
        query_start
        + len(continuous_tokens.continuation_ids)
        + first_recall_body_index
        - 1
    )
    closest_history_body_position = (
        max(body_positions)
        if body_positions
        else max(retained_positions)
    )
    return {
        **condition,
        "history_end_position": history_next_position - 1,
        "query_start_position": query_start,
        "model_position_limit": position_limit,
        "planned_end_position_exclusive": planned_end,
        "position_extrapolated": position_extrapolated,
        "query_prompt_tokens": len(
            continuous_tokens.continuation_ids
        ),
        "query_continuation_first_token_id": (
            continuous_tokens.continuation_ids[0]
        ),
        "query_continuation_starts_with_bos": (
            continuous_tokens.continuation_ids[0]
            == getattr(tokenizer, "bos_token_id", None)
        ),
        "gap0_equivalence": equivalence_report,
        "gap0_output_comparison": output_comparison,
        "use_knowledge_instruction": (
            config.use_knowledge_instruction
        ),
        "history_user_content": history_user_content,
        "query_user_content": query_user_content,
        "first_recall_body_query_position": (
            first_recall_body_query_position
        ),
        "closest_history_body_position": (
            closest_history_body_position
        ),
        "actual_body_to_recall_distance": (
            first_recall_body_query_position
            - closest_history_body_position
        ),
        "history_context_tokens": len(
            continuous_tokens.history_ids
        ),
        "history_generated_text": history["generated_text"],
        "history_generated_raw_text": history[
            "generated_raw_text"
        ],
        "history_generated_tokens": history["generated_tokens"],
        "history_stop_reason": history["stop_reason"],
        "history_body_gold_f1": _token_f1(
            history_body,
            gold_answer,
        ),
        "history_exact_gold": _exact_gold_in_body(
            history_body,
            gold_answer,
        ),
        "recall_generated_text": recall["generated_text"],
        "recall_generated_ids": recall["generated_ids"],
        "recall_generated_raw_text": recall[
            "generated_raw_text"
        ],
        "recall_generated_tokens": recall["generated_tokens"],
        "recall_stop_reason": recall["stop_reason"],
        "recall_body_gold_f1": _token_f1(
            recall_body,
            gold_answer,
        ),
        "recall_body_history_f1": _token_f1(
            recall_body,
            history_body,
        ),
        "recall_exact_gold": _exact_gold_in_body(
            recall_body,
            gold_answer,
        ),
        "continuous_recall_generated_text": (
            continuous_recall["generated_text"]
            if continuous_recall is not None
            else None
        ),
        "continuous_recall_generated_raw_text": (
            continuous_recall["generated_raw_text"]
            if continuous_recall is not None
            else None
        ),
        "continuous_recall_generated_ids": (
            continuous_recall["generated_ids"]
            if continuous_recall is not None
            else None
        ),
        "continuous_recall_generated_tokens": (
            continuous_recall["generated_tokens"]
            if continuous_recall is not None
            else None
        ),
        "continuous_recall_stop_reason": (
            continuous_recall["stop_reason"]
            if continuous_recall is not None
            else None
        ),
        "gold_answer": gold_answer,
        "specific_question": row["specific_question"],
    }


def _write_markdown(
    path: Path,
    *,
    case_index: int,
    target_fact_id: str,
    rows: list[dict],
) -> None:
    lines = [
        "# Fixed-case position-distance sweep",
        "",
        f"- Case index: `{case_index}`",
        f"- Target fact: `{target_fact_id}`",
        "",
    ]
    for result in rows:
        lines.extend(
            [
                (
                    "## "
                    f"{result['condition']} "
                    f"history_start={result['history_start_position']} "
                    f"gap={result['requested_gap']}"
                ),
                "",
                (
                    "- Actual body-to-recall distance: "
                    f"`{result['actual_body_to_recall_distance']}`"
                ),
                (
                    "- History/recall gold F1: "
                    f"`{result['history_body_gold_f1']:.6f}` / "
                    f"`{result['recall_body_gold_f1']:.6f}`"
                ),
                (
                    "- Recall exact gold: "
                    f"`{result['recall_exact_gold']}`"
                ),
                "",
                "### History",
                "",
                result["history_generated_text"],
                "",
                "### Recall",
                "",
                result["recall_generated_text"],
                "",
            ]
        )
        if result["continuous_recall_generated_text"] is not None:
            comparison = result["gap0_output_comparison"]
            lines.extend(
                [
                    "### Full-continuous oracle recall",
                    "",
                    result["continuous_recall_generated_text"],
                    "",
                    (
                        "- Output token sequence exact: "
                        f"`{comparison['exact']}`"
                    ),
                    (
                        "- Common output prefix tokens: "
                        f"`{comparison['common_prefix_tokens']}`"
                    ),
                    (
                        "- First output mismatch: "
                        f"`{comparison['first_mismatch']}`"
                    ),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_distance_sweep(config: DistanceSweepConfig) -> Path:
    if config.block_size <= 0:
        raise ValueError("block_size must be positive.")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    replay_config = _read_json(config.replay_dir / "config.json")
    convomem_root_value = replay_config.get("convomem_root")
    rows = [
        row
        for row in build_convomem_rows(
            (
                Path(convomem_root_value)
                if convomem_root_value
                else None
            ),
            max_facts=int(
                replay_config.get("max_facts", 24)
            ),
            seed=int(replay_config.get("seed", 13)),
            max_context_turns=int(
                replay_config.get("max_context_turns", 48)
            ),
            use_chat_template=True,
            use_knowledge_prompt=(
                config.use_knowledge_instruction
            ),
        )
        if row.get("condition_id") == "question_query"
    ]
    if not 0 <= config.case_index < len(rows):
        raise ValueError(
            f"case_index={config.case_index} outside "
            f"[0, {len(rows) - 1}]."
        )
    row = rows[config.case_index]
    conditions = _condition_plan(
        config.gaps,
        config.joint_shifts,
        config.control_gap,
    )
    if not conditions:
        raise ValueError("No distance conditions requested.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    saved_config = {
        **asdict(config),
        "replay_dir": str(config.replay_dir),
        "output_dir": str(config.output_dir),
        "gaps": list(config.gaps),
        "joint_shifts": list(config.joint_shifts),
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
            max_new_tokens=config.max_new_tokens,
            dtype=replay_config["dtype"],
            device=replay_config["device"],
        )
    )
    if config.attention_backend == "eager":
        _force_eager_attention(model)
    elif config.attention_backend != "default":
        raise ValueError(
            "attention_backend must be 'eager' or 'default'."
        )
    results = []
    try:
        for condition in conditions:
            results.append(
                _run_condition(
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    row=row,
                    condition=condition,
                    config=config,
                )
            )
            print(
                condition["condition"],
                condition["history_start_position"],
                condition["requested_gap"],
                results[-1]["recall_body_gold_f1"],
                flush=True,
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "results.jsonl", results)
    summary = {
        "case_index": config.case_index,
        "target_fact_id": row["target_fact_id"],
        "gold_answer": row["answer"],
        "conditions": len(results),
        "distance_sweep": [
            {
                key: result[key]
                for key in (
                    "requested_gap",
                    "actual_body_to_recall_distance",
                    "history_body_gold_f1",
                    "recall_body_gold_f1",
                    "recall_body_history_f1",
                    "recall_exact_gold",
                    "recall_generated_tokens",
                )
            }
            for result in results
            if result["condition"] == "distance_sweep"
        ],
        "joint_shift_control": [
            {
                key: result[key]
                for key in (
                    "history_start_position",
                    "requested_gap",
                    "actual_body_to_recall_distance",
                    "history_body_gold_f1",
                    "recall_body_gold_f1",
                    "recall_exact_gold",
                )
            }
            for result in results
            if result["condition"] == "joint_shift_control"
        ],
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        config.output_dir / "generated_outputs.md",
        case_index=config.case_index,
        target_fact_id=row["target_fact_id"],
        rows=results,
    )
    return config.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one fixed history/recall case across absolute "
            "position gaps."
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
    parser.add_argument("--case-index", type=int, default=3)
    parser.add_argument(
        "--gaps",
        type=_parse_nonnegative_ints,
        default=_parse_nonnegative_ints(
            "0,128,256,512,1024,2048,4096,8192,"
            "16384,32768,65536"
        ),
    )
    parser.add_argument(
        "--joint-shifts",
        type=_parse_nonnegative_ints,
        default=_parse_nonnegative_ints(
            "0,8192,32768,65536"
        ),
    )
    parser.add_argument("--control-gap", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--index-layer", type=int, default=40)
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "default"),
        default="eager",
    )
    parser.add_argument(
        "--use-knowledge-instruction",
        action="store_true",
        help=(
            "Wrap freshly built fact/query prompts in the "
            "knowledge instruction and require a FACT: prefix. "
            "The default is raw ConvoMem prompts with no FACT "
            "detection."
        ),
    )
    parser.add_argument(
        "--allow-position-extrapolation",
        action="store_true",
        help=(
            "Allow requested absolute positions beyond the "
            "model's configured max_position_embeddings. "
            "The requested position_ids are passed through "
            "unchanged."
        ),
    )
    args = parser.parse_args()
    output_dir = run_distance_sweep(
        DistanceSweepConfig(
            replay_dir=args.replay_dir,
            output_dir=args.output_dir,
            case_index=args.case_index,
            gaps=args.gaps,
            joint_shifts=args.joint_shifts,
            control_gap=args.control_gap,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            index_layer=args.index_layer,
            attention_backend=args.attention_backend,
            use_knowledge_instruction=(
                args.use_knowledge_instruction
            ),
            allow_position_extrapolation=(
                args.allow_position_extrapolation
            ),
        )
    )
    print(output_dir)


if __name__ == "__main__":
    main()
