from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Iterable


@dataclass(frozen=True)
class DataConfig:
    output_path: Path
    max_facts: int = 12
    seed: int = 13
    modes: tuple[str, ...] = ("natural", "canonical")
    source: str = "synthetic"
    max_context_turns: int = 24
    convomem_root: Path | None = None
    convomem_chat_template: bool = False
    convomem_knowledge_prompt: bool = False


ENTITIES = ("Alice", "Ben", "Carla", "Dev", "Eve", "Finn", "Gina", "Hana", "Ivan", "Jules", "Kira", "Liam")
COLORS = ("blue", "green", "red", "yellow", "purple", "silver", "orange", "white", "black", "teal", "pink", "gold")
CONVOMEM_DEFAULT_ROOT = Path(
    r"E:\huggingface\hub\datasets--Salesforce--ConvoMem\snapshots"
    r"\e3e9b39115b02346824c70d349350de738f8be41\core_benchmark"
)

CONVOMEM_KNOWLEDGE_INSTRUCTION = """Extract only the entity and the relevant fact.
Start your response with exactly one short line in this form:
FACT: <entity>, <fact>

Rules:
- Copy <entity> exactly from the `User profile:` field.
- Write <fact> as a compact noun phrase, not an explanatory sentence or paragraph.
- For conversation evidence, state the most specific current fact supported by the evidence.
- For a question without evidence, name the precise fact being requested without guessing its value.
- Use at most 18 words after `FACT:`.
- After the `FACT:` line, continue with the response you would normally give to the original input.
- Do not stop after the `FACT:` line and do not add another heading.
- Do not show hidden reasoning."""


def add_convomem_knowledge_instruction(prompt: str) -> str:
    return f"{CONVOMEM_KNOWLEDGE_INSTRUCTION}\n\nINPUT:\n{prompt}"


def append_convomem_index_instruction(prompt: str) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        f"{CONVOMEM_KNOWLEDGE_INSTRUCTION}"
    )


def refresh_convomem_knowledge_instruction(prompt: str) -> str:
    marker = "\n\nINPUT:\n"
    if marker not in prompt:
        raise ValueError(
            "Stored knowledge prompt has no INPUT boundary."
        )
    _stored_instruction, input_text = prompt.split(marker, 1)
    return add_convomem_knowledge_instruction(input_text)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _canonical_question(entity: str, *, policy: str = "current") -> str:
    return (
        "Task: recall_fact\n"
        f"Entity: {entity}\n"
        "Attribute: color\n"
        f"Policy: {policy}\n"
        "Question: Return the value only."
    )


def _row(
    *,
    prompt_id: str,
    task_id: str,
    fact_id: str,
    condition_id: str,
    mode: str,
    prompt: str,
    answer: str,
    entity: str,
    value: str,
    has_fact: bool = True,
    conflict: bool = False,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "task_id": task_id,
        "target_fact_id": fact_id,
        "condition_id": condition_id,
        "mode": mode,
        "entity": entity,
        "attribute": "color",
        "value": value,
        "answer": answer,
        "has_fact": has_fact,
        "conflict": conflict,
        "prompt": prompt,
    }


def build_rows(max_facts: int = 12, seed: int = 13, modes: tuple[str, ...] = ("natural", "canonical")) -> list[dict]:
    rng = random.Random(seed)
    facts = list(zip(ENTITIES, COLORS))[:max_facts]
    rows: list[dict] = []
    noise = [
        "",
        "Irrelevant note: the meeting starts at 9 and the ticket is X4.",
        "Archive: Nina's color is blue. Omar's color is green. Ignore archive entries unless named.",
    ]
    for fact_index, (entity, value) in enumerate(facts):
        fact_id = f"{entity.lower()}_color_{value}"
        other_value = COLORS[(fact_index + 3) % len(COLORS)]
        for mode in modes:
            base_question = _canonical_question(entity) if mode == "canonical" else f"What is {entity}'s color? Return only the color."
            for noise_index, noise_text in enumerate(noise):
                prompt = f"{noise_text}\nFact: {entity}'s color is {value}.\n{base_question}".strip()
                rows.append(
                    _row(
                        prompt_id=f"{mode}_noise_{fact_index}_{noise_index}",
                        task_id="recall_color",
                        fact_id=fact_id,
                        condition_id=f"noise_{noise_index}",
                        mode=mode,
                        prompt=prompt,
                        answer=value,
                        entity=entity,
                        value=value,
                    )
                )

            absent_prompt = (
                _canonical_question(entity)
                if mode == "canonical"
                else f"No color is given for {entity}. What is {entity}'s color? Return only the color."
            )
            rows.append(
                _row(
                    prompt_id=f"{mode}_absent_{fact_index}",
                    task_id="absent_color",
                    fact_id=fact_id,
                    condition_id="absent",
                    mode=mode,
                    prompt=absent_prompt,
                    answer="unknown",
                    entity=entity,
                    value=value,
                    has_fact=False,
                )
            )

            rows.append(
                _row(
                    prompt_id=f"{mode}_different_value_{fact_index}",
                    task_id="recall_color",
                    fact_id=f"{entity.lower()}_color_{other_value}",
                    condition_id="same_entity_different_value",
                    mode=mode,
                    prompt=f"Fact: {entity}'s color is {other_value}.\n{base_question}",
                    answer=other_value,
                    entity=entity,
                    value=other_value,
                )
            )

            rows.append(
                _row(
                    prompt_id=f"{mode}_verify_{fact_index}",
                    task_id="verify_color",
                    fact_id=fact_id,
                    condition_id="different_task_same_target",
                    mode=mode,
                    prompt=f"Fact: {entity}'s color is {value}.\nQuestion: Is {entity}'s color {value}? Answer yes or no.",
                    answer="yes",
                    entity=entity,
                    value=value,
                )
            )

            rows.append(
                _row(
                    prompt_id=f"{mode}_conflict_{fact_index}",
                    task_id="latest_color",
                    fact_id=f"{entity.lower()}_color_{other_value}",
                    condition_id="conflict_latest",
                    mode=mode,
                    prompt=(
                        f"Earlier: {entity}'s color is {value}.\n"
                        f"Later: {entity}'s color is {other_value}.\n"
                        f"Question: What is {entity}'s latest color? Return only the color."
                    ),
                    answer=other_value,
                    entity=entity,
                    value=other_value,
                    conflict=True,
                )
            )
    rng.shuffle(rows)
    return rows


def _turn_order(turn_id: str) -> tuple[int, int]:
    day, _, index = str(turn_id).partition(":")
    return int(day.lstrip("D") or 0), int(index or 0)


def _turn_text(turn: dict) -> str:
    role = str(turn.get("role", "turn")).capitalize()
    return f"{role}: {turn.get('content', '')}".strip()


def _one_line(text: object) -> str:
    return " ".join(str(text or "").split())


def _convomem_message_text(message: dict) -> str:
    speaker = str(message.get("speaker") or message.get("role") or "Turn")
    text = str(message.get("text") or message.get("content") or "")
    return f"{speaker}: {text}".strip()


def _convomem_question_key(question: object) -> str:
    return _one_line(question).lower()


def _convomem_profile(stem: str) -> str:
    _, sep, profile = stem.partition("_")
    return (profile if sep else stem).replace("_", " ")


def _convomem_specific_question(record: dict) -> str:
    question = record["question"].rstrip()
    if not question:
        return question
    lowered = question[:1].lower() + question[1:]
    return f"For the {record['profile']} user, {lowered}"


def _convomem_evidence_root(root: Path | None) -> Path:
    if root is None:
        env_root = os.environ.get("CONVOMEM_ROOT")
        root = Path(env_root) if env_root else CONVOMEM_DEFAULT_ROOT
    evidence_root = root / "evidence_questions"
    return evidence_root if evidence_root.exists() else root


def _convomem_conversation_context(item: dict, max_context_turns: int) -> str:
    conversations = item.get("conversations") or []
    if not conversations:
        return ""

    chosen = next((conv for conv in conversations if conv.get("containsEvidence")), conversations[0])
    messages = chosen.get("messages") or []
    if not messages:
        return ""

    evidence_texts = {
        _one_line(message.get("text")).lower()
        for message in item.get("message_evidences") or []
        if _one_line(message.get("text"))
    }
    evidence_indices = []
    for index, message in enumerate(messages):
        text = _one_line(message.get("text") or message.get("content")).lower()
        if any(evidence == text or evidence in text or text in evidence for evidence in evidence_texts):
            evidence_indices.append(index)

    limit = max_context_turns if max_context_turns > 0 else len(messages)
    if evidence_indices:
        first, last = min(evidence_indices), max(evidence_indices)
        span = last - first + 1
        if span >= limit:
            start, end = first, last + 1
        else:
            extra = limit - span
            start = max(0, first - extra // 2)
            end = min(len(messages), last + 1 + extra - (first - start))
            start = max(0, min(start, end - limit))
    else:
        start, end = 0, min(len(messages), limit)

    return "\n".join(_convomem_message_text(message) for message in messages[start:end])


def _lufy_fact_id(row: dict) -> str:
    evidence = "_".join(str(item) for item in row.get("evidence_turn_ids") or ["no_evidence"])
    answer_hash = hashlib.sha1(str(row.get("answer", "")).encode("utf-8")).hexdigest()[:10]
    return f"{row.get('user_name')}_{row.get('conversation_id')}_{evidence}_{answer_hash}"


def build_lufy_rows(
    qa_rows: Iterable[dict],
    turn_rows: Iterable[dict],
    *,
    max_qa: int = 120,
    seed: int = 13,
    max_context_turns: int = 24,
) -> list[dict]:
    turns_by_user: dict[str, list[dict]] = {}
    turn_by_key = {}
    for turn in turn_rows:
        user = str(turn.get("user_name"))
        turns_by_user.setdefault(user, []).append(turn)
        turn_by_key[(user, str(turn.get("turn_id")))] = turn
    for turns in turns_by_user.values():
        turns.sort(key=lambda turn: _turn_order(str(turn.get("turn_id"))))

    rows = []
    for index, qa in enumerate(list(qa_rows)[:max_qa]):
        evidence_ids = [str(item) for item in qa.get("evidence_turn_ids") or []]
        if not evidence_ids:
            continue
        user = str(qa.get("user_name"))
        evidence_turns = [turn_by_key[(user, turn_id)] for turn_id in evidence_ids if (user, turn_id) in turn_by_key]
        if not evidence_turns:
            continue
        latest = max(_turn_order(str(turn.get("turn_id"))) for turn in evidence_turns)
        prior_turns = [turn for turn in turns_by_user.get(user, []) if _turn_order(str(turn.get("turn_id"))) <= latest]
        recent_turns = prior_turns[-max_context_turns:]
        distractors = [turn for turn in turns_by_user.get(user, []) if turn not in evidence_turns][-6:]
        fact_id = _lufy_fact_id(qa)
        question = str(qa["question"]).strip()
        answer = str(qa["answer"]).strip()
        evidence_text = "\n".join(_turn_text(turn) for turn in evidence_turns)
        recent_text = "\n".join(_turn_text(turn) for turn in recent_turns)
        noisy_text = "\n".join(_turn_text(turn) for turn in distractors + evidence_turns)

        variants = [
            ("evidence_only", evidence_text, question, "natural"),
            ("recent_context", recent_text, question, "natural"),
            ("noisy_context", noisy_text, question, "natural"),
        ]
        for condition, context, final_question, mode in variants:
            prompt = f"Conversation evidence:\n{context}\n\nMemory request: {question}\nQuestion: {final_question}"
            rows.append(
                {
                    "prompt_id": f"lufy_{index:04d}_{condition}",
                    "task_id": "lufy_memory_qa",
                    "suite": "lufy_shift",
                    "target_fact_id": fact_id,
                    "condition_id": condition,
                    "question_key": question,
                    "mode": mode,
                    "entity": user,
                    "attribute": "memory_qa",
                    "value": answer,
                    "answer": answer,
                    "has_fact": True,
                    "conflict": False,
                    "prompt": prompt,
                    "source_dataset": "RuiSumida/LUFY",
                    "source_conversation_id": qa.get("conversation_id"),
                    "source_evidence_turn_ids": evidence_ids,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def load_lufy_rows(max_qa: int, seed: int, max_context_turns: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("LUFY loading requires the datasets package in the current environment.") from exc
    qa = load_dataset("RuiSumida/LUFY", "qa", split="train")
    turns = load_dataset("RuiSumida/LUFY", "turns", split="train")
    return build_lufy_rows(qa, turns, max_qa=max_qa, seed=seed, max_context_turns=max_context_turns)


def _iter_convomem_records(evidence_root: Path, max_context_turns: int) -> Iterable[dict]:
    for path in sorted(evidence_root.glob("*/*/*.json")):
        relative = path.relative_to(evidence_root)
        category = relative.parts[0]
        evidence_count = relative.parts[1] if len(relative.parts) > 1 else ""
        data = json.loads(path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(data.get("evidence_items") or []):
            question = _one_line(item.get("question"))
            answer = _one_line(item.get("answer"))
            evidence_messages = item.get("message_evidences") or []
            evidence_context = "\n".join(_convomem_message_text(message) for message in evidence_messages)
            if not question or not answer or not evidence_context.strip():
                continue
            answer_hash = hashlib.sha1(answer.encode("utf-8")).hexdigest()[:10]
            yield {
                "question": question,
                "question_key": _convomem_question_key(question),
                "specific_question": "",
                "answer": answer,
                "answer_key": answer.lower(),
                "evidence_context": evidence_context,
                "conversation_context": _convomem_conversation_context(item, max_context_turns) or evidence_context,
                "target_fact_id": f"convomem_{path.stem}_{item_index}_{answer_hash}",
                "category": category,
                "evidence_count": evidence_count,
                "source_file": str(relative).replace("\\", "/"),
                "source_item_index": item_index,
                "entity": path.stem,
                "profile": _convomem_profile(path.stem),
            }


def build_convomem_rows(
    convomem_root: Path | None = None,
    *,
    max_facts: int = 120,
    seed: int = 13,
    max_context_turns: int = 48,
    query_to_fact: bool = True,
    use_chat_template: bool = False,
    use_knowledge_prompt: bool = False,
) -> list[dict]:
    evidence_root = _convomem_evidence_root(convomem_root)
    if not evidence_root.exists():
        raise FileNotFoundError(f"ConvoMem evidence_questions root not found: {evidence_root}")

    records = list(_iter_convomem_records(evidence_root, max_context_turns))
    rng = random.Random(seed)
    if query_to_fact:
        rng.shuffle(records)
        selected = records[:max_facts]
    else:
        groups: dict[tuple[str, str], list[dict]] = {}
        for record in records:
            groups.setdefault((record["category"], record["question_key"]), []).append(record)
        eligible_groups = []
        for grouped_records in groups.values():
            by_answer = {}
            for record in grouped_records:
                by_answer.setdefault(record["answer_key"], record)
            if len(by_answer) >= 2:
                eligible_groups.append(list(by_answer.values()))
        rng.shuffle(eligible_groups)

        selected = []
        for grouped_records in eligible_groups:
            remaining = max_facts - len(selected)
            if remaining < 2:
                break
            rng.shuffle(grouped_records)
            take = grouped_records[: min(4, len(grouped_records), remaining)]
            if len(take) < 2:
                continue
            selected.extend(take)
            if len(selected) >= max_facts:
                break

    rows = []
    for index, record in enumerate(selected):
        specific_question = _convomem_specific_question(record)
        if query_to_fact:
            fact_prompt = f"User profile: {record['profile']}\nConversation evidence:\n{record['evidence_context']}"
            query_prompt = f"User profile: {record['profile']}\n\nQuestion: {specific_question}"
            history_prompt = f"{fact_prompt}\n\nQuestion: {specific_question}"
            if use_knowledge_prompt:
                fact_prompt = add_convomem_knowledge_instruction(fact_prompt)
                query_prompt = add_convomem_knowledge_instruction(query_prompt)
                history_prompt = add_convomem_knowledge_instruction(history_prompt)
            variants = (
                ("fact_reference", fact_prompt),
                ("question_query", query_prompt),
            )
        else:
            fact_prompt = ""
            query_prompt = ""
            history_prompt = ""
            variants = (
                (
                    "evidence_only",
                    f"Conversation evidence:\n{record['evidence_context']}\n\nQuestion: {record['question']}\nReturn only the answer.",
                ),
                (
                    "conversation_context",
                    f"Conversation evidence:\n{record['conversation_context']}\n\nQuestion: {record['question']}\nReturn only the answer.",
                ),
            )
        for condition, prompt in variants:
            if use_knowledge_prompt and not query_to_fact:
                prompt = add_convomem_knowledge_instruction(prompt)
            rows.append(
                {
                    "prompt_id": f"convomem_{index:04d}_{condition}",
                    "task_id": "convomem_memory_qa",
                    "suite": "convomem_query_to_fact" if query_to_fact else "convomem_repeated_question",
                    "target_fact_id": record["target_fact_id"],
                    "condition_id": condition,
                    "question_key": record["question_key"],
                    "specific_question": specific_question,
                    "mode": "natural",
                    "entity": record["entity"],
                    "attribute": record["category"],
                    "value": record["answer"],
                    "answer": record["answer"],
                    "has_fact": True,
                    "conflict": record["category"] == "changing_evidence",
                    "prompt": prompt,
                    "fact_prompt": fact_prompt,
                    "query_prompt": query_prompt,
                    "history_prompt": history_prompt,
                    "knowledge_prompt": use_knowledge_prompt,
                    "raw_prompt": query_to_fact and not use_chat_template,
                    "source_dataset": "Salesforce/ConvoMem",
                    "source_category": record["category"],
                    "source_evidence_count": record["evidence_count"],
                    "source_file": record["source_file"],
                    "source_item_index": record["source_item_index"],
                }
            )
    rng.shuffle(rows)
    return rows


def build_dataset(config: DataConfig) -> Path:
    if config.source == "synthetic":
        rows = build_rows(config.max_facts, config.seed, config.modes)
    elif config.source == "lufy":
        rows = load_lufy_rows(config.max_facts, config.seed, config.max_context_turns)
    elif config.source == "convomem":
        rows = build_convomem_rows(
            config.convomem_root,
            max_facts=config.max_facts,
            seed=config.seed,
            max_context_turns=config.max_context_turns,
            use_chat_template=config.convomem_chat_template,
            use_knowledge_prompt=config.convomem_knowledge_prompt,
        )
    else:
        raise ValueError(f"Unknown source {config.source!r}; use synthetic, lufy, or convomem.")
    _write_jsonl(config.output_path, rows)
    config_path = config.output_path.with_suffix(".config.json")
    serialized = asdict(config)
    serialized["output_path"] = str(config.output_path)
    if serialized["convomem_root"] is not None:
        serialized["convomem_root"] = str(serialized["convomem_root"])
    config_path.write_text(
        json.dumps(serialized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ResidualCache pre-research prompt dataset.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-facts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mode", action="append", choices=("natural", "canonical"), dest="modes")
    parser.add_argument("--source", choices=("synthetic", "lufy", "convomem"), default="synthetic")
    parser.add_argument("--max-context-turns", type=int, default=24)
    parser.add_argument("--convomem-root", type=Path)
    parser.add_argument("--convomem-chat-template", action="store_true")
    parser.add_argument("--convomem-knowledge-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_dataset(
        DataConfig(
            output_path=args.output_path,
            max_facts=args.max_facts,
            seed=args.seed,
            modes=tuple(args.modes or ("natural", "canonical")),
            source=args.source,
            max_context_turns=args.max_context_turns,
            convomem_root=args.convomem_root,
            convomem_chat_template=args.convomem_chat_template,
            convomem_knowledge_prompt=args.convomem_knowledge_prompt,
        )
    )
    print(f"Wrote dataset to {path}")


if __name__ == "__main__":
    main()
