from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class ConvoMemExample:
    example_id: str
    source_file: str
    source_item_index: int
    category: str
    question: str
    answer: str
    evidence_context: str
    conversation_context: str


def _message_text(message: dict[str, Any]) -> str:
    speaker = str(message.get("speaker") or message.get("role") or "Turn")
    text = " ".join(str(message.get("text") or message.get("content") or "").split())
    return f"{speaker}: {text}".strip()


def _conversation_text(item: dict[str, Any]) -> str:
    conversations = item.get("conversations") or []
    if not conversations:
        return ""
    chosen = next(
        (conversation for conversation in conversations if conversation.get("containsEvidence")),
        conversations[0],
    )
    return "\n".join(_message_text(message) for message in chosen.get("messages") or [])


def iter_convomem_examples(evidence_root: Path | str) -> Iterator[ConvoMemExample]:
    root = Path(evidence_root)
    for path in sorted(root.glob("*/*/*.json")):
        relative = path.relative_to(root).as_posix()
        category = relative.split("/", 1)[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        for item_index, item in enumerate(data.get("evidence_items") or []):
            question = " ".join(str(item.get("question") or "").split())
            answer = " ".join(str(item.get("answer") or "").split())
            evidence = "\n".join(
                _message_text(message) for message in item.get("message_evidences") or []
            )
            conversation = _conversation_text(item) or evidence
            if not question or not answer or not evidence:
                continue
            digest = hashlib.sha1(
                f"{relative}:{item_index}:{question}:{answer}".encode("utf-8")
            ).hexdigest()[:16]
            yield ConvoMemExample(
                example_id=f"convomem-{digest}",
                source_file=relative,
                source_item_index=item_index,
                category=category,
                question=question,
                answer=answer,
                evidence_context=evidence,
                conversation_context=conversation,
            )


def _split_for_source(source_file: str, seed: int) -> str:
    # The same ConvoMem profile can appear below multiple category/evidence-count
    # directories. Group by the profile filename so none of its variants leak
    # across train, validation, and test.
    profile_id = Path(source_file).stem
    digest = hashlib.sha256(f"{seed}:{profile_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False).input_ids]


def _render_prompt(tokenizer: Any, content: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [(int(start), int(end)) for start, end in encoded.offset_mapping]
    return rendered, [int(token) for token in encoded.input_ids], offsets


def _token_span(
    rendered: str,
    offsets: Sequence[tuple[int, int]],
    text: str,
    *,
    reverse: bool = False,
) -> tuple[int, int]:
    character_start = rendered.rfind(text) if reverse else rendered.find(text)
    if character_start < 0:
        raise RuntimeError(f"rendered chat prompt lost boundary text: {text[:40]!r}")
    character_end = character_start + len(text)
    selected = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > character_start and start < character_end
    ]
    if not selected:
        raise RuntimeError("boundary text did not align to tokenizer offsets")
    return min(selected), max(selected) + 1


def _distractor_tokens(
    target: ConvoMemExample,
    candidates: Sequence[ConvoMemExample],
    tokenizer: Any,
    required_tokens: int,
    rng: random.Random,
) -> tuple[list[int], list[str]]:
    if required_tokens <= 0:
        return [], []
    eligible = [
        candidate
        for candidate in candidates
        if candidate.example_id != target.example_id
        and target.answer.lower() not in candidate.conversation_context.lower()
    ]
    if not eligible:
        raise ValueError("no answer-disjoint ConvoMem distractors are available")
    rng.shuffle(eligible)
    tokens: list[int] = []
    identifiers: list[str] = []
    cursor = 0
    while len(tokens) < required_tokens:
        candidate = eligible[cursor % len(eligible)]
        cursor += 1
        text = (
            f"\n\nIntervening conversation {cursor}:\n"
            f"{candidate.conversation_context}\n"
        )
        piece = _token_ids(tokenizer, text)
        if not piece:
            continue
        take = min(required_tokens - len(tokens), len(piece))
        tokens.extend(piece[:take])
        identifiers.append(candidate.example_id)
    return tokens, identifiers


def _neutral_memory_chunk_tokens(tokenizer: Any, conversation: str) -> tuple[list[int], int, int]:
    prefix = _token_ids(tokenizer, "\n\nMemory conversation:\n")
    content = _token_ids(tokenizer, conversation)
    suffix = _token_ids(tokenizer, "\n")
    return prefix + content + suffix, len(prefix), len(prefix) + len(content)


def _neutral_distractor_tokens(
    target: ConvoMemExample,
    candidates: Sequence[ConvoMemExample],
    tokenizer: Any,
    required_tokens: int,
    rng: random.Random,
) -> tuple[list[int], list[str]]:
    if required_tokens <= 0:
        return [], []
    eligible = [
        candidate
        for candidate in candidates
        if candidate.example_id != target.example_id
        and target.answer.lower() not in candidate.conversation_context.lower()
    ]
    if not eligible:
        raise ValueError("no answer-disjoint ConvoMem distractors are available")
    rng.shuffle(eligible)
    tokens: list[int] = []
    identifiers: list[str] = []
    cursor = 0
    while len(tokens) < required_tokens:
        candidate = eligible[cursor % len(eligible)]
        cursor += 1
        piece, _, _ = _neutral_memory_chunk_tokens(
            tokenizer, candidate.conversation_context
        )
        if not piece:
            continue
        take = min(required_tokens - len(tokens), len(piece))
        tokens.extend(piece[:take])
        identifiers.append(candidate.example_id)
    return tokens, identifiers


def _stratified_target_start(
    *,
    insertion_start: int,
    target_length: int,
    latest_target_end: int,
    block_size: int,
    bin_count: int,
    bin_index: int,
    rng: random.Random,
) -> tuple[int, int]:
    first = ((insertion_start + block_size - 1) // block_size) * block_size
    starts = list(range(first, latest_target_end - target_length + 1, block_size))
    if len(starts) < bin_count:
        raise ValueError("not enough candidate block positions for evidence placement bins")
    lower = len(starts) * bin_index // bin_count
    upper = len(starts) * (bin_index + 1) // bin_count
    if lower >= upper:
        raise RuntimeError("empty evidence placement bin")
    return rng.choice(starts[lower:upper]), len(starts)


def build_convomem_long_sequences(
    examples: Iterable[ConvoMemExample],
    tokenizer: Any,
    *,
    split: str,
    sequence_length: int,
    sequence_count: int,
    seed: int = 13,
    sampling_seed: int | None = None,
    maximum_answer_tokens: int = 64,
    maximum_future_horizon: int = 16,
    evidence_placement: str = "fixed_start",
    evidence_placement_bins: int = 4,
    placement_block_size: int = 64,
    retrieval_local_context_length: int = 256,
) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if sequence_length < 512 or sequence_count <= 0:
        raise ValueError("sequence_length must be at least 512 and count must be positive")
    if maximum_answer_tokens <= 0 or maximum_future_horizon <= 0:
        raise ValueError("answer and future-horizon limits must be positive")
    if evidence_placement not in {"fixed_start", "stratified_random"}:
        raise ValueError("evidence_placement must be fixed_start or stratified_random")
    if evidence_placement_bins <= 0 or placement_block_size <= 0:
        raise ValueError("placement bins and block size must be positive")
    if retrieval_local_context_length <= 0:
        raise ValueError("retrieval local context length must be positive")

    all_examples = list(examples)
    selected_pool = [
        example
        for example in all_examples
        if _split_for_source(example.source_file, seed) == split
    ]
    effective_sampling_seed = seed if sampling_seed is None else sampling_seed
    rng = random.Random(
        effective_sampling_seed + {"train": 0, "validation": 1, "test": 2}[split]
    )
    rng.shuffle(selected_pool)
    records: list[dict[str, Any]] = []
    for target in selected_pool:
        answer_ids = _token_ids(tokenizer, target.answer)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if (
            not answer_ids
            or len(answer_ids) > maximum_answer_tokens
            or eos_token_id is None
        ):
            continue
        if evidence_placement == "fixed_start":
            content = (
                "Use the conversation memory to answer the final question. "
                "Return only the answer.\n\n"
                f"Relevant earlier conversation:\n{target.evidence_context}\n\n"
                f"Question: {target.question}\n"
            )
        else:
            content = (
                "Use the conversation memory to answer the final question. "
                "Return only the answer.\n\n"
                f"Question: {target.question}\n"
            )
        rendered, prompt_ids, offsets = _render_prompt(tokenizer, content)
        question_start, question_end = _token_span(
            rendered, offsets, f"Question: {target.question}", reverse=True
        )
        desired_prompt_length = sequence_length - len(answer_ids) - 1
        required_distractor_tokens = desired_prompt_length - len(prompt_ids)
        if required_distractor_tokens <= 0:
            continue
        placement_bin = None
        placement_candidate_count = None
        if evidence_placement == "fixed_start":
            evidence_start, evidence_end = _token_span(
                rendered, offsets, target.evidence_context
            )
            distractor_ids, distractor_examples = _distractor_tokens(
                target,
                selected_pool,
                tokenizer,
                required_distractor_tokens,
                rng,
            )
            prompt_ids = (
                prompt_ids[:question_start] + distractor_ids + prompt_ids[question_start:]
            )
            target_chunk_range = [evidence_start, evidence_end]
            distractor_ranges = [[question_start, question_start + len(distractor_ids)]]
            distractor_range = list(distractor_ranges[0])
            question_start += len(distractor_ids)
            question_end += len(distractor_ids)
        else:
            target_ids, evidence_offset_start, evidence_offset_end = (
                _neutral_memory_chunk_tokens(tokenizer, target.evidence_context)
            )
            memory_token_count = required_distractor_tokens
            if len(target_ids) >= memory_token_count:
                continue
            final_local_start = (
                desired_prompt_length - 1 - retrieval_local_context_length
            )
            latest_candidate_block_end = (
                final_local_start // placement_block_size
            ) * placement_block_size
            placement_bin = len(records) % evidence_placement_bins
            target_start, placement_candidate_count = _stratified_target_start(
                insertion_start=question_start,
                target_length=len(target_ids),
                latest_target_end=latest_candidate_block_end,
                block_size=placement_block_size,
                bin_count=evidence_placement_bins,
                bin_index=placement_bin,
                rng=rng,
            )
            prefix_count = target_start - question_start
            suffix_count = memory_token_count - prefix_count - len(target_ids)
            if suffix_count < 0:
                continue
            distractor_prefix, prefix_examples = _neutral_distractor_tokens(
                target, selected_pool, tokenizer, prefix_count, rng
            )
            distractor_suffix, suffix_examples = _neutral_distractor_tokens(
                target, selected_pool, tokenizer, suffix_count, rng
            )
            memory_ids = distractor_prefix + target_ids + distractor_suffix
            if len(memory_ids) != memory_token_count:
                raise RuntimeError("randomized memory region has the wrong token length")
            prompt_ids = prompt_ids[:question_start] + memory_ids + prompt_ids[question_start:]
            target_chunk_range = [target_start, target_start + len(target_ids)]
            evidence_start = target_start + evidence_offset_start
            evidence_end = target_start + evidence_offset_end
            distractor_ids = distractor_prefix + distractor_suffix
            distractor_examples = prefix_examples + suffix_examples
            distractor_ranges = [
                [question_start, target_start],
                [target_start + len(target_ids), question_start + memory_token_count],
            ]
            distractor_range = [question_start, question_start + memory_token_count]
            question_start += memory_token_count
            question_end += memory_token_count
        answer_start = len(prompt_ids)
        token_ids = prompt_ids + answer_ids + [int(eos_token_id)]
        if len(token_ids) != sequence_length:
            raise RuntimeError("ConvoMem synthesis did not produce the requested exact length")
        retrieval_position = answer_start - 2
        future_horizon = min(maximum_future_horizon, len(answer_ids))
        if retrieval_position < 0 or retrieval_position + 1 + future_horizon >= len(token_ids):
            raise RuntimeError("answer-aligned retrieval point is outside the sequence")
        records.append(
            {
                "sequence_id": f"convomem-{split}-{len(records):07d}-{target.example_id}",
                "token_ids": token_ids,
                "source": "Salesforce/ConvoMem",
                "split": split,
                "task": "long_distance_memory_qa",
                "question": target.question,
                "answer": target.answer,
                "answer_token_ids": answer_ids,
                "answer_start_position": answer_start,
                "answer_end_position": answer_start + len(answer_ids),
                "evidence_token_ranges": [[evidence_start, evidence_end]],
                "question_token_range": [question_start, question_end],
                "distractor_token_range": distractor_range,
                "distractor_token_ranges": distractor_ranges,
                "evidence_to_answer_distance_tokens": answer_start - evidence_end,
                "distractor_token_count": len(distractor_ids),
                "distractor_example_ids": distractor_examples,
                "target_example_id": target.example_id,
                "source_file": target.source_file,
                "split_group_id": f"convomem-profile:{Path(target.source_file).stem}",
                "source_item_index": target.source_item_index,
                "source_category": target.category,
                "evidence_placement": evidence_placement,
                "evidence_placement_bin": placement_bin,
                "placement_candidate_count": placement_candidate_count,
                "target_memory_chunk_range": target_chunk_range,
                "evidence_block_indices": list(
                    range(
                        evidence_start // placement_block_size,
                        (evidence_end - 1) // placement_block_size + 1,
                    )
                ),
                "retrieval_points": [
                    {
                        "name": "answer",
                        "retrieval_position": retrieval_position,
                        "future_horizon_length": future_horizon,
                    }
                ],
            }
        )
        if len(records) == sequence_count:
            break
    if len(records) != sequence_count:
        raise ValueError(
            f"found only {len(records)} eligible {split} examples; requested {sequence_count}"
        )
    return records


def prepare_convomem_jsonl(
    *,
    tokenizer_name: str,
    output_path: Path,
    split: str,
    sequence_length: int,
    sequence_count: int,
    seed: int,
    sampling_seed: int | None = None,
    dataset_name: str = "Salesforce/ConvoMem",
    cache_dir: Path | None = None,
    dataset_cache_dir: Path | None = None,
    tokenizer_cache_dir: Path | None = None,
    local_files_only: bool = False,
    maximum_answer_tokens: int = 64,
    maximum_future_horizon: int = 16,
    evidence_placement: str = "fixed_start",
    evidence_placement_bins: int = 4,
    placement_block_size: int = 64,
    retrieval_local_context_length: int = 256,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")
    snapshot = Path(
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            cache_dir=str(dataset_cache_dir or cache_dir) if (dataset_cache_dir or cache_dir) else None,
            local_files_only=local_files_only,
        )
    )
    evidence_root = snapshot / "core_benchmark" / "evidence_questions"
    if not evidence_root.is_dir():
        raise FileNotFoundError(f"ConvoMem evidence root is missing: {evidence_root}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=tokenizer_cache_dir or cache_dir,
        local_files_only=local_files_only,
    )
    records = build_convomem_long_sequences(
        iter_convomem_examples(evidence_root),
        tokenizer,
        split=split,
        sequence_length=sequence_length,
        sequence_count=sequence_count,
        seed=seed,
        sampling_seed=sampling_seed,
        maximum_answer_tokens=maximum_answer_tokens,
        maximum_future_horizon=maximum_future_horizon,
        evidence_placement=evidence_placement,
        evidence_placement_bins=evidence_placement_bins,
        placement_block_size=placement_block_size,
        retrieval_local_context_length=retrieval_local_context_length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "source": dataset_name,
        "split": split,
        "split_policy": "profile-filename grouped deterministic 80/10/10 hash split",
        "sequence_count": len(records),
        "sequence_length": sequence_length,
        "seed": seed,
        "sampling_seed": seed if sampling_seed is None else sampling_seed,
        "maximum_answer_tokens": maximum_answer_tokens,
        "maximum_future_horizon": maximum_future_horizon,
        "evidence_placement": evidence_placement,
        "evidence_placement_bins": evidence_placement_bins,
        "placement_block_size": placement_block_size,
        "retrieval_local_context_length": retrieval_local_context_length,
        "retrieval_point_policy": "metadata answer-aligned",
        "dataset_snapshot": str(snapshot),
        "tokenizer": tokenizer_name,
        "output_jsonl": str(output_path.resolve()),
        "mean_evidence_to_answer_distance_tokens": sum(
            record["evidence_to_answer_distance_tokens"] for record in records
        ) / len(records),
        "sequence_ids": [record["sequence_id"] for record in records],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize exact-length, answer-aligned ConvoMem sequences"
    )
    parser.add_argument("--dataset-name", default="Salesforce/ConvoMem")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Randomizes sampling and placement without changing the profile split seed",
    )
    parser.add_argument("--maximum-answer-tokens", type=int, default=64)
    parser.add_argument("--maximum-future-horizon", type=int, default=16)
    parser.add_argument(
        "--evidence-placement",
        choices=("fixed_start", "stratified_random"),
        default="fixed_start",
    )
    parser.add_argument("--evidence-placement-bins", type=int, default=4)
    parser.add_argument("--placement-block-size", type=int, default=64)
    parser.add_argument("--retrieval-local-context-length", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    manifest = prepare_convomem_jsonl(
        tokenizer_name=arguments.tokenizer,
        output_path=arguments.output,
        split=arguments.split,
        sequence_length=arguments.sequence_length,
        sequence_count=arguments.sequences,
        seed=arguments.seed,
        sampling_seed=arguments.sampling_seed,
        dataset_name=arguments.dataset_name,
        cache_dir=arguments.cache_dir,
        dataset_cache_dir=arguments.dataset_cache_dir,
        tokenizer_cache_dir=arguments.tokenizer_cache_dir,
        local_files_only=not arguments.allow_network,
        maximum_answer_tokens=arguments.maximum_answer_tokens,
        maximum_future_horizon=arguments.maximum_future_horizon,
        evidence_placement=arguments.evidence_placement,
        evidence_placement_bins=arguments.evidence_placement_bins,
        placement_block_size=arguments.placement_block_size,
        retrieval_local_context_length=arguments.retrieval_local_context_length,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
