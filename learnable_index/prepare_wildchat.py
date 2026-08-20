from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


WILDCHAT_SPLITS = ("train", "validation", "test")


def _split_for_row(row_index: int, seed: int) -> str:
    """Assign one source row to a stable, disjoint split.

    WildChat-1M exposes one ``train`` split.  The HPC runner still needs
    train/validation/test inputs, so we make deterministic splits from the
    source order instead of downloading and shuffling the corpus three times.
    The 90/5/5 allocation leaves enough of the long-tail conversations for
    the 4096-example training run.
    """

    digest = sha256(f"wildchat:{seed}:{row_index}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_500:
        return "validation"
    return "test"


def _clean_messages(conversation: Any) -> list[dict[str, str]]:
    if not isinstance(conversation, list):
        return []
    messages: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _chat_template_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return [int(token_id) for token_id in encoded]


def build_wildchat_sequences(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    split: str,
    sequence_length: int,
    sequence_count: int,
    seed: int = 13,
    dataset_name: str = "allenai/WildChat-1M",
    dataset_split: str = "train",
    minimum_turns: int = 10,
) -> list[dict[str, Any]]:
    """Make exact-length records from naturally long WildChat conversations.

    A WildChat ``turn`` is one user/assistant round.  We filter the long tail
    before tokenization and then keep only conversations that contain at least
    ``sequence_length`` chat-template tokens.  No synthetic distractors or
    random insertion are performed; the first 4096 tokens remain a contiguous
    piece of the original conversation.
    """

    if split not in WILDCHAT_SPLITS:
        raise ValueError(f"split must be one of {WILDCHAT_SPLITS}")
    if sequence_length < 2 or sequence_count <= 0:
        raise ValueError("sequence_length must be at least 2 and count must be positive")
    if minimum_turns <= 0:
        raise ValueError("minimum_turns must be positive")

    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if _split_for_row(row_index, seed) != split:
            continue
        messages = _clean_messages(row.get("conversation"))
        if not messages:
            continue
        turn_count = int(row.get("turn") or (len(messages) // 2))
        if turn_count < minimum_turns:
            continue
        token_ids = _chat_template_ids(tokenizer, messages)
        if len(token_ids) < sequence_length:
            continue
        conversation_hash = str(row.get("conversation_hash") or f"row-{row_index:09d}")
        records.append(
            {
                "sequence_id": f"wildchat-{split}-{row_index:09d}",
                "token_ids": token_ids[:sequence_length],
                "source": dataset_name,
                "subset": dataset_split,
                "split": split,
                "source_row_index": row_index,
                "conversation_hash": conversation_hash,
                "turn_count": turn_count,
                "message_count": len(messages),
                "original_token_count": len(token_ids),
                "minimum_turns": minimum_turns,
            }
        )
        if len(records) == sequence_count:
            break
    if len(records) != sequence_count:
        raise ValueError(
            f"found only {len(records)} eligible WildChat conversations in split={split}; "
            f"requested {sequence_count} sequences of {sequence_length} tokens "
            f"with minimum_turns={minimum_turns}"
        )
    return records


def prepare_wildchat_jsonl(
    *,
    tokenizer_name: str,
    output_path: Path,
    split: str,
    sequence_length: int,
    sequence_count: int,
    seed: int,
    dataset_name: str = "allenai/WildChat-1M",
    dataset_split: str = "train",
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    minimum_turns: int = 10,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    manifest_path = output_path.with_suffix(".manifest.json")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    from datasets import DownloadConfig, load_dataset

    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    records = build_wildchat_sequences(
        dataset,
        tokenizer,
        split=split,
        sequence_length=sequence_length,
        sequence_count=sequence_count,
        seed=seed,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        minimum_turns=minimum_turns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")

    manifest = {
        "schema_version": 1,
        "source": dataset_name,
        "subset": dataset_split,
        "split": split,
        "selection_policy": (
            "stable 90/5/5 row split; minimum WildChat turns; "
            "contiguous chat-template prefix of exact requested length"
        ),
        "sequence_count": len(records),
        "sequence_length": sequence_length,
        "minimum_turns": minimum_turns,
        "seed": seed,
        "streaming": True,
        "tokenizer": {
            "requested_name": tokenizer_name,
            "resolved_name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "output_jsonl": str(output_path.resolve()),
        "sequence_ids": [record["sequence_id"] for record in records],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare long, natural WildChat conversations for learnable_index"
    )
    parser.add_argument("--dataset-name", default="allenai/WildChat-1M")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=WILDCHAT_SPLITS, default="train")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--sequences", type=int, default=4096)
    parser.add_argument("--minimum-turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    manifest = prepare_wildchat_jsonl(
        tokenizer_name=arguments.tokenizer,
        output_path=arguments.output,
        split=arguments.split,
        sequence_length=arguments.sequence_length,
        sequence_count=arguments.sequences,
        seed=arguments.seed,
        dataset_name=arguments.dataset_name,
        dataset_split=arguments.dataset_split,
        cache_dir=arguments.cache_dir,
        local_files_only=not arguments.allow_network,
        minimum_turns=arguments.minimum_turns,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
