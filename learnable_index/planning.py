from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import BlockRange


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    token_ids: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if len(self.token_ids) < 2:
            raise ValueError("a sequence must contain at least two tokens")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token ids must be non-negative")


@dataclass(frozen=True)
class PlanConfig:
    local_context_length: int = 256
    block_size: int = 32
    future_horizon_length: int = 16
    retrieval_interval: int = 32
    minimum_candidate_blocks: int = 1
    maximum_candidate_blocks: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "local_context_length",
            "block_size",
            "future_horizon_length",
            "retrieval_interval",
            "minimum_candidate_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_candidate_blocks is not None:
            if self.maximum_candidate_blocks < self.minimum_candidate_blocks:
                raise ValueError("maximum_candidate_blocks cannot be smaller than minimum_candidate_blocks")


@dataclass(frozen=True)
class RetrievalPlan:
    sample_id: str
    sequence_id: str
    retrieval_position: int
    first_future_position_affected_by_retrieval: int
    future_horizon_length: int
    local_context_start: int
    local_context_end: int
    candidate_blocks: tuple[BlockRange, ...]

    @property
    def future_start(self) -> int:
        return self.first_future_position_affected_by_retrieval

    @property
    def future_end(self) -> int:
        return self.future_start + self.future_horizon_length


def mechanical_blocks(sequence_id: str, token_count: int, block_size: int) -> tuple[BlockRange, ...]:
    if token_count <= 0 or block_size <= 0:
        raise ValueError("token_count and block_size must be positive")
    return tuple(
        BlockRange(
            block_id=f"{sequence_id}:block:{start:09d}-{start + block_size:09d}",
            start_position=start,
            end_position=start + block_size,
        )
        for start in range(0, token_count - block_size + 1, block_size)
    )


def build_retrieval_plans(record: SequenceRecord, config: PlanConfig) -> list[RetrievalPlan]:
    """Create retrieval points while reserving one next-token target after the horizon."""

    token_count = len(record.token_ids)
    blocks = mechanical_blocks(record.sequence_id, token_count, config.block_size)
    first_retrieval = (
        config.local_context_length
        + config.minimum_candidate_blocks * config.block_size
        - 1
    )
    # Future query positions are [t+1, t+H]. Their logits predict through t+H+1.
    last_retrieval = token_count - config.future_horizon_length - 2
    plans: list[RetrievalPlan] = []
    if first_retrieval > last_retrieval:
        return plans
    for retrieval_position in range(
        first_retrieval,
        last_retrieval + 1,
        config.retrieval_interval,
    ):
        local_end = retrieval_position + 1
        local_start = local_end - config.local_context_length
        candidates = tuple(block for block in blocks if block.end_position <= local_start)
        if config.maximum_candidate_blocks is not None:
            candidates = candidates[-config.maximum_candidate_blocks :]
        if len(candidates) < config.minimum_candidate_blocks:
            continue
        plans.append(
            RetrievalPlan(
                sample_id=f"{record.sequence_id}:retrieval:{retrieval_position:09d}",
                sequence_id=record.sequence_id,
                retrieval_position=retrieval_position,
                first_future_position_affected_by_retrieval=retrieval_position + 1,
                future_horizon_length=config.future_horizon_length,
                local_context_start=local_start,
                local_context_end=local_end,
                candidate_blocks=candidates,
            )
        )
    return plans


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def load_sequence_records(
    path: Path | str,
    tokenizer,
    *,
    maximum_sequences: int | None = None,
    maximum_tokens: int | None = None,
) -> list[SequenceRecord]:
    """Tokenize each row exactly once for both teacher and student branches."""

    records: list[SequenceRecord] = []
    for row_index, row in enumerate(_read_jsonl(Path(path))):
        sequence_id = str(row.get("sequence_id", f"sequence-{row_index:06d}"))
        if "token_ids" in row:
            token_ids = [int(token_id) for token_id in row["token_ids"]]
            source = "provided_token_ids"
        elif "messages" in row:
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
            )
            token_ids = list(encoded)
            source = "chat_template"
        elif "text" in row:
            token_ids = list(tokenizer(str(row["text"]), add_special_tokens=True).input_ids)
            source = "text"
        else:
            raise ValueError(f"row {row_index} must contain token_ids, messages, or text")
        if maximum_tokens is not None:
            token_ids = token_ids[:maximum_tokens]
        metadata = {key: value for key, value in row.items() if key not in {"token_ids", "messages", "text"}}
        metadata["tokenization_source"] = source
        records.append(SequenceRecord(sequence_id, tuple(token_ids), metadata))
        if maximum_sequences is not None and len(records) >= maximum_sequences:
            break
    if not records:
        raise ValueError("input JSONL produced no sequences")
    if len({record.sequence_id for record in records}) != len(records):
        raise ValueError("sequence_id values must be unique")
    return records

