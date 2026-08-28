from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from cluster_router_validation.contracts import EvaluationExample


class DynamicConvoMemDataset:
    """Synthesize random-position ConvoMem cases in memory, never on disk."""

    def __init__(
        self,
        *,
        tokenizer_name: str,
        sequence_count: int = 256,
        sequence_length: int = 4096,
        split: str = "test",
        seed: int = 13,
        sampling_seed: int = 97,
        dataset_name: str = "Salesforce/ConvoMem",
        dataset_cache_dir: str | None = None,
        tokenizer_cache_dir: str | None = None,
        local_files_only: bool = True,
        maximum_answer_tokens: int = 64,
        maximum_future_horizon: int = 64,
        evidence_placement_bins: int = 4,
        block_size: int = 64,
        local_context_length: int = 512,
    ) -> None:
        if sequence_count <= 0 or sequence_length <= 0:
            raise ValueError("sequence_count and sequence_length must be positive")
        self.tokenizer_name = tokenizer_name
        self.sequence_count = int(sequence_count)
        self.sequence_length = int(sequence_length)
        self.split = split
        self.seed = int(seed)
        self.sampling_seed = int(sampling_seed)
        self.dataset_name = dataset_name
        self.dataset_cache_dir = dataset_cache_dir
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.local_files_only = bool(local_files_only)
        self.maximum_answer_tokens = int(maximum_answer_tokens)
        self.maximum_future_horizon = int(maximum_future_horizon)
        self.evidence_placement_bins = int(evidence_placement_bins)
        self.block_size = int(block_size)
        self.local_context_length = int(local_context_length)
        self._rows: list[dict[str, Any]] | None = None
        self._snapshot: Path | None = None

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return {
            "adapter": "dynamic_convomem_random_position",
            "dataset_id": self.dataset_name,
            "split": self.split,
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "seed": self.seed,
            "sampling_seed": self.sampling_seed,
            "maximum_answer_tokens": self.maximum_answer_tokens,
            "maximum_future_horizon": self.maximum_future_horizon,
            "evidence_placement": "stratified_random",
            "evidence_placement_bins": self.evidence_placement_bins,
            "block_size": self.block_size,
            "local_context_length": self.local_context_length,
            "materialization": "process_memory_only",
        }

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        from learnable_index.prepare_convomem import (
            build_convomem_long_sequences,
            iter_convomem_examples,
        )

        self._snapshot = Path(
            snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                cache_dir=self.dataset_cache_dir,
                local_files_only=self.local_files_only,
            )
        )
        evidence_root = self._snapshot / "core_benchmark" / "evidence_questions"
        if not evidence_root.is_dir():
            raise FileNotFoundError(f"ConvoMem evidence root is missing: {evidence_root}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.tokenizer_cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        self._rows = build_convomem_long_sequences(
            iter_convomem_examples(evidence_root),
            tokenizer,
            split=self.split,
            sequence_length=self.sequence_length,
            sequence_count=self.sequence_count,
            seed=self.seed,
            sampling_seed=self.sampling_seed,
            maximum_answer_tokens=self.maximum_answer_tokens,
            maximum_future_horizon=self.maximum_future_horizon,
            evidence_placement="stratified_random",
            evidence_placement_bins=self.evidence_placement_bins,
            placement_block_size=self.block_size,
            retrieval_local_context_length=self.local_context_length,
        )
        return self._rows

    def __iter__(self) -> Iterable[EvaluationExample]:
        for row in self._load():
            sequence_id = str(row["sequence_id"])
            evidence_ranges = tuple(
                (int(start), int(end)) for start, end in row["evidence_token_ranges"]
            )
            evidence_block_ids = tuple(
                f"{sequence_id}:block:{int(index) * self.block_size:09d}-"
                f"{(int(index) + 1) * self.block_size:09d}"
                for index in row.get("evidence_block_indices", ())
            )
            metadata = {
                key: row[key]
                for key in (
                    "source",
                    "split",
                    "task",
                    "source_category",
                    "evidence_placement",
                    "evidence_placement_bin",
                    "placement_candidate_count",
                    "target_example_id",
                    "target_memory_chunk_range",
                )
                if key in row
            }
            yield EvaluationExample(
                sample_id=sequence_id,
                group_id=str(row["split_group_id"]),
                reference_answer=str(row["answer"]),
                reference_token_ids=tuple(int(value) for value in row["answer_token_ids"]),
                sequence_length=len(row["token_ids"]),
                evidence_distance_tokens=int(row["evidence_to_answer_distance_tokens"]),
                evidence_token_count=sum(end - start for start, end in evidence_ranges),
                evidence_block_ids=evidence_block_ids,
                metadata=metadata,
                payload=row,
            )


def dynamic_convomem_dataset_factory(**kwargs: Any) -> DynamicConvoMemDataset:
    return DynamicConvoMemDataset(**kwargs)
