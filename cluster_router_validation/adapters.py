from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import DistributionState, EvaluationExample, ModelRun


class JsonlEvaluationDataset:
    """Reference adapter for ConvoMem-style JSONL rows.

    The original row is preserved in ``EvaluationExample.payload`` so a model
    adapter can consume token IDs, prompts, or custom dataset fields without
    coupling the validation runner to one dataset implementation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if maximum_samples is not None and maximum_samples <= 0:
            raise ValueError("maximum_samples must be positive")
        self.maximum_samples = maximum_samples
        hasher = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        self._descriptor = {
            "adapter": "jsonl",
            "dataset_id": dataset_id or self.path.stem,
            "split": split,
            "path": str(self.path.resolve()),
            "sha256": digest,
        }

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self._descriptor

    @staticmethod
    def _evidence_token_count(row: Mapping[str, Any]) -> int:
        if "evidence_token_count" in row:
            return int(row["evidence_token_count"])
        ranges = row.get("evidence_token_ranges", ())
        return sum(max(0, int(end) - int(start)) for start, end in ranges)

    @staticmethod
    def _sequence_length(row: Mapping[str, Any]) -> int:
        if "sequence_length" in row:
            return int(row["sequence_length"])
        token_ids = row.get("token_ids")
        if token_ids is None:
            raise ValueError("JSONL row needs sequence_length or token_ids")
        return len(token_ids)

    def __iter__(self) -> Iterable[EvaluationExample]:
        seen: set[str] = set()
        yielded = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(
                    row.get("sample_id")
                    or row.get("sequence_id")
                    or f"row-{line_number:08d}"
                )
                if sample_id in seen:
                    raise ValueError(f"duplicate sample_id in JSONL: {sample_id}")
                seen.add(sample_id)
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
                    )
                    if key in row
                }
                yield EvaluationExample(
                    sample_id=sample_id,
                    group_id=(
                        None
                        if row.get("split_group_id") is None
                        else str(row["split_group_id"])
                    ),
                    reference_answer=str(row.get("answer", "")),
                    reference_token_ids=tuple(
                        int(value) for value in row.get("answer_token_ids", ())
                    ),
                    sequence_length=self._sequence_length(row),
                    evidence_distance_tokens=int(
                        row.get("evidence_to_answer_distance_tokens", 0)
                    ),
                    evidence_token_count=self._evidence_token_count(row),
                    evidence_block_ids=tuple(
                        str(value) for value in row.get("evidence_block_indices", ())
                    ),
                    metadata=metadata,
                    payload=row,
                )
                yielded += 1
                if self.maximum_samples is not None and yielded >= self.maximum_samples:
                    break


def jsonl_dataset_factory(**kwargs: Any) -> JsonlEvaluationDataset:
    return JsonlEvaluationDataset(**kwargs)


def compact_torch_logits(
    reference: ModelRun,
    candidate: ModelRun,
    target_token_ids: Sequence[int],
) -> DistributionState:
    """Build exact compact statistics from transient full-vocabulary logits.

    Model adapters can keep logits only for the duration of one sample and use
    this helper from ``EvaluationSession.compact_distribution``. The state file
    receives six scalars/counts rather than ``[answer_tokens, vocabulary]``.
    """

    try:
        import torch
        from torch.nn import functional as F
    except ImportError as error:  # pragma: no cover - production runtime has torch.
        raise RuntimeError("compact_torch_logits requires torch") from error
    reference_logits = torch.as_tensor(reference.distribution_payload).detach().float()
    candidate_logits = (
        torch.as_tensor(candidate.distribution_payload)
        .detach()
        .to(reference_logits.device)
        .float()
    )
    targets = torch.as_tensor(
        tuple(target_token_ids), dtype=torch.long, device=reference_logits.device
    )
    if reference_logits.ndim != 2 or candidate_logits.shape != reference_logits.shape:
        raise ValueError("reference/candidate logits must share [tokens, vocab] shape")
    if reference_logits.shape[0] != targets.numel() or targets.numel() == 0:
        raise ValueError("target_token_ids do not align with logits")
    reference_log_probabilities = reference_logits.log_softmax(dim=-1)
    reference_probabilities = reference_log_probabilities.exp()
    candidate_log_probabilities = candidate_logits.log_softmax(dim=-1)
    target_nll = F.nll_loss(
        candidate_log_probabilities, targets, reduction="sum"
    )
    reference_entropy = -(
        reference_probabilities * reference_log_probabilities
    ).sum()
    reference_cross_entropy = -(
        reference_probabilities * candidate_log_probabilities
    ).sum()
    reference_argmax = reference_logits.argmax(dim=-1)
    candidate_argmax = candidate_logits.argmax(dim=-1)
    return DistributionState(
        token_count=int(targets.numel()),
        target_nll_sum=max(0.0, float(target_nll)),
        reference_entropy_sum=max(0.0, float(reference_entropy)),
        reference_cross_entropy_sum=max(0.0, float(reference_cross_entropy)),
        argmax_agreement_count=int((reference_argmax == candidate_argmax).sum()),
        target_accuracy_count=int((candidate_argmax == targets).sum()),
    )
