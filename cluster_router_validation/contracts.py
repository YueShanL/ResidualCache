from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence


STATE_SCHEMA_VERSION = 1


def _finite(value: float, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _non_negative_map(values: Mapping[int, int], *, name: str) -> dict[int, int]:
    result = {int(layer): int(count) for layer, count in values.items()}
    if any(layer < 0 or count < 0 for layer, count in result.items()):
        raise ValueError(f"{name} must contain non-negative layers and counts")
    return result


@dataclass(frozen=True)
class EvaluationExample:
    """Dataset-neutral input contract for one memory QA evaluation case."""

    sample_id: str
    reference_answer: str
    sequence_length: int
    evidence_distance_tokens: int
    evidence_token_count: int
    reference_token_ids: tuple[int, ...] = ()
    evidence_block_ids: tuple[str, ...] = ()
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.evidence_distance_tokens < 0 or self.evidence_token_count < 0:
            raise ValueError("evidence distance/count cannot be negative")
        object.__setattr__(
            self, "reference_token_ids", tuple(int(value) for value in self.reference_token_ids)
        )
        object.__setattr__(
            self, "evidence_block_ids", tuple(str(value) for value in self.evidence_block_ids)
        )
        if len(set(self.evidence_block_ids)) != len(self.evidence_block_ids):
            raise ValueError("evidence_block_ids must be unique")

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "reference_answer": self.reference_answer,
            "reference_token_ids": list(self.reference_token_ids),
            "sequence_length": self.sequence_length,
            "evidence_distance_tokens": self.evidence_distance_tokens,
            "evidence_token_count": self.evidence_token_count,
            "evidence_block_ids": list(self.evidence_block_ids),
            "metadata": dict(self.metadata),
        }


class EvaluationDataset(Protocol):
    @property
    def descriptor(self) -> Mapping[str, Any]: ...

    def __iter__(self) -> Iterable[EvaluationExample]: ...


@dataclass(frozen=True)
class ClusterCandidate:
    """One current memory leaf exposed to the built-in comparison policies."""

    layer: int
    cluster_id: str
    record_ids: tuple[str, ...]
    record_token_count: int
    latest_position: int
    learned_probability: float | None = None
    learned_log_score: float | None = None
    teacher_attention_mass: float | None = None
    evidence_record_count: int = 0
    evidence_token_count: int = 0
    evidence_block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layer < 0 or not self.cluster_id:
            raise ValueError("cluster layer/id is invalid")
        records = tuple(str(value) for value in self.record_ids)
        if len(set(records)) != len(records):
            raise ValueError("record_ids must be unique within a cluster")
        object.__setattr__(self, "record_ids", records)
        if self.record_token_count < 0 or self.latest_position < 0:
            raise ValueError("cluster token count/position cannot be negative")
        if not 0 <= self.evidence_record_count <= len(records):
            raise ValueError("evidence_record_count is inconsistent with record_ids")
        if self.evidence_token_count < 0:
            raise ValueError("evidence_token_count cannot be negative")
        for name in (
            "learned_probability",
            "learned_log_score",
            "teacher_attention_mass",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name=name)
        if self.learned_probability is not None and self.learned_probability < 0.0:
            raise ValueError("learned_probability cannot be negative")
        if self.teacher_attention_mass is not None and self.teacher_attention_mass < 0.0:
            raise ValueError("teacher_attention_mass cannot be negative")
        blocks = tuple(str(value) for value in self.evidence_block_ids)
        if len(set(blocks)) != len(blocks):
            raise ValueError("evidence_block_ids must be unique within a cluster")
        object.__setattr__(self, "evidence_block_ids", blocks)

    def state_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "cluster_id": self.cluster_id,
            "record_ids": list(self.record_ids),
            "record_count": len(self.record_ids),
            "record_token_count": self.record_token_count,
            "latest_position": self.latest_position,
            "learned_probability": self.learned_probability,
            "learned_log_score": self.learned_log_score,
            "teacher_attention_mass": self.teacher_attention_mass,
            "evidence_record_count": self.evidence_record_count,
            "evidence_token_count": self.evidence_token_count,
            "evidence_block_ids": list(self.evidence_block_ids),
        }


@dataclass(frozen=True)
class ResourceUsage:
    """Raw resource state; ratios are deliberately computed offline."""

    historical_tokens_by_layer: Mapping[int, int] = field(default_factory=dict)
    local_tokens_by_layer: Mapping[int, int] = field(default_factory=dict)
    full_history_tokens_by_layer: Mapping[int, int] = field(default_factory=dict)
    kv_bytes_visible: int = 0
    cuda_peak_allocated_bytes: int = 0
    cuda_peak_reserved_bytes: int = 0
    cuda_incremental_peak_allocated_bytes: int = 0
    cuda_incremental_peak_reserved_bytes: int = 0
    attention_query_key_pairs: int = 0
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "historical_tokens_by_layer",
            "local_tokens_by_layer",
            "full_history_tokens_by_layer",
        ):
            object.__setattr__(
                self, name, _non_negative_map(getattr(self, name), name=name)
            )
        for name in (
            "kv_bytes_visible",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "cuda_incremental_peak_allocated_bytes",
            "cuda_incremental_peak_reserved_bytes",
            "attention_query_key_pairs",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if _finite(self.latency_seconds, name="latency_seconds") < 0.0:
            raise ValueError("latency_seconds cannot be negative")

    def with_default_historical_tokens(
        self, values: Mapping[int, int]
    ) -> "ResourceUsage":
        if self.historical_tokens_by_layer:
            return self
        return ResourceUsage(
            historical_tokens_by_layer=values,
            local_tokens_by_layer=self.local_tokens_by_layer,
            full_history_tokens_by_layer=self.full_history_tokens_by_layer,
            kv_bytes_visible=self.kv_bytes_visible,
            cuda_peak_allocated_bytes=self.cuda_peak_allocated_bytes,
            cuda_peak_reserved_bytes=self.cuda_peak_reserved_bytes,
            cuda_incremental_peak_allocated_bytes=(
                self.cuda_incremental_peak_allocated_bytes
            ),
            cuda_incremental_peak_reserved_bytes=(
                self.cuda_incremental_peak_reserved_bytes
            ),
            attention_query_key_pairs=self.attention_query_key_pairs,
            latency_seconds=self.latency_seconds,
        )

    def state_dict(self) -> dict[str, Any]:
        def encoded(values: Mapping[int, int]) -> dict[str, int]:
            return {str(layer): count for layer, count in sorted(values.items())}

        return {
            "historical_tokens_by_layer": encoded(self.historical_tokens_by_layer),
            "local_tokens_by_layer": encoded(self.local_tokens_by_layer),
            "full_history_tokens_by_layer": encoded(self.full_history_tokens_by_layer),
            "kv_bytes_visible": int(self.kv_bytes_visible),
            "cuda_peak_allocated_bytes": int(self.cuda_peak_allocated_bytes),
            "cuda_peak_reserved_bytes": int(self.cuda_peak_reserved_bytes),
            "cuda_incremental_peak_allocated_bytes": int(
                self.cuda_incremental_peak_allocated_bytes
            ),
            "cuda_incremental_peak_reserved_bytes": int(
                self.cuda_incremental_peak_reserved_bytes
            ),
            "attention_query_key_pairs": int(self.attention_query_key_pairs),
            "latency_seconds": float(self.latency_seconds),
        }


@dataclass
class ModelRun:
    """Transient model result. ``distribution_payload`` is never serialized."""

    predicted_text: str
    predicted_token_ids: tuple[int, ...] = ()
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    distribution_payload: Any = None
    state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicted_token_ids = tuple(int(value) for value in self.predicted_token_ids)


@dataclass(frozen=True)
class DistributionState:
    """Compact sufficient statistics for offline NLL/KL/agreement metrics."""

    token_count: int
    target_nll_sum: float
    reference_entropy_sum: float
    reference_cross_entropy_sum: float
    argmax_agreement_count: int
    target_accuracy_count: int

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not 0 <= self.argmax_agreement_count <= self.token_count:
            raise ValueError("argmax_agreement_count is invalid")
        if not 0 <= self.target_accuracy_count <= self.token_count:
            raise ValueError("target_accuracy_count is invalid")
        for name in (
            "target_nll_sum",
            "reference_entropy_sum",
            "reference_cross_entropy_sum",
        ):
            if _finite(getattr(self, name), name=name) < -1e-6:
                raise ValueError(f"{name} cannot be negative")

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "target_nll_sum": self.target_nll_sum,
            "reference_entropy_sum": self.reference_entropy_sum,
            "reference_cross_entropy_sum": self.reference_cross_entropy_sum,
            "argmax_agreement_count": self.argmax_agreement_count,
            "target_accuracy_count": self.target_accuracy_count,
        }


class EvaluationSession(Protocol):
    def cluster_candidates(self) -> Sequence[ClusterCandidate]: ...

    def run_full_context(self) -> ModelRun: ...

    def run_evidence_only(self) -> ModelRun:
        """Run the unconstrained evidence-only upper bound.

        This condition is independent of clustered retrieval: the correct
        source context is the only non-local history visible to the model.
        """
        ...

    def run_local_only(self) -> ModelRun: ...

    def run_with_clusters(
        self,
        selected_cluster_ids: Mapping[int, Sequence[str]],
        *,
        strategy: str,
        budget: int,
    ) -> ModelRun: ...

    def compact_distribution(
        self, reference: ModelRun, candidate: ModelRun
    ) -> DistributionState | None: ...


class EvaluationModel(Protocol):
    @property
    def descriptor(self) -> Mapping[str, Any]: ...

    def open(self, example: EvaluationExample) -> EvaluationSession: ...
