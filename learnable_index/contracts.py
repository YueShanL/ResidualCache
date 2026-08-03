from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


SCHEMA_VERSION = 1
STUDENT_STATE_SOURCE = "student_restricted"


@dataclass(frozen=True)
class BlockRange:
    """A half-open logical token range belonging to one historical block."""

    block_id: str
    start_position: int
    end_position: int

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must be non-empty")
        if self.start_position < 0 or self.end_position <= self.start_position:
            raise ValueError("block positions must form a non-empty half-open range")

    @property
    def length(self) -> int:
        return self.end_position - self.start_position

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start_position": self.start_position,
            "end_position": self.end_position,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BlockRange":
        return cls(
            block_id=str(payload["block_id"]),
            start_position=int(payload["start_position"]),
            end_position=int(payload["end_position"]),
        )


@dataclass
class RetrievalSample:
    """Tensor contract for one supervised retrieval point.

    Teacher attention appears only in label fields.  Query and block summaries
    must declare that they came from the restricted student trajectory.
    """

    sample_id: str
    sequence_id: str
    retrieval_position: int
    first_future_position_affected_by_retrieval: int
    future_horizon_length: int
    local_context_start: int
    local_context_end: int
    candidate_blocks: tuple[BlockRange, ...]
    query_summary: torch.Tensor
    block_summaries: torch.Tensor
    absolute_teacher_block_mass: torch.Tensor
    total_teacher_historical_mass: torch.Tensor
    conditional_teacher_distribution: torch.Tensor
    per_future_teacher_block_mass: torch.Tensor | None = None
    teacher_layer_head_future_block_mass: torch.Tensor | None = None
    aggregation_metadata: dict[str, Any] = field(default_factory=dict)
    logical_position_metadata: dict[str, Any] = field(default_factory=dict)
    query_state_source: str = STUDENT_STATE_SOURCE
    block_state_source: str = STUDENT_STATE_SOURCE

    def validate(self, *, atol: float = 1e-5) -> "RetrievalSample":
        if not self.sample_id or not self.sequence_id:
            raise ValueError("sample_id and sequence_id must be non-empty")
        if self.local_context_start < 0 or self.local_context_end <= self.local_context_start:
            raise ValueError("local context must be a non-empty half-open range")
        if not self.local_context_start <= self.retrieval_position < self.local_context_end:
            raise ValueError("retrieval_position must lie inside the local context")
        if self.first_future_position_affected_by_retrieval <= self.retrieval_position:
            raise ValueError("retrieval can only affect a subsequent forward")
        if self.future_horizon_length <= 0:
            raise ValueError("future_horizon_length must be positive")
        if self.query_state_source != STUDENT_STATE_SOURCE:
            raise ValueError("query summary must come from the restricted student trajectory")
        if self.block_state_source != STUDENT_STATE_SOURCE:
            raise ValueError("block summaries must come from the restricted student trajectory")
        if not self.candidate_blocks:
            raise ValueError("at least one candidate block is required")
        block_ids = [block.block_id for block in self.candidate_blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("candidate block ids must be unique")
        if any(block.end_position > self.local_context_start for block in self.candidate_blocks):
            raise ValueError("candidate blocks must be entirely outside the current local window")

        tensors = {
            "query_summary": self.query_summary,
            "block_summaries": self.block_summaries,
            "absolute_teacher_block_mass": self.absolute_teacher_block_mass,
            "total_teacher_historical_mass": self.total_teacher_historical_mass,
            "conditional_teacher_distribution": self.conditional_teacher_distribution,
        }
        if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError("sample tensors must contain only finite values")
        if self.query_summary.ndim != 1:
            raise ValueError("query_summary must have shape [residual_dim]")
        if self.block_summaries.ndim != 2:
            raise ValueError("block_summaries must have shape [blocks, residual_dim]")
        block_count = len(self.candidate_blocks)
        if self.block_summaries.shape != (block_count, self.query_summary.shape[0]):
            raise ValueError("query and block summary dimensions do not match")
        for name, tensor in (
            ("absolute_teacher_block_mass", self.absolute_teacher_block_mass),
            ("conditional_teacher_distribution", self.conditional_teacher_distribution),
        ):
            if tensor.shape != (block_count,):
                raise ValueError(f"{name} must have one value per candidate block")
            if torch.any(tensor < 0):
                raise ValueError(f"{name} cannot contain negative values")
        if self.total_teacher_historical_mass.numel() != 1:
            raise ValueError("total_teacher_historical_mass must be scalar")
        total = float(self.total_teacher_historical_mass.reshape(()))
        if total < 0:
            raise ValueError("total_teacher_historical_mass cannot be negative")
        absolute_sum = float(self.absolute_teacher_block_mass.sum())
        if abs(total - absolute_sum) > atol * max(1.0, total, absolute_sum):
            raise ValueError("total historical mass must equal the saved absolute block masses")
        conditional_sum = float(self.conditional_teacher_distribution.sum())
        if total > atol and abs(conditional_sum - 1.0) > atol:
            raise ValueError("conditional teacher distribution must sum to one")
        if total <= atol and conditional_sum > atol:
            raise ValueError("zero-demand samples must have a zero conditional distribution")
        if self.per_future_teacher_block_mass is not None:
            per_future = self.per_future_teacher_block_mass
            if per_future.shape != (self.future_horizon_length, block_count):
                raise ValueError(
                    "per_future_teacher_block_mass must have shape [future_horizon, blocks]"
                )
            if not torch.isfinite(per_future).all() or torch.any(per_future < 0):
                raise ValueError("per-future teacher block mass must be finite and non-negative")
        if self.teacher_layer_head_future_block_mass is not None:
            raw = self.teacher_layer_head_future_block_mass
            if raw.ndim != 4:
                raise ValueError(
                    "teacher_layer_head_future_block_mass must have shape "
                    "[layers, heads, future_horizon, blocks]"
                )
            if raw.shape[2:] != (self.future_horizon_length, block_count):
                raise ValueError("raw teacher aggregation does not align with horizon and blocks")
            if not torch.isfinite(raw).all() or torch.any(raw < 0):
                raise ValueError("raw teacher aggregation must be finite and non-negative")
        return self

    @property
    def residual_dim(self) -> int:
        return int(self.query_summary.shape[0])

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "sequence_id": self.sequence_id,
            "retrieval_position": self.retrieval_position,
            "first_future_position_affected_by_retrieval": (
                self.first_future_position_affected_by_retrieval
            ),
            "future_horizon_length": self.future_horizon_length,
            "local_context_start": self.local_context_start,
            "local_context_end": self.local_context_end,
            "candidate_blocks": [block.to_dict() for block in self.candidate_blocks],
            "query_summary": self.query_summary.detach().cpu(),
            "block_summaries": self.block_summaries.detach().cpu(),
            "absolute_teacher_block_mass": self.absolute_teacher_block_mass.detach().cpu(),
            "total_teacher_historical_mass": (
                self.total_teacher_historical_mass.detach().cpu().reshape(())
            ),
            "conditional_teacher_distribution": (
                self.conditional_teacher_distribution.detach().cpu()
            ),
            "per_future_teacher_block_mass": (
                None
                if self.per_future_teacher_block_mass is None
                else self.per_future_teacher_block_mass.detach().cpu()
            ),
            "teacher_layer_head_future_block_mass": (
                None
                if self.teacher_layer_head_future_block_mass is None
                else self.teacher_layer_head_future_block_mass.detach().cpu()
            ),
            "aggregation_metadata": dict(self.aggregation_metadata),
            "logical_position_metadata": dict(self.logical_position_metadata),
            "query_state_source": self.query_state_source,
            "block_state_source": self.block_state_source,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RetrievalSample":
        version = int(payload.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported sample schema version: {version}")
        sample = cls(
            sample_id=str(payload["sample_id"]),
            sequence_id=str(payload["sequence_id"]),
            retrieval_position=int(payload["retrieval_position"]),
            first_future_position_affected_by_retrieval=int(
                payload["first_future_position_affected_by_retrieval"]
            ),
            future_horizon_length=int(payload["future_horizon_length"]),
            local_context_start=int(payload["local_context_start"]),
            local_context_end=int(payload["local_context_end"]),
            candidate_blocks=tuple(
                BlockRange.from_dict(item) for item in payload["candidate_blocks"]
            ),
            query_summary=torch.as_tensor(payload["query_summary"]).float(),
            block_summaries=torch.as_tensor(payload["block_summaries"]).float(),
            absolute_teacher_block_mass=torch.as_tensor(
                payload["absolute_teacher_block_mass"]
            ).float(),
            total_teacher_historical_mass=torch.as_tensor(
                payload["total_teacher_historical_mass"]
            ).float().reshape(()),
            conditional_teacher_distribution=torch.as_tensor(
                payload["conditional_teacher_distribution"]
            ).float(),
            per_future_teacher_block_mass=(
                None
                if payload.get("per_future_teacher_block_mass") is None
                else torch.as_tensor(payload["per_future_teacher_block_mass"]).float()
            ),
            teacher_layer_head_future_block_mass=(
                None
                if payload.get("teacher_layer_head_future_block_mass") is None
                else torch.as_tensor(payload["teacher_layer_head_future_block_mass"]).float()
            ),
            aggregation_metadata=dict(payload.get("aggregation_metadata", {})),
            logical_position_metadata=dict(payload.get("logical_position_metadata", {})),
            query_state_source=str(payload.get("query_state_source", "")),
            block_state_source=str(payload.get("block_state_source", "")),
        )
        return sample.validate()
