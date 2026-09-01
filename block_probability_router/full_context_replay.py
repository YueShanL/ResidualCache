from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from learnable_index.collectors import StudentCollectionConfig
from learnable_index.model_adapter import (
    ModelBundle,
    cache_from_layer_kv,
    forward_tokens,
    hidden_state_at_layer,
    layer_kv_from_cache,
    new_full_dynamic_cache,
)
from learnable_index.planning import RetrievalPlan, SequenceRecord


FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL = "full_context_prefill_posthoc_block_cut_v2"


@dataclass(frozen=True)
class FullContextReplayState:
    """Router and replay state cut from one full-context prefix trajectory."""

    query_summary: torch.Tensor
    block_summaries: torch.Tensor
    block_layer_kv: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]
    local_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_positions: tuple[int, ...]
    forward_calls: int
    forwarded_tokens: int


def _validated_cut_bounds(
    plan: RetrievalPlan,
    *,
    local_context_length: int,
    block_size: int,
) -> tuple[int, int]:
    if local_context_length <= 0 or block_size <= 0:
        raise ValueError("local context length and block size must be positive")
    if local_context_length % block_size:
        raise ValueError("local context length must be divisible by block size")
    local_start = (int(plan.local_context_start) // block_size) * block_size
    local_end = int(plan.local_context_end)
    local_length = local_end - local_start
    if not local_context_length <= local_length < local_context_length + block_size:
        raise RuntimeError("full-context replay local slice is outside block-aligned bounds")
    if local_end != int(plan.first_future_position_affected_by_retrieval):
        raise RuntimeError("full-context prefix must end immediately before the future token")
    if any(block.end_position > local_start for block in plan.candidate_blocks):
        raise RuntimeError("historical candidate overlaps the block-aligned local slice")
    return local_start, local_end


@torch.inference_mode()
def collect_full_context_replay_state(
    bundle: ModelBundle,
    record: SequenceRecord,
    plan: RetrievalPlan,
    student_config: StudentCollectionConfig,
    *,
    block_size: int,
    prefill_chunk_size: int,
    capture_layers: Sequence[int],
) -> FullContextReplayState:
    """Prefill once with full history, then cut router states and replay K/V.

    Every query summary, candidate block summary, historical K/V payload, and
    local K/V entry comes from the same causal full-context trajectory.  The
    cut happens only after the prefix through ``retrieval_position`` has been
    materialized, so the evaluation varies retrieval selection rather than the
    state lineage before the fixed retrieval point.
    """

    if prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    local_start, local_end = _validated_cut_bounds(
        plan,
        local_context_length=student_config.local_context_length,
        block_size=int(block_size),
    )
    layers = tuple(sorted({int(layer) for layer in capture_layers}))
    if not layers or layers[0] < 0:
        raise ValueError("capture_layers must contain non-negative layer indices")

    cache = new_full_dynamic_cache()
    hidden_chunks: list[torch.Tensor] = []
    forward_calls = 0
    for start in range(0, local_end, prefill_chunk_size):
        end = min(start + prefill_chunk_size, local_end)
        output = forward_tokens(
            bundle,
            record.token_ids[start:end],
            range(start, end),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            logical_cache_position=True,
        )
        cache = cache_from_layer_kv(layer_kv_from_cache(output.past_key_values))
        hidden_chunks.append(
            hidden_state_at_layer(output, student_config.residual_layer)[0].detach()
        )
        forward_calls += 1

    pairs = layer_kv_from_cache(cache)
    if any(int(key.shape[2]) != local_end for key, _value in pairs):
        raise RuntimeError("full-context replay source cache has an invalid length")
    if layers[-1] >= len(pairs):
        raise IndexError("full-context replay capture layer is outside the cache")
    hidden = torch.cat(hidden_chunks, dim=0)
    if hidden.shape[0] != local_end:
        raise RuntimeError("full-context residual trajectory has an invalid length")

    if student_config.query_summary == "last":
        query_summary = hidden[local_end - 1]
    else:
        query_length = min(student_config.query_summary_length, local_end)
        query_summary = hidden[local_end - query_length : local_end].mean(dim=0)
    block_summaries = torch.stack(
        [
            hidden[block.start_position : block.end_position].mean(dim=0)
            for block in plan.candidate_blocks
        ]
    )
    block_layer_kv = {
        index: {
            layer: (
                pairs[layer][0][
                    :, :, block.start_position : block.end_position, :
                ].detach(),
                pairs[layer][1][
                    :, :, block.start_position : block.end_position, :
                ].detach(),
            )
            for layer in layers
        }
        for index, block in enumerate(plan.candidate_blocks)
    }
    local_layer_kv = tuple(
        (
            key[:, :, local_start:local_end, :].detach(),
            value[:, :, local_start:local_end, :].detach(),
        )
        for key, value in pairs
    )
    return FullContextReplayState(
        query_summary=query_summary.detach(),
        block_summaries=block_summaries.detach(),
        block_layer_kv=block_layer_kv,
        local_layer_kv=local_layer_kv,
        local_positions=tuple(range(local_start, local_end)),
        forward_calls=forward_calls,
        forwarded_tokens=local_end,
    )


__all__ = [
    "FULL_CONTEXT_REPLAY_SOURCE_PROTOCOL",
    "FullContextReplayState",
    "collect_full_context_replay_state",
]
