"""Single-pass rolling-context collection for the concrete experiment adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from learnable_index.contracts import BlockRange
from learnable_index.model_adapter import (
    build_rolling_local_mask,
    cache_suffix,
    forward_tokens,
    hidden_state_at_layer,
    layer_kv_from_cache,
)
from learnable_index.planning import RetrievalPlan, SequenceRecord, mechanical_blocks


@dataclass(frozen=True)
class EvictedStreamingBlock:
    """The contiguous part of a completed block that just left local context."""

    block: BlockRange
    logical_positions: tuple[int, ...]
    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class RollingCollectionResult:
    query_summary: torch.Tensor
    local_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_positions: tuple[int, ...]
    forward_calls: int
    forwarded_tokens: int
    evicted_blocks: int
    evicted_tokens: int
    completed_blocks: int
    maximum_forward_context_length: int


class RollingContextCollector:
    """Advance once and emit a block exactly when it leaves the local window."""

    def __init__(
        self,
        bundle,
        *,
        local_context_length: int,
        block_size: int,
        residual_layer: int,
        query_summary_length: int,
    ) -> None:
        self.bundle = bundle
        self.local_context_length = int(local_context_length)
        self.block_size = int(block_size)
        self.residual_layer = int(residual_layer)
        self.query_summary_length = int(query_summary_length)
        if min(
            self.local_context_length,
            self.block_size,
            self.query_summary_length,
        ) <= 0:
            raise ValueError("rolling context sizes must be positive")
        if self.local_context_length % self.block_size:
            raise ValueError("local_context_length must be divisible by block_size")

    def collect(
        self,
        record: SequenceRecord,
        plan: RetrievalPlan,
        *,
        on_block_ready: Callable[[BlockRange, torch.Tensor], None],
        on_evict: Callable[[EvictedStreamingBlock], None],
        progress: Callable[[int, int], None] | None = None,
    ) -> RollingCollectionResult:
        full_blocks = mechanical_blocks(
            record.sequence_id, plan.local_context_end, self.block_size
        )
        chunks = list(full_blocks)
        full_end = full_blocks[-1].end_position if full_blocks else 0
        if full_end < plan.local_context_end:
            chunks.append(
                BlockRange(
                    block_id=(
                        f"{record.sequence_id}:stream-tail:"
                        f"{full_end:09d}-{plan.local_context_end:09d}"
                    ),
                    start_position=full_end,
                    end_position=plan.local_context_end,
                )
            )
        if not chunks:
            raise ValueError("rolling prefix contains no tokens")
        cache = None
        cache_start = 0
        recent_hidden: torch.Tensor | None = None
        evicted_block_ids: set[object] = set()
        evicted_tokens = 0
        ready_blocks: dict[object, BlockRange] = {}

        for block in chunks:
            query_positions = tuple(range(block.start_position, block.end_position))
            past_positions = tuple(range(cache_start, block.start_position))
            # The retained cache is 256 tokens.  One complete incoming block is
            # allowed to extend the physical/effective forward context to 320;
            # only after that forward do we unload the oldest block.
            mask = build_rolling_local_mask(
                self.bundle,
                past_positions=past_positions,
                query_positions=query_positions,
                local_context_length=self.local_context_length + self.block_size,
            )
            output = forward_tokens(
                self.bundle,
                record.token_ids[block.start_position : block.end_position],
                query_positions,
                past_key_values=cache,
                attention_mask=mask,
                use_cache=True,
                output_hidden_states=True,
                logical_cache_position=True,
            )
            cache = output.past_key_values
            hidden = hidden_state_at_layer(output, self.residual_layer)[0].detach()
            summary = hidden.mean(dim=0)
            if block.end_position - block.start_position == self.block_size:
                # The learned block key is available while this completed block
                # is still the newest 64-token region of local context, long
                # before its first token can leave a 256-token window.
                ready_blocks[block.block_id] = block
                on_block_ready(block, summary)
            recent_hidden = (
                hidden
                if recent_hidden is None
                else torch.cat((recent_hidden, hidden), dim=0)
            )[-self.query_summary_length :]

            new_cache_start = max(0, block.end_position - self.local_context_length)
            exit_start = cache_start
            while exit_start < new_cache_start:
                mechanical_start = (exit_start // self.block_size) * self.block_size
                mechanical_end = mechanical_start + self.block_size
                mechanical_id = (
                    f"{record.sequence_id}:block:"
                    f"{mechanical_start:09d}-{mechanical_end:09d}"
                )
                evicted_block = ready_blocks.get(mechanical_id)
                if evicted_block is None:
                    raise RuntimeError(
                        "a token left local context before its block key was ready"
                    )
                exit_end = min(new_cache_start, mechanical_end)
                positions = tuple(range(exit_start, exit_end))
                start_index = exit_start - cache_start
                end_index = exit_end - cache_start
                pairs = layer_kv_from_cache(cache)
                layer_kv = tuple(
                    (
                        key[:, :, start_index:end_index, :].detach(),
                        value[:, :, start_index:end_index, :].detach(),
                    )
                    for key, value in pairs
                )
                on_evict(
                    EvictedStreamingBlock(
                        block=evicted_block,
                        logical_positions=positions,
                        layer_kv=layer_kv,
                    )
                )
                evicted_block_ids.add(evicted_block.block_id)
                evicted_tokens += len(positions)
                if progress is not None:
                    progress(evicted_tokens, plan.local_context_start)
                exit_start = exit_end
            if new_cache_start > cache_start:
                cache = cache_suffix(cache, self.local_context_length)
                cache_start = new_cache_start

        if evicted_tokens != plan.local_context_start:
            raise RuntimeError("rolling collection did not ingest every evicted token")
        local_positions = tuple(range(cache_start, plan.local_context_end))
        if local_positions != tuple(range(plan.local_context_start, plan.local_context_end)):
            raise RuntimeError("final rolling cache is not the planned local context")
        if recent_hidden is None or recent_hidden.shape[0] < self.query_summary_length:
            raise RuntimeError("insufficient residual states for the query summary")
        if cache is None:
            raise RuntimeError("rolling collection produced no cache")
        local_layer_kv = tuple(
            (key.detach(), value.detach()) for key, value in layer_kv_from_cache(cache)
        )
        return RollingCollectionResult(
            query_summary=recent_hidden.mean(dim=0),
            local_layer_kv=local_layer_kv,
            local_positions=local_positions,
            forward_calls=len(chunks),
            forwarded_tokens=plan.local_context_end,
            evicted_blocks=len(evicted_block_ids),
            evicted_tokens=evicted_tokens,
            completed_blocks=len(full_blocks),
            maximum_forward_context_length=self.local_context_length
            + self.block_size,
        )


__all__ = [
    "EvictedStreamingBlock",
    "RollingCollectionResult",
    "RollingContextCollector",
]
