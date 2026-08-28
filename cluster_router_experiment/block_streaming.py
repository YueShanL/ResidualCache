"""Block-aligned rolling collection for block-record memory.

This is intentionally separate from :mod:`cluster_router_experiment.streaming`.
The token-record collector maintains an exact local suffix and may emit a
partial oldest block.  A block-record memory instead retains that partial block
locally and unloads only complete mechanical blocks.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, ContextManager

import torch

from learnable_index.contracts import BlockRange
from learnable_index.model_adapter import (
    build_rolling_local_mask,
    cache_from_layer_kv,
    cache_suffix,
    forward_tokens,
    hidden_state_at_layer,
    layer_kv_from_cache,
)
from learnable_index.planning import RetrievalPlan, SequenceRecord, mechanical_blocks


@dataclass(frozen=True)
class EvictedCompleteBlock:
    block: BlockRange
    logical_positions: tuple[int, ...]
    layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class BlockAlignedCollectionResult:
    query_summary: torch.Tensor
    local_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_positions: tuple[int, ...]
    forward_calls: int
    forwarded_tokens: int
    evicted_blocks: int
    evicted_tokens: int
    completed_blocks: int
    maximum_forward_context_length: int
    minimum_final_local_length: int
    maximum_final_local_length: int


class BlockAlignedRollingContextCollector:
    """Advance once while unloading only complete mechanical blocks.

    The physical cache is allowed to contain between ``local_context_length``
    and ``local_context_length + block_size - 1`` tokens at a retrieval point.
    Reaching ``local_context_length + block_size`` atomically unloads the oldest
    complete block and returns the cache to ``local_context_length`` tokens.
    """

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
        on_evict: Callable[[EvictedCompleteBlock], None],
        progress: Callable[[int, int], None] | None = None,
        forward_context: Callable[[], ContextManager] | None = None,
    ) -> BlockAlignedCollectionResult:
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
        ready_blocks: dict[object, BlockRange] = {}
        evicted_blocks = 0
        evicted_tokens = 0
        maximum_seen = 0

        for block in chunks:
            query_positions = tuple(range(block.start_position, block.end_position))
            past_positions = tuple(range(cache_start, block.start_position))
            mask = build_rolling_local_mask(
                self.bundle,
                past_positions=past_positions,
                query_positions=query_positions,
                local_context_length=self.local_context_length + self.block_size,
            )
            context = nullcontext() if forward_context is None else forward_context()
            with context:
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
            cache = cache_from_layer_kv(layer_kv_from_cache(output.past_key_values))
            physical_length = int(layer_kv_from_cache(cache)[0][0].shape[2])
            expected_length = block.end_position - cache_start
            if physical_length != expected_length:
                raise RuntimeError(
                    "block-aligned cache length does not match logical positions"
                )
            maximum_seen = max(maximum_seen, physical_length)

            hidden = hidden_state_at_layer(output, self.residual_layer)[0].detach()
            if block.end_position - block.start_position == self.block_size:
                ready_blocks[block.block_id] = block
                on_block_ready(block, hidden.mean(dim=0))
            recent_hidden = (
                hidden
                if recent_hidden is None
                else torch.cat((recent_hidden, hidden), dim=0)
            )[-self.query_summary_length :]

            while block.end_position - cache_start >= (
                self.local_context_length + self.block_size
            ):
                block_end = cache_start + self.block_size
                block_id = (
                    f"{record.sequence_id}:block:"
                    f"{cache_start:09d}-{block_end:09d}"
                )
                completed = ready_blocks.pop(block_id, None)
                if completed is None:
                    raise RuntimeError(
                        "a complete block reached the unload boundary before its "
                        "router key was ready"
                    )
                pairs = layer_kv_from_cache(cache)
                on_evict(
                    EvictedCompleteBlock(
                        block=completed,
                        logical_positions=tuple(range(cache_start, block_end)),
                        layer_kv=tuple(
                            (
                                key[:, :, : self.block_size, :].detach(),
                                value[:, :, : self.block_size, :].detach(),
                            )
                            for key, value in pairs
                        ),
                    )
                )
                cache_start = block_end
                evicted_blocks += 1
                evicted_tokens += self.block_size
                retained = block.end_position - cache_start
                cache = cache_suffix(cache, retained)
                if progress is not None:
                    progress(evicted_tokens, plan.local_context_start)

        if cache is None:
            raise RuntimeError("rolling collection produced no cache")
        local_positions = tuple(range(cache_start, plan.local_context_end))
        local_length = len(local_positions)
        if plan.local_context_end >= self.local_context_length:
            if not (
                self.local_context_length
                <= local_length
                < self.local_context_length + self.block_size
            ):
                raise RuntimeError("final dynamic local context is outside its bounds")
        if cache_start % self.block_size:
            raise RuntimeError("historical-memory boundary is not block aligned")
        if evicted_tokens != cache_start:
            raise RuntimeError("evicted token count does not match the memory boundary")
        if recent_hidden is None or recent_hidden.shape[0] < self.query_summary_length:
            raise RuntimeError("insufficient residual states for the query summary")
        local_layer_kv = tuple(
            (key.detach(), value.detach()) for key, value in layer_kv_from_cache(cache)
        )
        return BlockAlignedCollectionResult(
            query_summary=recent_hidden.mean(dim=0),
            local_layer_kv=local_layer_kv,
            local_positions=local_positions,
            forward_calls=len(chunks),
            forwarded_tokens=plan.local_context_end,
            evicted_blocks=evicted_blocks,
            evicted_tokens=evicted_tokens,
            completed_blocks=len(full_blocks),
            maximum_forward_context_length=maximum_seen,
            minimum_final_local_length=min(
                self.local_context_length, plan.local_context_end
            ),
            maximum_final_local_length=(
                min(
                    plan.local_context_end,
                    self.local_context_length + self.block_size - 1,
                )
            ),
        )


__all__ = [
    "BlockAlignedCollectionResult",
    "BlockAlignedRollingContextCollector",
    "EvictedCompleteBlock",
]
