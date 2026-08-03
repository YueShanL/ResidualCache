from __future__ import annotations

import torch

from .contracts import BlockRange, RetrievalSample


def make_synthetic_samples(
    *,
    sample_count: int = 128,
    residual_dim: int = 16,
    min_blocks: int = 3,
    max_blocks: int = 8,
    block_size: int = 8,
    local_context_length: int = 256,
    future_horizon_length: int = 8,
    seed: int = 13,
) -> list[RetrievalSample]:
    """Create a deterministic, learnable contract-level smoke dataset."""

    if sample_count <= 0 or residual_dim <= 0:
        raise ValueError("sample_count and residual_dim must be positive")
    if min_blocks <= 0 or max_blocks < min_blocks:
        raise ValueError("invalid candidate block range")
    generator = torch.Generator().manual_seed(seed)
    teacher_map = torch.randn(residual_dim, residual_dim, generator=generator) / residual_dim**0.5
    samples: list[RetrievalSample] = []
    for sample_index in range(sample_count):
        block_count = int(
            torch.randint(min_blocks, max_blocks + 1, (1,), generator=generator).item()
        )
        query = torch.randn(residual_dim, generator=generator)
        blocks = torch.randn(block_count, residual_dim, generator=generator)
        teacher_logits = blocks @ (teacher_map @ query) / residual_dim**0.5
        conditional = teacher_logits.softmax(dim=0)
        demand = torch.sigmoid(0.8 * query[0] - 0.3 * query[1]).clamp(1e-4, 1 - 1e-4)
        absolute = conditional * demand

        candidate_blocks = tuple(
            BlockRange(
                block_id=f"sequence-{sample_index:05d}-block-{block_index:03d}",
                start_position=block_index * block_size,
                end_position=(block_index + 1) * block_size,
            )
            for block_index in range(block_count)
        )
        local_start = block_count * block_size + 32
        local_end = local_start + local_context_length
        retrieval_position = local_end - 1
        samples.append(
            RetrievalSample(
                sample_id=f"synthetic-{sample_index:05d}",
                sequence_id=f"sequence-{sample_index:05d}",
                retrieval_position=retrieval_position,
                first_future_position_affected_by_retrieval=retrieval_position + 1,
                future_horizon_length=future_horizon_length,
                local_context_start=local_start,
                local_context_end=local_end,
                candidate_blocks=candidate_blocks,
                query_summary=query,
                block_summaries=blocks,
                absolute_teacher_block_mass=absolute,
                total_teacher_historical_mass=demand.reshape(()),
                conditional_teacher_distribution=conditional,
                aggregation_metadata={
                    "source": "synthetic_smoke",
                    "future_reduction": "mean",
                    "absolute_mass_preserved": True,
                },
                logical_position_metadata={
                    "position_semantics": "logical_half_open",
                    "local_context_length": local_context_length,
                },
            ).validate()
        )
    return samples

