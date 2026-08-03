from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .config import AttentionAggregationConfig
from .contracts import BlockRange


@dataclass(frozen=True)
class TeacherAttentionTarget:
    absolute_block_mass: torch.Tensor
    total_historical_mass: torch.Tensor
    conditional_block_distribution: torch.Tensor
    distribution_basis_mass: torch.Tensor
    per_future_absolute_block_mass: torch.Tensor
    per_layer_head_future_block_mass: torch.Tensor
    metadata: dict


def _select_axis(tensor: torch.Tensor, axis: int, indices: tuple[int, ...] | None) -> torch.Tensor:
    if indices is None:
        return tensor
    if not indices:
        raise ValueError("selected attention indices cannot be empty")
    size = tensor.shape[axis]
    if any(index < 0 or index >= size for index in indices):
        raise IndexError(f"attention selection is outside axis of length {size}")
    index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, axis, index)


def aggregate_teacher_attention(
    attention: torch.Tensor,
    key_logical_positions: torch.Tensor,
    candidate_blocks: tuple[BlockRange, ...],
    config: AttentionAggregationConfig,
) -> TeacherAttentionTarget:
    """Aggregate full-context teacher attention into aligned block labels.

    Parameters
    ----------
    attention:
        Tensor shaped ``[layers, heads, future_queries, keys]`` containing
        normalized teacher attention probabilities.
    key_logical_positions:
        One logical position for each key column.  Positions, rather than
        physical packed-cache offsets, determine block membership.
    candidate_blocks:
        Historical blocks included for this retrieval sample.

    Teacher tensors are reduced only into labels returned by this function;
    they are never mixed with student residual inputs.
    """

    if attention.ndim != 4:
        raise ValueError("attention must have shape [layers, heads, future_queries, keys]")
    if not torch.is_floating_point(attention):
        attention = attention.float()
    if not torch.isfinite(attention).all() or torch.any(attention < 0):
        raise ValueError("attention must contain finite non-negative probabilities")
    if key_logical_positions.ndim != 1 or key_logical_positions.numel() != attention.shape[-1]:
        raise ValueError("key_logical_positions must align with the attention key axis")
    if not candidate_blocks:
        raise ValueError("at least one candidate block is required")

    selected = _select_axis(attention, 0, config.teacher_layers)
    selected = _select_axis(selected, 1, config.teacher_heads)
    reduced = selected.mean(dim=(0, 1))  # [future_queries, keys]

    future_count = reduced.shape[0]
    if config.future_weights is None:
        weights = torch.ones(future_count, dtype=reduced.dtype, device=reduced.device)
    else:
        if len(config.future_weights) != future_count:
            raise ValueError("future_weights must contain one value per future query")
        weights = torch.tensor(config.future_weights, dtype=reduced.dtype, device=reduced.device)
    if config.future_reduction == "mean":
        weights = weights / weights.sum()
    weighted = (reduced * weights[:, None]).sum(dim=0)  # [keys]

    absolute_masses: list[torch.Tensor] = []
    per_future_masses: list[torch.Tensor] = []
    per_layer_head_future_masses: list[torch.Tensor] = []
    for block in candidate_blocks:
        membership = (key_logical_positions >= block.start_position) & (
            key_logical_positions < block.end_position
        )
        absolute_masses.append(weighted[membership].sum())
        per_future_masses.append(reduced[:, membership].sum(dim=-1))
        per_layer_head_future_masses.append(selected[..., membership].sum(dim=-1))
    absolute = torch.stack(absolute_masses)
    per_future_absolute = torch.stack(per_future_masses, dim=-1)
    per_layer_head_future = torch.stack(per_layer_head_future_masses, dim=-1)
    total = absolute.sum()

    if config.length_normalize_blocks:
        lengths = torch.tensor(
            [block.length for block in candidate_blocks],
            dtype=absolute.dtype,
            device=absolute.device,
        )
        distribution_basis = absolute / lengths
    else:
        distribution_basis = absolute
    basis_total = distribution_basis.sum()
    conditional = torch.where(
        basis_total > config.epsilon,
        distribution_basis / basis_total.clamp_min(config.epsilon),
        torch.zeros_like(distribution_basis),
    )

    metadata = {
        **asdict(config),
        "attention_shape": list(attention.shape),
        "selected_layer_count": int(selected.shape[0]),
        "selected_head_count": int(selected.shape[1]),
        "future_query_count": future_count,
        "block_ranges": [block.to_dict() for block in candidate_blocks],
        "absolute_mass_preserved": True,
    }
    return TeacherAttentionTarget(
        absolute_block_mass=absolute,
        total_historical_mass=total.reshape(()),
        conditional_block_distribution=conditional,
        distribution_basis_mass=distribution_basis,
        per_future_absolute_block_mass=per_future_absolute,
        per_layer_head_future_block_mass=per_layer_head_future,
        metadata=metadata,
    )
