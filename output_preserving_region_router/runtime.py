from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from block_probability_router.streaming_collection import (
    StreamingStudentState,
    collect_streaming_student_state,
)
from learnable_index.collectors import StudentCollectionConfig
from learnable_index.model_adapter import (
    build_rolling_local_mask,
    cache_from_layer_kv,
    forward_tokens,
    layer_kv_from_cache,
    new_full_dynamic_cache,
)
from learnable_index.planning import RetrievalPlan, SequenceRecord
from residual_cache.gemma4_memory_adapter import Gemma4StaticKVAdapter

from .gated_kv_adapter import Gemma4SoftBlockGateAdapter


@dataclass(frozen=True)
class TrainingStudentState:
    query_summary: torch.Tensor
    block_summaries: torch.Tensor
    block_layer_kv: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]
    local_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    local_positions: tuple[int, ...]


def physical_full_attention_layers(bundle) -> tuple[int, ...]:
    layer_types = tuple(
        bundle.text_config.layer_types[: bundle.physical_cache_layer_count]
    )
    layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )
    if not layers:
        raise ValueError("Gemma 4 model exposes no physical full-attention layers")
    return layers


def _normal_clone(value: torch.Tensor) -> torch.Tensor:
    # Frozen-model collection runs under inference_mode.  Such tensors cannot
    # be saved for backward even when requires_grad=False, so materialize a
    # normal detached tensor before it enters the differentiable student pass.
    return value.detach().clone()


def collect_training_student_state(
    bundle,
    record: SequenceRecord,
    plan: RetrievalPlan,
    student_config: StudentCollectionConfig,
    *,
    block_size: int,
    capture_layers: Sequence[int],
) -> TrainingStudentState:
    state: StreamingStudentState = collect_streaming_student_state(
        bundle,
        record,
        plan,
        student_config,
        block_size=block_size,
        capture_layers=capture_layers,
    )
    return TrainingStudentState(
        query_summary=_normal_clone(state.query_summary),
        block_summaries=_normal_clone(state.block_summaries),
        block_layer_kv={
            block: {
                layer: (_normal_clone(key), _normal_clone(value))
                for layer, (key, value) in layer_kv.items()
            }
            for block, layer_kv in state.block_layer_kv.items()
        },
        local_layer_kv=tuple(
            (_normal_clone(key), _normal_clone(value))
            for key, value in state.local_layer_kv
        ),
        local_positions=state.local_positions,
    )


def pack_block_layer_kv(
    state: TrainingStudentState,
    indices: Sequence[int],
    layers: Sequence[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    selected = tuple(int(index) for index in indices)
    if len(set(selected)) != len(selected) or any(index < 0 for index in selected):
        raise ValueError("selected block indices must be unique and non-negative")
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        pairs = []
        for index in selected:
            try:
                pairs.append(state.block_layer_kv[index][int(layer)])
            except KeyError as error:
                raise ValueError(
                    f"streaming state is missing block={index}, layer={layer}"
                ) from error
        if pairs:
            result[int(layer)] = (
                torch.cat([pair[0] for pair in pairs], dim=2),
                torch.cat([pair[1] for pair in pairs], dim=2),
            )
    return result


@torch.inference_mode()
def full_context_future_logits(
    bundle,
    record: SequenceRecord,
    plan: RetrievalPlan,
    *,
    prefill_chunk_size: int,
) -> torch.Tensor:
    """Return full-context logits at every future query position."""

    if prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    cache = new_full_dynamic_cache()
    for start in range(0, plan.future_start, prefill_chunk_size):
        end = min(plan.future_start, start + prefill_chunk_size)
        output = forward_tokens(
            bundle,
            record.token_ids[start:end],
            range(start, end),
            past_key_values=cache,
            use_cache=True,
            logical_cache_position=True,
        )
        cache = output.past_key_values
    output = forward_tokens(
        bundle,
        record.token_ids[plan.future_start : plan.future_end],
        range(plan.future_start, plan.future_end),
        past_key_values=cache,
        use_cache=True,
        logical_cache_position=True,
    )
    logits = output.logits[:, -plan.future_horizon_length :, :].detach().float().clone()
    if logits.shape[1] != plan.future_horizon_length:
        raise RuntimeError("full-context forward did not return the future horizon")
    return logits


def _future_forward_kwargs(bundle, record, plan, state, local_context_length, block_size):
    positions = tuple(range(plan.future_start, plan.future_end))
    mask = build_rolling_local_mask(
        bundle,
        past_positions=state.local_positions,
        query_positions=positions,
        local_context_length=local_context_length + block_size,
    )
    return {
        "input_ids": torch.tensor(
            [record.token_ids[plan.future_start : plan.future_end]],
            dtype=torch.long,
            device=bundle.input_device,
        ),
        "position_ids": torch.tensor(
            [positions], dtype=torch.long, device=bundle.input_device
        ),
        "past_key_values": cache_from_layer_kv(state.local_layer_kv),
        "attention_mask": mask,
        "use_cache": True,
        "return_dict": True,
    }


def soft_gated_future_logits(
    bundle,
    record: SequenceRecord,
    plan: RetrievalPlan,
    state: TrainingStudentState,
    block_gates: torch.Tensor,
    *,
    full_attention_layers: Sequence[int],
    local_context_length: int,
    block_size: int,
    gate_epsilon: float,
) -> torch.Tensor:
    if block_gates.shape != (1, len(plan.candidate_blocks)):
        raise ValueError("block gates must align with the retrieval plan")
    all_indices = tuple(range(len(plan.candidate_blocks)))
    historical = pack_block_layer_kv(state, all_indices, full_attention_layers)
    block_lengths = tuple(block.length for block in plan.candidate_blocks)
    adapter = Gemma4SoftBlockGateAdapter(
        bundle.model,
        historical,
        block_gates,
        block_lengths,
        gate_epsilon=gate_epsilon,
    )
    with adapter:
        output = bundle.model(
            **_future_forward_kwargs(
                bundle,
                record,
                plan,
                state,
                local_context_length,
                block_size,
            )
        )
    logits = output.logits[:, -plan.future_horizon_length :, :]
    if logits.shape[1] != plan.future_horizon_length:
        raise RuntimeError("gated student did not return the future horizon")
    return logits


@torch.inference_mode()
def hard_region_future_logits(
    bundle,
    record: SequenceRecord,
    plan: RetrievalPlan,
    state: TrainingStudentState,
    selected_indices: Sequence[int],
    *,
    full_attention_layers: Sequence[int],
    local_context_length: int,
    block_size: int,
) -> torch.Tensor:
    historical = pack_block_layer_kv(
        state, selected_indices, full_attention_layers
    )
    adapter = (
        Gemma4StaticKVAdapter(bundle.model, historical)
        if historical
        else None
    )
    if adapter is None:
        output = bundle.model(
            **_future_forward_kwargs(
                bundle,
                record,
                plan,
                state,
                local_context_length,
                block_size,
            )
        )
    else:
        with adapter:
            output = bundle.model(
                **_future_forward_kwargs(
                    bundle,
                    record,
                    plan,
                    state,
                    local_context_length,
                    block_size,
                )
            )
    return output.logits[:, -plan.future_horizon_length :, :].detach().float().clone()


__all__ = [
    "TrainingStudentState",
    "collect_training_student_state",
    "full_context_future_logits",
    "hard_region_future_logits",
    "pack_block_layer_kv",
    "physical_full_attention_layers",
    "soft_gated_future_logits",
]
