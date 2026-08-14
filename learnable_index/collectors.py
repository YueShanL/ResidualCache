from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch

from .config import AttentionAggregationConfig
from .contracts import RetrievalSample
from .kv_store import KVBlock, KVBlockStore
from .model_adapter import (
    ModelBundle,
    extract_cache_token_range,
    forward_tokens,
    hidden_state_at_layer,
    layer_kv_from_cache,
    new_full_dynamic_cache,
)
from .planning import RetrievalPlan, SequenceRecord
from .targets import TeacherAttentionTarget, aggregate_teacher_attention


@dataclass(frozen=True)
class StudentCollectionConfig:
    local_context_length: int = 256
    residual_layer: int = -1
    query_summary: Literal["last", "mean"] = "mean"
    query_summary_length: int = 16

    def __post_init__(self) -> None:
        if self.local_context_length <= 0 or self.query_summary_length <= 0:
            raise ValueError("student context and query summary lengths must be positive")
        if self.query_summary not in {"last", "mean"}:
            raise ValueError("query_summary must be 'last' or 'mean'")


class TeacherAttentionCollector:
    def __init__(
        self,
        bundle: ModelBundle,
        aggregation_config: AttentionAggregationConfig,
        *,
        prefill_chunk_size: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.aggregation_config = aggregation_config
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive when set")
        self.prefill_chunk_size = prefill_chunk_size

    def collect(self, record: SequenceRecord, plan: RetrievalPlan) -> TeacherAttentionTarget:
        teacher_end = plan.future_end
        if teacher_end > len(record.token_ids):
            raise ValueError("teacher horizon extends beyond the aligned sequence")
        future_start = plan.future_start
        layer_count = int(self.bundle.text_config.num_hidden_layers)
        selected_layer_indices = (
            tuple(range(layer_count))
            if self.aggregation_config.teacher_layers is None
            else self.aggregation_config.teacher_layers
        )
        if any(index < 0 or index >= layer_count for index in selected_layer_indices):
            raise IndexError("configured teacher layer is outside the model")

        captured: dict[int, torch.Tensor] = {}
        handles = []
        decoder_layers = getattr(self.bundle.text_model, "layers", None)
        attention_modules = []
        if decoder_layers is not None:
            for layer_index in selected_layer_indices:
                attention_module = getattr(decoder_layers[layer_index], "self_attn", None)
                if attention_module is None:
                    attention_modules = []
                    break
                attention_modules.append((layer_index, attention_module))

        chunked = self.prefill_chunk_size is not None and bool(attention_modules)
        for layer_index, attention_module in attention_modules:
            def capture(_module, _inputs, output, *, index=layer_index):
                weights = output[1] if isinstance(output, tuple) else None
                if weights is None or weights.ndim != 4:
                    raise RuntimeError(f"teacher hook did not receive attention at layer {index}")
                query_weights = weights[0, :, :, :teacher_end]
                if not chunked:
                    query_weights = query_weights[:, future_start:teacher_end]
                if query_weights.shape[-2:] != (
                    teacher_end - future_start,
                    teacher_end,
                ):
                    raise RuntimeError(
                        f"teacher attention shape mismatch at layer {index}: "
                        f"{tuple(query_weights.shape)}"
                    )
                captured[index] = query_weights.detach().float().cpu()

            handles.append(attention_module.register_forward_hook(capture))
        if handles:
            try:
                if chunked:
                    cache = new_full_dynamic_cache()
                    chunk_size = int(self.prefill_chunk_size)
                    # Hooks must observe only the final answer horizon. Disable
                    # them while materializing the linear-memory full prefix.
                    for handle in handles:
                        handle.remove()
                    handles = []
                    for start in range(0, future_start, chunk_size):
                        end = min(start + chunk_size, future_start)
                        forward_tokens(
                            self.bundle,
                            record.token_ids[start:end],
                            range(start, end),
                            past_key_values=cache,
                            use_cache=True,
                            output_attentions=False,
                        )
                    for layer_index, attention_module in attention_modules:
                        def capture_chunked(_module, _inputs, output, *, index=layer_index):
                            weights = output[1] if isinstance(output, tuple) else None
                            if weights is None or weights.ndim != 4:
                                raise RuntimeError(
                                    f"teacher hook did not receive attention at layer {index}"
                                )
                            query_weights = weights[0, :, :, :teacher_end]
                            if query_weights.shape[-2:] != (
                                teacher_end - future_start,
                                teacher_end,
                            ):
                                raise RuntimeError(
                                    f"chunked teacher attention shape mismatch at layer {index}: "
                                    f"{tuple(query_weights.shape)}"
                                )
                            captured[index] = query_weights.detach().float().cpu()

                        handles.append(attention_module.register_forward_hook(capture_chunked))
                    forward_tokens(
                        self.bundle,
                        record.token_ids[future_start:teacher_end],
                        range(future_start, teacher_end),
                        past_key_values=cache,
                        use_cache=False,
                        output_attentions=False,
                    )
                else:
                    forward_tokens(
                        self.bundle,
                        record.token_ids[:teacher_end],
                        range(teacher_end),
                        use_cache=False,
                        output_attentions=False,
                    )
            finally:
                for handle in handles:
                    handle.remove()
            missing = set(selected_layer_indices) - set(captured)
            if missing:
                raise RuntimeError(f"teacher attention hooks missed layers: {sorted(missing)}")
            selected = torch.stack([captured[index] for index in selected_layer_indices])
            aggregation_config = replace(self.aggregation_config, teacher_layers=None)
            collection_mode = (
                "chunked_prefix_selected_layer_eager_hooks"
                if chunked
                else "selected_layer_eager_hooks"
            )
        else:
            # Test doubles and compatible non-Gemma wrappers may expose only
            # public output_attentions. This path retains strict validation.
            output = forward_tokens(
                self.bundle,
                record.token_ids[:teacher_end],
                range(teacher_end),
                use_cache=False,
                output_attentions=True,
            )
            all_layers = output.attentions
            if all_layers is None or len(all_layers) != layer_count:
                raise RuntimeError("model output did not contain every teacher attention layer")
            selected = torch.stack(
                [
                    all_layers[index][0, :, future_start:teacher_end, :teacher_end]
                    .detach()
                    .float()
                    .cpu()
                    for index in selected_layer_indices
                ]
            )
            aggregation_config = replace(self.aggregation_config, teacher_layers=None)
            collection_mode = "public_output_attentions_fallback"
        target = aggregate_teacher_attention(
            selected,
            torch.arange(teacher_end),
            plan.candidate_blocks,
            aggregation_config,
        )
        return TeacherAttentionTarget(
            absolute_block_mass=target.absolute_block_mass.detach().float().cpu(),
            total_historical_mass=target.total_historical_mass.detach().float().cpu(),
            conditional_block_distribution=(
                target.conditional_block_distribution.detach().float().cpu()
            ),
            distribution_basis_mass=target.distribution_basis_mass.detach().float().cpu(),
            per_future_absolute_block_mass=(
                target.per_future_absolute_block_mass.detach().float().cpu()
            ),
            per_layer_head_future_block_mass=(
                target.per_layer_head_future_block_mass.detach().float().cpu()
            ),
            metadata={
                **target.metadata,
                "collector": "full_context_teacher",
                "collection_mode": collection_mode,
                "selected_teacher_layers_original": list(selected_layer_indices),
                "teacher_visible_range": [0, teacher_end],
                "future_query_range": [future_start, teacher_end],
                "information_boundary": "labels_only",
                "teacher_prefill_chunk_size": self.prefill_chunk_size,
            },
        )


class RestrictedStudentCollector:
    def __init__(self, bundle: ModelBundle, config: StudentCollectionConfig) -> None:
        self.bundle = bundle
        self.config = config

    def _forward_window(self, token_ids: tuple[int, ...], start: int, end: int):
        if not 0 <= start < end <= len(token_ids):
            raise ValueError("student window is outside the aligned sequence")
        cache = new_full_dynamic_cache()
        output = forward_tokens(
            self.bundle,
            token_ids[start:end],
            range(start, end),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        physical_count = len(output.past_key_values.layers)
        if physical_count != self.bundle.physical_cache_layer_count:
            raise RuntimeError(
                f"student cache has {physical_count} physical layers, expected "
                f"{self.bundle.physical_cache_layer_count}"
            )
        return output

    def collect_query(self, record: SequenceRecord, plan: RetrievalPlan) -> torch.Tensor:
        if plan.local_context_end - plan.local_context_start > self.config.local_context_length:
            raise ValueError("retrieval plan exceeds the configured student context boundary")
        output = self._forward_window(
            record.token_ids,
            plan.local_context_start,
            plan.local_context_end,
        )
        hidden = hidden_state_at_layer(output, self.config.residual_layer)[0]
        if self.config.query_summary == "last":
            summary = hidden[-1]
        else:
            summary = hidden[-min(self.config.query_summary_length, hidden.shape[0]) :].mean(dim=0)
        return summary.detach().float().cpu()

    def collect_local_cache(self, record: SequenceRecord, plan: RetrievalPlan):
        """Return the local-only cache at the decision point for schedule-correct replay."""

        output = self._forward_window(
            record.token_ids,
            plan.local_context_start,
            plan.local_context_end,
        )
        return tuple(
            (key.detach(), value.detach()) for key, value in layer_kv_from_cache(output.past_key_values)
        )

    def collect_block(self, record: SequenceRecord, block) -> KVBlock:
        context_end = block.end_position
        context_start = max(0, context_end - self.config.local_context_length)
        output = self._forward_window(record.token_ids, context_start, context_end)
        hidden = hidden_state_at_layer(output, self.config.residual_layer)[0]
        block_start_index = block.start_position - context_start
        block_end_index = block.end_position - context_start
        if block_start_index < 0:
            raise RuntimeError("block is larger than the restricted collection context")
        summary = hidden[block_start_index:block_end_index].mean(dim=0).detach().float().cpu()
        layer_kv = extract_cache_token_range(
            output.past_key_values,
            input_length=context_end - context_start,
            start_index=block_start_index,
            end_index=block_end_index,
        )
        return KVBlock(
            block=block,
            sequence_id=record.sequence_id,
            token_ids=record.token_ids[block.start_position:block.end_position],
            logical_positions=tuple(range(block.start_position, block.end_position)),
            layer_kv=layer_kv,
            residual_summary=summary,
            model_fingerprint=self.bundle.fingerprint,
            metadata={
                "collector": "restricted_student",
                "visible_context": [context_start, context_end],
                "residual_layer": self.config.residual_layer,
                "pooling": "mean",
                "config": asdict(self.config),
            },
        ).validate()

    def ensure_blocks(
        self,
        record: SequenceRecord,
        plan: RetrievalPlan,
        store: KVBlockStore,
        *,
        persist: bool = True,
    ) -> list[KVBlock]:
        blocks: list[KVBlock] = []
        for block_range in plan.candidate_blocks:
            if not persist:
                blocks.append(self.collect_block(record, block_range))
                continue
            if not store.contains(block_range.block_id):
                store.save(self.collect_block(record, block_range))
            block = store.load(block_range.block_id)
            if block.sequence_id != record.sequence_id:
                raise RuntimeError("KV block sequence identity mismatch")
            if block.block != block_range:
                raise RuntimeError("KV block logical range mismatch")
            if block.model_fingerprint != self.bundle.fingerprint:
                raise RuntimeError("KV block was collected from a different model fingerprint")
            if block.metadata.get("config") != asdict(self.config):
                raise RuntimeError(
                    "KV block was collected with a different restricted-student configuration"
                )
            blocks.append(block)
        return blocks


def assemble_retrieval_sample(
    record: SequenceRecord,
    plan: RetrievalPlan,
    query_summary: torch.Tensor,
    blocks: list[KVBlock],
    target: TeacherAttentionTarget,
) -> RetrievalSample:
    if [block.block.block_id for block in blocks] != [
        block.block_id for block in plan.candidate_blocks
    ]:
        raise ValueError("candidate block order is not aligned across student and teacher data")
    block_summaries = torch.stack([block.residual_summary for block in blocks])
    return RetrievalSample(
        sample_id=plan.sample_id,
        sequence_id=record.sequence_id,
        retrieval_position=plan.retrieval_position,
        first_future_position_affected_by_retrieval=(
            plan.first_future_position_affected_by_retrieval
        ),
        future_horizon_length=plan.future_horizon_length,
        local_context_start=plan.local_context_start,
        local_context_end=plan.local_context_end,
        candidate_blocks=plan.candidate_blocks,
        query_summary=query_summary,
        block_summaries=block_summaries,
        absolute_teacher_block_mass=target.absolute_block_mass,
        total_teacher_historical_mass=target.total_historical_mass,
        conditional_teacher_distribution=target.conditional_block_distribution,
        per_future_teacher_block_mass=target.per_future_absolute_block_mass,
        teacher_layer_head_future_block_mass=target.per_layer_head_future_block_mass,
        aggregation_metadata=target.metadata,
        logical_position_metadata={
            "tokenization": "single_aligned_sequence",
            "position_semantics": "original_logical_positions",
            "student_visible_range": [plan.local_context_start, plan.local_context_end],
            "teacher_visible_range": [0, plan.future_end],
            "first_future_position_affected_by_retrieval": plan.future_start,
            "student_memory_policy": "local_only_no_recurrent_retrieval_during_collection",
            "split_group_id": record.metadata.get(
                "split_group_id", record.sequence_id
            ),
        },
    ).validate()
