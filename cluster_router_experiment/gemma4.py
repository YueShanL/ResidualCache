from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from cluster_router_bridge import LearnableRouterEncoder
from cluster_router_validation.adapters import compact_torch_logits
from cluster_router_validation.contracts import (
    ClusterCandidate,
    DistributionState,
    EvaluationExample,
    ModelRun,
    ResourceUsage,
)
from learnable_index.model_adapter import (
    build_rolling_local_mask,
    cache_from_layer_kv,
    forward_tokens,
    load_frozen_gemma,
)
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from learnable_index.replay import _full_context_logits
from residual_cache.gpu_local_cluster_memory import (
    GpuLocalClusterMemory,
    GpuLocalClusterMemoryConfig,
)
from residual_cache.gemma4_memory_adapter import Gemma4StaticKVAdapter

from .streaming import EvictedStreamingBlock, RollingContextCollector


def _kv_bytes(layer_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in layer_kv.values()
    )


def _cuda_measure(operation):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        value = operation()
        return value, time.perf_counter() - start, (0, 0, 0, 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return value, latency, (
        peak_allocated,
        peak_reserved,
        max(0, peak_allocated - baseline_allocated),
        max(0, peak_reserved - baseline_reserved),
    )


def build_evidence_only_teacher_forcing_input(
    row: Mapping[str, Any],
    token_ids: Sequence[int],
    target_token_ids: Sequence[int],
) -> tuple[tuple[int, ...], int, int]:
    """Remove the whole mixed memory region and insert only its gold chunk."""

    memory_start, memory_end = (int(value) for value in row["distractor_token_range"])
    target_start, target_end = (
        int(value) for value in row["target_memory_chunk_range"]
    )
    answer_start = int(row["answer_start_position"])
    tokens = tuple(int(value) for value in token_ids)
    targets = tuple(int(value) for value in target_token_ids)
    if not targets:
        raise ValueError("evidence-only teacher forcing requires answer targets")
    if not (
        0 <= memory_start <= target_start < target_end <= memory_end <= answer_start
        <= len(tokens)
    ):
        raise ValueError("ConvoMem memory/evidence/answer ranges are inconsistent")
    prefix = (
        tokens[:memory_start]
        + tokens[target_start:target_end]
        + tokens[memory_end:answer_start]
    )
    return prefix + targets[:-1], len(prefix), target_end - target_start


class Gemma4ClusterRouterModel:
    """Concrete Gemma 4 composition adapter for the generic validation runner."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        model_name: str = "google/gemma-4-E4B-it",
        model_cache_dir: str | None = None,
        local_files_only: bool = True,
        device: str = "cuda",
        dtype: str = "bfloat16",
        router_device: str | None = None,
        block_size: int = 64,
        local_context_length: int = 256,
        residual_layer: int = 40,
        query_summary_length: int = 16,
        prefill_chunk_size: int = 64,
        memory_budget_bytes_per_layer: int | None = 16 * 1024 * 1024,
        memory_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.block_size = int(block_size)
        self.local_context_length = int(local_context_length)
        self.residual_layer = int(residual_layer)
        self.query_summary_length = int(query_summary_length)
        self.prefill_chunk_size = int(prefill_chunk_size)
        if min(
            self.block_size,
            self.local_context_length,
            self.query_summary_length,
            self.prefill_chunk_size,
        ) <= 0:
            raise ValueError("block, context, query, and prefill lengths must be positive")
        if self.prefill_chunk_size != self.block_size:
            raise ValueError(
                "block-transaction streaming requires prefill_chunk_size == block_size"
            )
        memory_values = dict(memory_config or {})
        if memory_budget_bytes_per_layer is not None:
            configured = memory_values.setdefault(
                "memory_budget_bytes", int(memory_budget_bytes_per_layer)
            )
            if int(configured) != int(memory_budget_bytes_per_layer):
                raise ValueError(
                    "memory budget conflicts with memory_config.memory_budget_bytes"
                )
        self.memory_config = GpuLocalClusterMemoryConfig(**memory_values)
        self.bundle = load_frozen_gemma(
            model_name,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            cache_dir=model_cache_dir,
        )
        self.router = LearnableRouterEncoder.from_checkpoint(
            self.checkpoint_path, device=(device if router_device is None else router_device)
        )
        self.router_dim = int(self.router.model.config.projection_dim)
        text_config = self.bundle.text_config
        first_shared = int(text_config.num_hidden_layers) - int(
            getattr(text_config, "num_kv_shared_layers", 0)
        )
        self.memory_layers = tuple(
            index
            for index, layer_type in enumerate(text_config.layer_types[:first_shared])
            if layer_type == "full_attention"
        )
        if not self.memory_layers:
            raise ValueError("Gemma 4 configuration exposes no physical full-attention layers")
        self._descriptor = {
            "adapter": "gemma4_cluster_router",
            "model": self.bundle.fingerprint,
            "checkpoint": str(self.checkpoint_path.resolve()),
            "block_size": self.block_size,
            "local_context_length": self.local_context_length,
            "residual_layer": self.residual_layer,
            "query_summary_length": self.query_summary_length,
            "prefill_chunk_size": self.prefill_chunk_size,
            "memory_layers": list(self.memory_layers),
            "memory_config": asdict(self.memory_config),
            "ingestion_backend": "gpu_local_vmf",
            "ingestion_protocol": (
                "single-pass 256-to-320 block context; block keys are prepared at "
                "completion; the oldest 64-token block is unloaded, posterior-scored "
                "against one pre-commit state, and committed simultaneously; CPU "
                "prefetches bounded cluster IDs only"
            ),
            "answer_evaluation": "teacher_forced_argmax_over_complete_answer",
            "historical_attention_policy": (
                "selected K/V augments physical and KV-shared full-attention layers; "
                "sliding-attention layers remain native"
            ),
        }

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self._descriptor

    def open(self, example: EvaluationExample):
        return _Gemma4ClusterRouterSession(self, example)


class _Gemma4ClusterRouterSession:
    def __init__(self, owner: Gemma4ClusterRouterModel, example: EvaluationExample):
        self.owner = owner
        self.example = example
        if not isinstance(example.payload, Mapping):
            raise TypeError("Gemma 4 adapter requires the original ConvoMem row payload")
        self.row = dict(example.payload)
        self.record = SequenceRecord(
            sequence_id=example.sample_id,
            token_ids=tuple(int(value) for value in self.row["token_ids"]),
            metadata={
                key: value
                for key, value in self.row.items()
                if key != "token_ids"
            },
        )
        plans = build_retrieval_plans(
            self.record,
            PlanConfig(
                local_context_length=owner.local_context_length,
                block_size=owner.block_size,
                future_horizon_length=max(1, len(example.reference_token_ids)),
                retrieval_interval=owner.block_size,
                minimum_candidate_blocks=1,
                retrieval_point_policy="metadata",
            ),
        )
        if len(plans) != 1:
            raise ValueError(f"expected one answer-aligned retrieval plan, found {len(plans)}")
        self.plan = plans[0]
        self.target_token_ids = tuple(
            self.record.token_ids[self.plan.future_start + 1 : self.plan.future_end + 1]
        )
        if not self.target_token_ids or self.target_token_ids != example.reference_token_ids:
            raise ValueError("answer-aligned next-token targets do not match reference_token_ids")
        (
            self.evidence_only_input_ids,
            self.evidence_only_prompt_length,
            self.evidence_only_evidence_length,
        ) = build_evidence_only_teacher_forcing_input(
            self.row, self.record.token_ids, self.target_token_ids
        )
        self.streaming_collector = RollingContextCollector(
            owner.bundle,
            local_context_length=owner.local_context_length,
            block_size=owner.block_size,
            residual_layer=owner.residual_layer,
            query_summary_length=owner.query_summary_length,
        )
        self.memories: dict[int, GpuLocalClusterMemory] = {}
        self._bytes_per_token_by_layer: dict[int, int] = {}
        self._block_router_keys: dict[object, torch.Tensor] = {}

        def prepare_block_key(block, residual_summary: torch.Tensor) -> None:
            if block.block_id in self._block_router_keys:
                raise RuntimeError(f"duplicate completed block {block.block_id!r}")
            self._block_router_keys[block.block_id] = (
                owner.router.encode_block_tensor(residual_summary)
            )

        def ingest_evicted(block: EvictedStreamingBlock) -> None:
            try:
                router_key = self._block_router_keys[block.block.block_id]
            except KeyError as error:
                raise RuntimeError(
                    f"evicted block {block.block.block_id!r} has no prepared router key"
                ) from error
            positions = block.logical_positions
            for layer in owner.memory_layers:
                key, value = block.layer_kv[layer]
                memory = self.memories.get(layer)
                if memory is None:
                    memory = GpuLocalClusterMemory(
                        kv_heads=int(key.shape[1]),
                        head_dim=int(key.shape[3]),
                        router_dim=owner.router_dim,
                        device=key.device,
                        dtype=key.dtype,
                        config=owner.memory_config,
                    )
                    self.memories[layer] = memory
                memory.ingest_block(
                    key,
                    value,
                    router_key=router_key,
                    block_id=block.block.block_id,
                    logical_positions=positions,
                    router_block_size=block.block.length,
                )

        def report_progress(completed: int, total: int) -> None:
            if completed == 1 or completed % 512 == 0 or completed == total:
                print(
                    f"[{example.sample_id}] rolling memory ingestion "
                    f"{completed}/{total} evicted tokens; retained_records="
                    f"{sum(memory.active_record_count for memory in self.memories.values())}; "
                    f"gpu_memory_mib="
                    f"{sum(memory.memory_bytes for memory in self.memories.values()) / (1024 ** 2):.2f}",
                    flush=True,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.rolling = self.streaming_collector.collect(
            self.record,
            self.plan,
            on_block_ready=prepare_block_key,
            on_evict=ingest_evicted,
            progress=report_progress,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ingestion_seconds = time.perf_counter() - start
        self.query_router_key = owner.router.encode_query_tensor(
            self.rolling.query_summary
        )
        self._bytes_per_token_by_layer = {
            layer: int(
                (key.numel() * key.element_size() + value.numel() * value.element_size())
                // key.shape[2]
            )
            for layer, (key, value) in enumerate(self.rolling.local_layer_kv)
        }
        if set(self.memories) != set(owner.memory_layers):
            raise RuntimeError("rolling collection did not initialize every memory layer")
        self._candidates = self._build_candidates()

    def _is_evidence_position(self, position: int) -> bool:
        return any(
            int(start) <= position < int(end)
            for start, end in self.row["evidence_token_ranges"]
        )

    def _build_candidates(self) -> tuple[ClusterCandidate, ...]:
        rows: list[ClusterCandidate] = []
        evidence_blocks = set(self.example.evidence_block_ids)
        for layer, memory in sorted(self.memories.items()):
            for cluster in memory.router_clusters(self.query_router_key):
                positions = list(cluster.logical_positions)
                cluster_evidence_blocks = tuple(
                    sorted(
                        {
                            str(block_id)
                            for block_id in cluster.block_ids
                            if str(block_id) in evidence_blocks
                        }
                    )
                )
                evidence_count = sum(
                    1 for position in positions if self._is_evidence_position(position)
                )
                rows.append(
                    ClusterCandidate(
                        layer=layer,
                        cluster_id=cluster.cluster_id,
                        record_ids=cluster.record_ids,
                        record_token_count=len(cluster.record_ids),
                        latest_position=max(positions),
                        learned_probability=cluster.probability,
                        learned_log_score=cluster.log_score,
                        evidence_record_count=evidence_count,
                        evidence_token_count=evidence_count,
                        evidence_block_ids=cluster_evidence_blocks,
                    )
                )
        return tuple(rows)

    def cluster_candidates(self) -> Sequence[ClusterCandidate]:
        return self._candidates

    def _full_history_tokens(self) -> dict[int, int]:
        count = self.plan.local_context_start
        return {
            layer: count
            for layer in range(self.owner.bundle.physical_cache_layer_count)
        }

    def _local_tokens(self) -> dict[int, int]:
        count = self.plan.local_context_end - self.plan.local_context_start
        return {
            layer: count
            for layer in range(self.owner.bundle.physical_cache_layer_count)
        }

    def _resource_usage(
        self,
        *,
        historical_tokens: Mapping[int, int],
        historical_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        latency: float,
        peaks: tuple[int, int, int, int],
        full_context: bool = False,
    ) -> ResourceUsage:
        local = self._local_tokens()
        full_history = self._full_history_tokens()
        if full_context:
            kv_bytes = sum(
                self._bytes_per_token_by_layer[layer] * self.plan.future_end
                for layer in self._bytes_per_token_by_layer
            )
        else:
            kv_bytes = _kv_bytes(historical_kv) + sum(
                self._bytes_per_token_by_layer[layer] * local[layer]
                for layer in self._bytes_per_token_by_layer
            )
        if full_context:
            end = self.plan.future_end
            window = int(getattr(self.owner.bundle.text_config, "sliding_window", 0))
            full_pairs = end * (end + 1) // 2
            sliding_pairs = (
                full_pairs
                if window <= 0 or end <= window
                else window * (window + 1) // 2 + (end - window) * window
            )
            attention_pairs = sum(
                sliding_pairs if layer_type == "sliding_attention" else full_pairs
                for layer_type in self.owner.bundle.text_config.layer_types
            )
        else:
            query_length = self.plan.future_horizon_length
            native_visible = self.owner.local_context_length
            attention_pairs = int(
                self.owner.bundle.text_config.num_hidden_layers
                * query_length
                * native_visible
            )
            actual_layers = self.owner.bundle.text_config.layer_types
            first_shared = int(self.owner.bundle.text_config.num_hidden_layers) - int(
                getattr(self.owner.bundle.text_config, "num_kv_shared_layers", 0)
            )
            previous = list(actual_layers[:first_shared])
            for actual_layer, layer_type in enumerate(actual_layers):
                if actual_layer < first_shared:
                    source = actual_layer
                else:
                    source = len(previous) - 1 - previous[::-1].index(layer_type)
                attention_pairs += query_length * int(historical_tokens.get(source, 0))
        return ResourceUsage(
            historical_tokens_by_layer=dict(historical_tokens),
            local_tokens_by_layer=local,
            full_history_tokens_by_layer=full_history,
            kv_bytes_visible=kv_bytes,
            cuda_peak_allocated_bytes=peaks[0],
            cuda_peak_reserved_bytes=peaks[1],
            cuda_incremental_peak_allocated_bytes=peaks[2],
            cuda_incremental_peak_reserved_bytes=peaks[3],
            attention_query_key_pairs=attention_pairs,
            latency_seconds=latency,
        )

    def _to_run(
        self,
        logits: torch.Tensor,
        *,
        resources: ResourceUsage,
        state: Mapping[str, Any],
    ) -> ModelRun:
        predicted = tuple(int(value) for value in logits.argmax(dim=-1).tolist())
        return ModelRun(
            predicted_text=self.owner.bundle.tokenizer.decode(
                predicted, skip_special_tokens=True
            ).strip(),
            predicted_token_ids=predicted,
            resources=resources,
            distribution_payload=logits,
            state=dict(state),
        )

    def _restricted_logits(
        self, layer_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        start = self.plan.future_start
        end = self.plan.future_end
        query_positions = tuple(range(start, end))
        cache = cache_from_layer_kv(self.rolling.local_layer_kv)
        mask = build_rolling_local_mask(
            self.owner.bundle,
            past_positions=self.rolling.local_positions,
            query_positions=query_positions,
            local_context_length=self.owner.local_context_length,
        )
        operation = lambda: forward_tokens(
            self.owner.bundle,
            self.record.token_ids[start:end],
            query_positions,
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            logical_cache_position=True,
        )
        if layer_kv:
            with Gemma4StaticKVAdapter(self.owner.bundle.model, dict(layer_kv)):
                output = operation()
        else:
            output = operation()
        return output.logits[0].detach().float().cpu()

    def run_full_context(self) -> ModelRun:
        print(f"[{self.example.sample_id}] condition full_context", flush=True)
        (logits, _pairs), latency, peaks = _cuda_measure(
            lambda: _full_context_logits(
                self.owner.bundle,
                self.record,
                self.plan,
                prefill_chunk_size=self.owner.prefill_chunk_size,
            )
        )
        history = self._full_history_tokens()
        return self._to_run(
            logits,
            resources=self._resource_usage(
                historical_tokens=history,
                historical_kv={},
                latency=latency,
                peaks=peaks,
                full_context=True,
            ),
            state={"baseline": "full_4096_context", "teacher_forced": True},
        )

    def run_evidence_only(self) -> ModelRun:
        print(f"[{self.example.sample_id}] condition evidence_only", flush=True)
        def operation():
            output = forward_tokens(
                self.owner.bundle,
                self.evidence_only_input_ids,
                range(len(self.evidence_only_input_ids)),
                use_cache=False,
            )
            return output.logits[
                0, -self.plan.future_horizon_length :
            ].detach().float().cpu()

        logits, latency, peaks = _cuda_measure(operation)
        physical_layers = range(self.owner.bundle.physical_cache_layer_count)
        history = {
            layer: self.evidence_only_evidence_length for layer in physical_layers
        }
        local_count = self.evidence_only_prompt_length - self.evidence_only_evidence_length
        local = {
            layer: local_count
            for layer in range(self.owner.bundle.physical_cache_layer_count)
        }
        input_length = len(self.evidence_only_input_ids)
        window = int(getattr(self.owner.bundle.text_config, "sliding_window", 0))
        full_pairs = input_length * (input_length + 1) // 2
        sliding_pairs = (
            full_pairs
            if window <= 0 or input_length <= window
            else window * (window + 1) // 2
            + (input_length - window) * window
        )
        attention_pairs = sum(
            sliding_pairs if layer_type == "sliding_attention" else full_pairs
            for layer_type in self.owner.bundle.text_config.layer_types
        )
        resources = ResourceUsage(
            historical_tokens_by_layer=history,
            local_tokens_by_layer=local,
            full_history_tokens_by_layer=self._full_history_tokens(),
            kv_bytes_visible=sum(
                self._bytes_per_token_by_layer[layer] * input_length
                for layer in self._bytes_per_token_by_layer
            ),
            cuda_peak_allocated_bytes=peaks[0],
            cuda_peak_reserved_bytes=peaks[1],
            cuda_incremental_peak_allocated_bytes=peaks[2],
            cuda_incremental_peak_reserved_bytes=peaks[3],
            attention_query_key_pairs=attention_pairs,
            latency_seconds=latency,
        )
        return self._to_run(
            logits,
            resources=resources,
            state={
                "upper_bound": "correct_context_only",
                "cluster_budget_constrained": False,
                "teacher_forced": True,
                "distractor_token_count": 0,
                "compact_prompt_token_count": self.evidence_only_prompt_length,
            },
        )

    def run_local_only(self) -> ModelRun:
        print(f"[{self.example.sample_id}] condition local_only", flush=True)
        logits, latency, peaks = _cuda_measure(lambda: self._restricted_logits({}))
        return self._to_run(
            logits,
            resources=self._resource_usage(
                historical_tokens={},
                historical_kv={},
                latency=latency,
                peaks=peaks,
            ),
            state={"baseline": "local_256_only", "teacher_forced": True},
        )

    def run_with_clusters(
        self,
        selected_cluster_ids: Mapping[int, Sequence[str]],
        *,
        strategy: str,
        budget: int,
    ) -> ModelRun:
        print(
            f"[{self.example.sample_id}] condition {strategy}@{budget}",
            flush=True,
        )
        layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        history: dict[int, int] = {}
        for layer, cluster_ids in selected_cluster_ids.items():
            memory = self.memories[int(layer)]
            key, value, _positions = memory.selected_kv(cluster_ids)
            if key.shape[2] > 0:
                layer_kv[int(layer)] = (key, value)
                history[int(layer)] = int(key.shape[2])
        logits, latency, peaks = _cuda_measure(
            lambda: self._restricted_logits(layer_kv)
        )
        return self._to_run(
            logits,
            resources=self._resource_usage(
                historical_tokens=history,
                historical_kv=layer_kv,
                latency=latency,
                peaks=peaks,
            ),
            state={
                "strategy": strategy,
                "cluster_budget_per_layer": budget,
                "teacher_forced": True,
                "session_ingestion_seconds": self.ingestion_seconds,
                "retained_memory_records": sum(
                    memory.active_record_count for memory in self.memories.values()
                ),
                "retained_memory_bytes": sum(
                    memory.memory_bytes for memory in self.memories.values()
                ),
                "rolling_forward_calls": self.rolling.forward_calls,
                "rolling_forwarded_tokens": self.rolling.forwarded_tokens,
                "rolling_completed_blocks": self.rolling.completed_blocks,
                "rolling_evicted_blocks": self.rolling.evicted_blocks,
                "rolling_evicted_tokens": self.rolling.evicted_tokens,
                "rolling_maximum_forward_context_length": (
                    self.rolling.maximum_forward_context_length
                ),
                "memory_backend": {
                    str(layer): memory.snapshot()
                    for layer, memory in sorted(self.memories.items())
                },
            },
        )

    def compact_distribution(
        self, reference: ModelRun, candidate: ModelRun
    ) -> DistributionState:
        return compact_torch_logits(reference, candidate, self.target_token_ids)

    def close(self) -> None:
        self._candidates = ()
        self._block_router_keys.clear()
        self.memories.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def gemma4_cluster_router_model_factory(**kwargs: Any) -> Gemma4ClusterRouterModel:
    return Gemma4ClusterRouterModel(**kwargs)
