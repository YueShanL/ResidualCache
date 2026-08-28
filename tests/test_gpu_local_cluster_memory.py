from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _block(vectors):
    tensor = torch.tensor(vectors, dtype=torch.float32).view(1, 1, len(vectors), -1)
    return tensor, tensor.roll(1, dims=-1)


def test_gpu_local_ingestion_grows_and_router_is_metadata_only():
    from residual_cache.gpu_local_cluster_memory import (
        GpuLocalClusterMemory,
        GpuLocalClusterMemoryConfig,
    )

    memory = GpuLocalClusterMemory(
        kv_heads=1,
        head_dim=32,
        router_dim=32,
        device="cpu",
        dtype=torch.float32,
        config=GpuLocalClusterMemoryConfig(
            memory_budget_bytes=256_000,
            slot_capacity=8,
            initial_record_capacity=2,
            candidate_capacity=2,
            locality_bits=4,
            write_chunk_size=1,
            tau_new=1.0,
        ),
    )
    vectors = [[1.0] + [0.0] * 31 for _ in range(4)]
    keys, values = _block(vectors)
    router_a = torch.tensor([1.0] + [0.0] * 31)
    router_b = torch.tensor([0.0, 1.0] + [0.0] * 30)
    memory.ingest_block(
        keys[:, :, :2],
        values[:, :, :2],
        router_key=router_a,
        block_id="a",
        logical_positions=(0, 1),
        router_block_size=2,
    )
    slots_before_router_change = memory.active_slot_count
    memory.ingest_block(
        keys[:, :, 2:],
        values[:, :, 2:],
        router_key=router_b,
        block_id="b",
        logical_positions=(2, 3),
        router_block_size=2,
    )

    # The first pre-commit chunk may create more than one slot, but changing an
    # orthogonal learned block key cannot create any additional native K/V slot.
    assert memory.active_slot_count == slots_before_router_change
    assert memory.active_record_count == 4
    clusters = memory.router_clusters(router_a)
    assert len(clusters) == slots_before_router_change
    assert sorted(block for cluster in clusters for block in cluster.block_ids) == [
        "a",
        "a",
        "b",
        "b",
    ]
    assert sum(cluster.total_weight for cluster in clusters) == pytest.approx(2.0)
    selected_key, selected_value, positions = memory.selected_kv(
        tuple(cluster.cluster_id for cluster in clusters)
    )
    assert selected_key.shape == selected_value.shape == (1, 1, 4, 32)
    assert positions == (0, 1, 2, 3)

    snapshot = memory.snapshot()
    assert snapshot["global_assignment_scans"] == 0
    assert snapshot["maximum_candidate_slots_considered"] <= 2
    assert snapshot["cpu_prefetch_payload"] == "bounded_cluster_ids_only"
    assert snapshot["record_capacity_policy"] == "dynamic_unbounded"
    assert snapshot["usage_eviction_enabled"] is True
    assert snapshot["memory_budget_role"] == "legacy_eviction_enable_switch_only"
    assert snapshot["allocated_record_capacity"] >= 4


def test_gpu_local_records_grow_without_global_eviction():
    from residual_cache.gpu_local_cluster_memory import (
        GpuLocalClusterMemory,
        GpuLocalClusterMemoryConfig,
    )

    memory = GpuLocalClusterMemory(
        kv_heads=1,
        head_dim=32,
        router_dim=32,
        device="cpu",
        dtype=torch.float32,
        config=GpuLocalClusterMemoryConfig(
            memory_budget_bytes=4_500,
            slot_capacity=4,
            initial_record_capacity=2,
            candidate_capacity=2,
            locality_bits=4,
            write_chunk_size=1,
            tau_new=1.0,
        ),
    )
    initial_capacity = memory.record_capacity
    count = initial_capacity + 4
    vectors = [[1.0] + [0.0] * 31 for _ in range(count)]
    keys, values = _block(vectors)
    router = torch.tensor([1.0] + [0.0] * 31)
    for start in range(0, count, 2):
        end = min(count, start + 2)
        memory.ingest_block(
            keys[:, :, start:end],
            values[:, :, start:end],
            router_key=router,
            block_id=f"block-{start // 2}",
            logical_positions=tuple(range(start, end)),
            router_block_size=end - start,
        )

    assert memory.active_record_count == count
    assert memory.record_capacity >= count
    assert memory.evicted_records == 0
    assert memory.memory_bytes > 4_500
    assert memory.snapshot()["global_assignment_scans"] == 0


def test_usage_eviction_is_confined_to_complete_recalled_clusters(monkeypatch):
    from residual_cache.gpu_local_cluster_memory import (
        GpuLocalClusterMemory,
        GpuLocalClusterMemoryConfig,
    )

    memory = GpuLocalClusterMemory(
        kv_heads=1,
        head_dim=32,
        router_dim=32,
        device="cpu",
        dtype=torch.float32,
        config=GpuLocalClusterMemoryConfig(
            eviction_enabled=True,
            eviction_usage_threshold=0.05,
            eviction_min_records_per_cluster=1,
            slot_capacity=1,
            initial_record_capacity=2,
            candidate_capacity=1,
            locality_bits=2,
            write_chunk_size=1,
            tau_new=1.0,
        ),
    )
    router = torch.tensor([1.0] + [0.0] * 31)
    first, first_values = _block([[1.0] + [0.0] * 31 for _ in range(4)])
    for index in range(4):
        memory.ingest_block(
            first[:, :, index : index + 1],
            first_values[:, :, index : index + 1],
            router_key=router,
            block_id="recalled",
            logical_positions=(index,),
        )
    original_posterior = memory._posterior

    def force_new(directions, candidate_ids, candidate_valid):
        selected, _create_new = original_posterior(
            directions, candidate_ids, candidate_valid
        )
        return selected, torch.ones_like(selected, dtype=torch.bool)

    monkeypatch.setattr(memory, "_posterior", force_new)
    second, second_values = _block([[0.0, 1.0] + [0.0] * 30])
    memory.ingest_block(
        second,
        second_values,
        router_key=router,
        block_id="untouched",
        logical_positions=(4,),
    )
    clusters = memory.router_clusters(router)
    recalled = next(cluster for cluster in clusters if len(cluster.record_ids) == 4)
    untouched = next(cluster for cluster in clusters if len(cluster.record_ids) == 1)

    with pytest.raises(ValueError, match="cover every active record"):
        memory.observe_recall_usage(recalled.record_ids[:1], [1.0])

    result = memory.observe_recall_usage(
        recalled.record_ids,
        [0.5, 0.01, 0.001, 0.0],
    )
    assert result["evicted_records"] == 3
    assert memory.active_record_count == 2
    remaining = memory.router_clusters(router)
    assert any(cluster.record_ids == untouched.record_ids for cluster in remaining)
    assert memory.snapshot()["eviction_scope"] == "recalled_cluster_only"


def test_gpu_block_transaction_scores_one_batch_against_precommit_state(monkeypatch):
    from residual_cache.gpu_local_cluster_memory import (
        GpuLocalClusterMemory,
        GpuLocalClusterMemoryConfig,
    )

    memory = GpuLocalClusterMemory(
        kv_heads=1,
        head_dim=32,
        router_dim=32,
        device="cpu",
        dtype=torch.float32,
        config=GpuLocalClusterMemoryConfig(
            memory_budget_bytes=64_000,
            slot_capacity=4,
            candidate_capacity=1,
            locality_bits=2,
            write_chunk_size=4,
            tau_new=1.0,
        ),
    )
    keys, values = _block(
        [
            [1.0] + [0.0] * 31,
            [0.0, 1.0] + [0.0] * 30,
            [1.0] + [0.0] * 31,
            [0.0, 0.0, 1.0] + [0.0] * 29,
        ]
    )
    router = torch.tensor([1.0] + [0.0] * 31)
    observed_record_counts = []

    def forced_posterior(directions, candidate_ids, candidate_valid):
        del candidate_ids, candidate_valid
        observed_record_counts.append(memory.active_record_count)
        return (
            torch.zeros(
                (directions.shape[0],),
                device=directions.device,
                dtype=torch.int64,
            ),
            torch.ones(
                (directions.shape[0],), device=directions.device, dtype=torch.bool
            ),
        )

    monkeypatch.setattr(memory, "_posterior", forced_posterior)
    memory.ingest_block(
        keys,
        values,
        router_key=router,
        block_id="block",
        logical_positions=(0, 1, 2, 3),
        router_block_size=4,
    )

    assert observed_record_counts == [0]
    assert memory.active_record_count == 4
    assert memory.active_slot_count == 4
    assert memory.snapshot()["global_assignment_scans"] == 0


def test_rolling_context_emits_only_blocks_that_leave_the_window(monkeypatch):
    from learnable_index.contracts import BlockRange
    from learnable_index.model_adapter import cache_from_layer_kv, layer_kv_from_cache
    from learnable_index.planning import RetrievalPlan, SequenceRecord
    from cluster_router_experiment.streaming import RollingContextCollector

    calls = []
    visible_context_lengths = []

    def fake_forward(
        _bundle,
        token_ids,
        logical_positions,
        *,
        past_key_values,
        attention_mask,
        use_cache,
        output_hidden_states,
        logical_cache_position,
        **_kwargs,
    ):
        positions = tuple(logical_positions)
        calls.append(positions)
        assert use_cache and output_hidden_states and logical_cache_position
        current = torch.tensor(positions, dtype=torch.float32).view(1, 1, -1, 1)
        current = current.repeat(1, 1, 1, 32)
        if past_key_values is None:
            pairs = ((current, current + 100), (current + 10, current + 110))
        else:
            pairs = tuple(
                (
                    torch.cat((key, current + layer * 10), dim=2),
                    torch.cat((value, current + 100 + layer * 10), dim=2),
                )
                for layer, (key, value) in enumerate(
                    layer_kv_from_cache(past_key_values)
                )
            )
        mask = attention_mask["full_attention"]
        assert mask.shape[-2] == len(positions)
        visible = int((mask[0, 0] == 0).sum(dim=-1).max().item())
        visible_context_lengths.append(visible)
        # Four retained tokens plus one incoming two-token block.
        assert visible <= 6
        hidden = torch.tensor(positions, dtype=torch.float32).view(1, -1, 1).repeat(
            1, 1, 4
        )
        return SimpleNamespace(
            past_key_values=cache_from_layer_kv(pairs),
            hidden_states=(torch.zeros_like(hidden), hidden),
        )

    monkeypatch.setattr(
        "cluster_router_experiment.streaming.forward_tokens", fake_forward
    )
    embed = SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))
    bundle = SimpleNamespace(
        input_device=torch.device("cpu"),
        text_model=SimpleNamespace(embed_tokens=embed),
        text_config=SimpleNamespace(layer_types=("full_attention",)),
    )
    record = SequenceRecord("s", tuple(range(14)), {})
    candidates = tuple(
        BlockRange(f"s:block:{start:09d}-{start + 2:09d}", start, start + 2)
        for start in range(0, 8, 2)
    )
    plan = RetrievalPlan(
        sample_id="sample",
        sequence_id="s",
        retrieval_position=11,
        first_future_position_affected_by_retrieval=12,
        future_horizon_length=1,
        local_context_start=8,
        local_context_end=12,
        candidate_blocks=candidates,
    )
    evicted = []
    ready = []
    result = RollingContextCollector(
        bundle,
        local_context_length=4,
        block_size=2,
        residual_layer=0,
        query_summary_length=2,
    ).collect(
        record,
        plan,
        on_block_ready=lambda block, summary: ready.append((block, summary)),
        on_evict=evicted.append,
    )

    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    assert result.forward_calls == 6
    assert result.forwarded_tokens == 12
    assert result.evicted_blocks == 4
    assert result.evicted_tokens == 8
    assert result.completed_blocks == 6
    assert result.maximum_forward_context_length == 6
    assert max(visible_context_lengths) == 6
    assert result.local_positions == (8, 9, 10, 11)
    assert result.query_summary.tolist() == pytest.approx([10.5] * 4)
    assert [block.block.start_position for block in evicted] == [0, 2, 4, 6]
    assert [block.start_position for block, _summary in ready] == [0, 2, 4, 6, 8, 10]
    assert [
        block.layer_kv[0][0][0, 0, :, 0].tolist() for block in evicted
    ] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]

    # A retrieval point inside a mechanical block keeps an exact four-token
    # local suffix and never writes the incomplete tail into historical memory.
    calls.clear()
    visible_context_lengths.clear()
    evicted.clear()
    ready.clear()
    unaligned_plan = RetrievalPlan(
        sample_id="sample-unaligned",
        sequence_id="s",
        retrieval_position=10,
        first_future_position_affected_by_retrieval=11,
        future_horizon_length=1,
        local_context_start=7,
        local_context_end=11,
        candidate_blocks=candidates[:3],
    )
    unaligned = RollingContextCollector(
        bundle,
        local_context_length=4,
        block_size=2,
        residual_layer=0,
        query_summary_length=2,
    ).collect(
        record,
        unaligned_plan,
        on_block_ready=lambda block, summary: ready.append((block, summary)),
        on_evict=evicted.append,
    )

    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10,)]
    assert unaligned.forward_calls == 6
    assert unaligned.forwarded_tokens == 11
    assert unaligned.evicted_blocks == 4
    assert unaligned.evicted_tokens == 7
    assert unaligned.completed_blocks == 5
    assert unaligned.maximum_forward_context_length == 6
    assert max(visible_context_lengths) == 6
    assert unaligned.local_positions == (7, 8, 9, 10)
    assert unaligned.query_summary.tolist() == pytest.approx([9.5] * 4)
    assert [block.block.start_position for block in evicted] == [0, 2, 4, 6]
    assert [block.logical_positions for block in evicted] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6,),
    ]
    assert [block.start_position for block, _summary in ready] == [0, 2, 4, 6, 8]
