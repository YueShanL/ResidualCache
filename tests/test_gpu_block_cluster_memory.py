import pytest
from types import SimpleNamespace


torch = pytest.importorskip("torch")


def _payload(start: float, tokens: int, *, head_dim: int = 32):
    values = torch.arange(
        start,
        start + tokens * head_dim,
        dtype=torch.float32,
    ).reshape(1, 1, tokens, head_dim)
    return values, values + 1000.0


def _router(axis: int, dimension: int = 32):
    value = torch.zeros(dimension, dtype=torch.float32)
    value[axis] = 1.0
    return value


def _memory(**overrides):
    from residual_cache.gpu_block_cluster_memory import (
        GpuBlockClusterMemory,
        GpuBlockClusterMemoryConfig,
    )

    values = {
        "block_size": 4,
        "slot_capacity": 2,
        "initial_record_capacity": 1,
        "candidate_capacity": 1,
        "locality_bits": 2,
        "locality_probe_radius": 0,
        "tau_new": 1.0,
    }
    values.update(overrides)
    memory = GpuBlockClusterMemory(
        kv_heads=1,
        head_dim=32,
        router_dim=32,
        device="cpu",
        dtype=torch.float32,
        config=GpuBlockClusterMemoryConfig(**values),
    )
    # Keep candidate availability deterministic; posterior scoring remains real.
    memory._codes = lambda directions: [0] * int(directions.shape[0])
    return memory


def test_one_block_is_one_record_and_router_key_is_the_classification_feature(monkeypatch):
    memory = _memory()
    observed = []
    original = memory._posterior

    def capture(directions, candidate_ids, candidate_valid):
        observed.append(directions.detach().clone())
        return original(directions, candidate_ids, candidate_valid)

    monkeypatch.setattr(memory, "_posterior", capture)
    first_key, first_value = _payload(0, 4)
    second_key, second_value = _payload(10_000, 4)
    router = _router(3)
    first_id = memory.ingest_block(
        first_key,
        first_value,
        router_key=router,
        block_id="block-a",
        logical_positions=(0, 1, 2, 3),
    )
    second_id = memory.ingest_block(
        second_key,
        second_value,
        router_key=router,
        block_id="block-b",
        logical_positions=(4, 5, 6, 7),
    )

    assert memory.active_record_count == 2
    assert memory.active_token_count == 8
    assert memory.active_slot_count == 1
    assert len(observed) == 2
    assert torch.allclose(observed[0][0], router)
    assert torch.allclose(observed[1][0], router)
    cluster = memory.router_clusters(router)[0]
    assert cluster.record_ids == (first_id, second_id)
    assert cluster.block_ids == ("block-a", "block-b")
    assert cluster.block_lengths == (4, 4)
    assert cluster.logical_positions == tuple(range(8))

    packed = memory.selected_kv_blocks((cluster.cluster_id,))
    assert packed.keys.shape == packed.values.shape == (1, 1, 8, 32)
    assert torch.equal(packed.keys[:, :, :4], first_key)
    assert torch.equal(packed.keys[:, :, 4:], second_key)
    assert packed.record_ids == (first_id, second_id)
    assert packed.record_slices == ((first_id, 0, 4), (second_id, 4, 8))
    assert packed.token_record_ids == (first_id,) * 4 + (second_id,) * 4


def test_router_query_ranks_independently_created_block_key_clusters(monkeypatch):
    memory = _memory()
    key, value = _payload(0, 4)
    memory.ingest_block(
        key,
        value,
        router_key=_router(0),
        block_id="left",
        logical_positions=(0, 1, 2, 3),
    )
    original = memory._posterior

    def force_new(directions, candidate_ids, candidate_valid):
        selected, _create_new = original(directions, candidate_ids, candidate_valid)
        return selected, torch.ones_like(selected, dtype=torch.bool)

    monkeypatch.setattr(memory, "_posterior", force_new)
    other_key, other_value = _payload(5000, 4)
    memory.ingest_block(
        other_key,
        other_value,
        router_key=_router(1),
        block_id="right",
        logical_positions=(4, 5, 6, 7),
    )

    assert memory.router_clusters(_router(0))[0].block_ids == ("left",)
    assert memory.router_clusters(_router(1))[0].block_ids == ("right",)
    assert memory.snapshot()["classification_feature"] == "learned_router_block_key"
    assert memory.snapshot()["record_unit"] == "layer_local_block"
    assert memory.snapshot()["global_assignment_scans"] == 0


def test_partial_block_is_rejected_and_duplicate_id_is_rejected():
    memory = _memory(block_size=4)
    key, value = _payload(0, 3)
    with pytest.raises(ValueError, match="only complete"):
        memory.ingest_block(
            key,
            value,
            router_key=_router(0),
            block_id="partial",
            logical_positions=(9, 10, 11),
        )

    full_key, full_value = _payload(100, 4)
    memory.ingest_block(
        full_key,
        full_value,
        router_key=_router(0),
        block_id="full",
        logical_positions=(12, 13, 14, 15),
    )
    with pytest.raises(ValueError, match="already active"):
        memory.ingest_block(
            full_key,
            full_value,
            router_key=_router(0),
            block_id="full",
            logical_positions=(16, 17, 18, 19),
        )


def test_usage_eviction_removes_whole_block_records_only():
    memory = _memory(
        eviction_enabled=True,
        eviction_usage_threshold=0.05,
        eviction_min_records_per_cluster=1,
    )
    router = _router(0)
    record_ids = []
    for block_index in range(3):
        key, value = _payload(block_index * 1000, 4)
        record_ids.append(
            memory.ingest_block(
                key,
                value,
                router_key=router,
                block_id=f"block-{block_index}",
                logical_positions=tuple(range(block_index * 4, block_index * 4 + 4)),
            )
        )
    cluster = memory.router_clusters(router)[0]
    with pytest.raises(ValueError, match="cover every active block record"):
        memory.observe_recall_usage(cluster.record_ids[:1], [1.0])

    result = memory.observe_recall_usage(cluster.record_ids, [0.5, 0.01, 0.0])
    assert result["evicted_records"] == 2
    assert memory.active_record_count == 1
    assert memory.active_token_count == 4
    assert memory.evicted_tokens == 8
    remaining = memory.all_kv_blocks()
    assert remaining.record_ids == (record_ids[0],)
    assert remaining.keys.shape[2] == 4


def test_block_memory_has_a_separate_config_and_entrypoint():
    from residual_cache.gpu_block_cluster_memory import (
        GpuBlockClusterMemory,
        GpuBlockClusterMemoryConfig,
    )
    from residual_cache.gpu_local_cluster_memory import (
        GpuLocalClusterMemory,
        GpuLocalClusterMemoryConfig,
    )

    assert GpuBlockClusterMemory is not GpuLocalClusterMemory
    assert GpuBlockClusterMemoryConfig is not GpuLocalClusterMemoryConfig
    assert "index_mode" not in GpuBlockClusterMemoryConfig.__dataclass_fields__
    assert "router_count_exponent" not in GpuBlockClusterMemoryConfig.__dataclass_fields__


def test_block_aligned_collector_retains_partial_oldest_block(monkeypatch):
    from cluster_router_experiment.block_streaming import (
        BlockAlignedRollingContextCollector,
    )
    from learnable_index.contracts import BlockRange
    from learnable_index.model_adapter import cache_from_layer_kv, layer_kv_from_cache
    from learnable_index.planning import RetrievalPlan, SequenceRecord

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
        assert tuple(token_ids) == positions
        assert use_cache and output_hidden_states and logical_cache_position
        current = torch.tensor(positions, dtype=torch.float32).view(1, 1, -1, 1)
        current = current.repeat(1, 1, 1, 32)
        if past_key_values is None:
            pairs = ((current, current + 100),)
        else:
            pairs = tuple(
                (
                    torch.cat((key, current), dim=2),
                    torch.cat((value, current + 100), dim=2),
                )
                for key, value in layer_kv_from_cache(past_key_values)
            )
        mask = attention_mask["full_attention"]
        visible_context_lengths.append(
            int((mask[0, 0] == 0).sum(dim=-1).max().item())
        )
        hidden = torch.tensor(positions, dtype=torch.float32).view(1, -1, 1)
        hidden = hidden.repeat(1, 1, 32)
        return SimpleNamespace(
            past_key_values=cache_from_layer_kv(pairs),
            hidden_states=(torch.zeros_like(hidden), hidden),
        )

    monkeypatch.setattr(
        "cluster_router_experiment.block_streaming.forward_tokens", fake_forward
    )
    bundle = SimpleNamespace(
        input_device=torch.device("cpu"),
        text_model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.zeros(1))
        ),
        text_config=SimpleNamespace(layer_types=("full_attention",)),
    )
    record = SequenceRecord("s", tuple(range(11)), {})
    candidates = tuple(
        BlockRange(f"s:block:{start:09d}-{start + 2:09d}", start, start + 2)
        for start in range(0, 6, 2)
    )
    plan = RetrievalPlan(
        sample_id="sample",
        sequence_id="s",
        retrieval_position=10,
        first_future_position_affected_by_retrieval=11,
        future_horizon_length=1,
        local_context_start=7,
        local_context_end=11,
        candidate_blocks=candidates,
    )
    evicted = []
    ready = []
    result = BlockAlignedRollingContextCollector(
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

    assert max(visible_context_lengths) == 6
    assert result.maximum_forward_context_length == 6
    assert result.local_positions == (6, 7, 8, 9, 10)
    assert result.evicted_blocks == 3
    assert result.evicted_tokens == 6
    assert all(len(block.logical_positions) == 2 for block in evicted)
    assert [block.block.start_position for block in evicted] == [0, 2, 4]
    assert [
        block.layer_kv[0][0][0, 0, :, 0].tolist() for block in evicted
    ] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert [block.start_position for block, _summary in ready] == [0, 2, 4, 6, 8]
