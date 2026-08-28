from types import SimpleNamespace

import pytest

from residual_cache.gemma4_memory_adapter import Gemma4StaticKVController
from residual_cache.probabilistic_hierarchical_memory import (
    HierarchicalVMFMemory,
    MemoryConfig,
)
from cluster_router_experiment.gemma4 import build_evidence_only_teacher_forcing_input


torch = pytest.importorskip("torch")


def _module(layer_idx, *, shared=False, layer_type="full_attention"):
    config = SimpleNamespace(
        num_hidden_layers=6,
        num_kv_shared_layers=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    return SimpleNamespace(
        layer_idx=layer_idx,
        layer_type=layer_type,
        is_kv_shared_layer=shared,
        config=config,
        num_key_value_groups=2,
        head_dim=4,
        training=False,
    )


def test_static_controller_injects_variable_prefix_and_maps_shared_layers():
    historical_key = torch.randn(1, 1, 3, 4)
    historical_value = torch.randn(1, 1, 3, 4)
    controller = Gemma4StaticKVController(
        {3: (historical_key, historical_value)},
        collect_historical_usage=True,
    )
    physical = _module(3)
    shared = _module(5, shared=True)

    assert controller.physical_source_layer(physical) == 3
    assert controller.physical_source_layer(shared) == 3

    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 1, 2, 4)
    value = torch.randn(1, 1, 2, 4)
    output, weights = controller.attend(
        shared,
        query,
        key,
        value,
        None,
        dropout=0.0,
        scaling=0.5,
        softcap=None,
    )

    assert output.shape == (1, 2, 2, 4)
    assert weights.shape == (1, 2, 2, 5)
    assert controller.retrieved_tokens_by_layer == {3: 3}
    usage = controller.historical_usage_rates()[3]
    assert usage.shape == (3,)
    assert torch.allclose(usage, weights[..., :3].float().mean(dim=(0, 1, 2)))


def test_evidence_only_prompt_removes_both_distractor_sides():
    # instruction=[0,1], mixed memory=[10..19], question suffix=[20,21], answer=[30,31]
    tokens = (0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 30, 31)
    inputs, prompt_length, evidence_length = build_evidence_only_teacher_forcing_input(
        {
            "distractor_token_range": [2, 12],
            "target_memory_chunk_range": [5, 8],
            "answer_start_position": 14,
        },
        tokens,
        (30, 31),
    )

    assert inputs == (0, 1, 13, 14, 15, 20, 21, 30)
    assert prompt_length == 7
    assert evidence_length == 3


def test_legacy_budget_does_not_impose_a_global_hard_cap():
    budget = 2_000
    memory = HierarchicalVMFMemory(
        MemoryConfig(
            memory_budget_bytes=budget,
            budget_step_size=0.0,
            enable_split_merge=False,
        )
    )
    for index in range(20):
        memory.write(
            (1.0, 0.01 * index),
            (1.0, 0.01 * index),
            (0.0, 1.0),
            router_key=(1.0, 0.1),
            router_block_id=f"block-{index // 4}",
            router_block_size=4,
        )
        memory.maintain()

    assert memory.record_count == 20
    assert memory.memory_bytes > budget
    assert memory.snapshot()["memory_cost_lambda"] == 0.0
