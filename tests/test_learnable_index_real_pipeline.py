from __future__ import annotations

from types import SimpleNamespace

import torch

from learnable_index.aligned_builder import AlignedCollectionConfig, collect_aligned_dataset
from learnable_index.collectors import StudentCollectionConfig
from learnable_index.config import (
    AttentionAggregationConfig,
    LossConfig,
    RouterConfig,
    TrainConfig,
)
from learnable_index.kv_store import KVBlock, KVBlockStore, merge_layer_kv
from learnable_index.model_adapter import (
    ModelBundle,
    build_sparse_prefix_mask,
    cache_from_layer_kv,
    model_fingerprint,
    trim_prefix_and_local_kv,
)
from learnable_index.planning import PlanConfig, SequenceRecord, build_retrieval_plans
from learnable_index.replay import ReplayConfig, evaluate_retrieval_replay
from learnable_index.retrieval import RetrievalPolicyConfig, decide_retrieval
from learnable_index.synthetic import make_synthetic_samples
from learnable_index.trainer import fit_router


class FakeGemmaConfig:
    model_type = "gemma4_text"
    hidden_size = 4
    num_hidden_layers = 2
    num_kv_shared_layers = 0
    layer_types = ["full_attention", "sliding_attention"]
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    sliding_window = 8
    _attn_implementation = "eager"

    def get_text_config(self, decoder=True):
        return self


class FakeTextBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(64, 4)


class FakeGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeGemmaConfig()
        self.model = FakeTextBody()

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
        output_attentions=False,
        **kwargs,
    ):
        batch, query_length = input_ids.shape
        positions = position_ids.float()
        base_hidden = positions[..., None].repeat(1, 1, 4) / 10.0
        hidden_states = (
            base_hidden,
            base_hidden + 0.1,
            base_hidden + 0.2,
        ) if output_hidden_states else None

        if use_cache:
            assert past_key_values is not None
            for layer_index in range(self.config.num_hidden_layers):
                key = positions[:, None, :, None].repeat(1, 1, 1, 2) + layer_index
                value = key + 100
                past_key_values.update(key, value, layer_index)

        attentions = None
        if output_attentions:
            causal = torch.tril(torch.ones(query_length, query_length, device=input_ids.device))
            causal = causal / causal.sum(dim=-1, keepdim=True)
            attention = causal[None, None].repeat(batch, self.config.num_attention_heads, 1, 1)
            attentions = tuple(attention.clone() for _ in range(self.config.num_hidden_layers))

        vocabulary = 64
        logits = torch.full(
            (batch, query_length, vocabulary),
            -4.0,
            device=input_ids.device,
        )
        predicted = (input_ids + 1) % vocabulary
        logits.scatter_(2, predicted[..., None], 4.0)
        return SimpleNamespace(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            past_key_values=past_key_values,
        )


def fake_bundle() -> ModelBundle:
    model = FakeGemma().eval()
    return ModelBundle(
        model=model,
        tokenizer=None,
        model_name="fake-gemma4",
        input_device=torch.device("cpu"),
        fingerprint=model_fingerprint(model, "fake-gemma4"),
    )


def test_retrieval_planning_reserves_first_affected_forward_and_target():
    record = SequenceRecord("s", tuple(range(20)), {})
    plans = build_retrieval_plans(
        record,
        PlanConfig(
            local_context_length=4,
            block_size=2,
            future_horizon_length=3,
            retrieval_interval=2,
        ),
    )

    first = plans[0]
    assert first.retrieval_position == 5
    assert first.local_context_start == 2
    assert first.first_future_position_affected_by_retrieval == 6
    assert first.future_end == 9
    assert first.candidate_blocks[0].end_position <= first.local_context_start
    assert plans[-1].future_end < len(record.token_ids)


def test_kv_store_round_trip_merge_and_trim(tmp_path):
    sample = make_synthetic_samples(sample_count=1, residual_dim=4, min_blocks=2, max_blocks=2)[0]
    store = KVBlockStore(tmp_path / "kv")
    for index, block_range in enumerate(sample.candidate_blocks):
        key = torch.full((1, 1, block_range.length, 2), float(index))
        store.save(
            KVBlock(
                block=block_range,
                sequence_id=sample.sequence_id,
                token_ids=tuple(range(block_range.length)),
                logical_positions=tuple(range(block_range.start_position, block_range.end_position)),
                layer_kv=((key, key + 10), (key + 20, key + 30)),
                residual_summary=torch.ones(4) * index,
                model_fingerprint={"model": "fake"},
                metadata={},
            )
        )
    loaded = store.load_many([block.block_id for block in reversed(sample.candidate_blocks)])
    merged = merge_layer_kv(loaded, device="cpu")
    assert merged[0][0].shape[2] == sum(block.length for block in sample.candidate_blocks)
    cache = cache_from_layer_kv(merged)
    assert cache.get_seq_length() == merged[0][0].shape[2]
    with_local = tuple(
        (torch.cat((key, key[:, :, :3]), dim=2), torch.cat((value, value[:, :, :3]), dim=2))
        for key, value in merged
    )
    trimmed = trim_prefix_and_local_kv(
        with_local,
        prefix_length=merged[0][0].shape[2],
        maximum_local_tokens=2,
    )
    assert trimmed[0][0].shape[2] == merged[0][0].shape[2] + 2


def test_sparse_prefix_mask_and_manual_score_threshold_policy():
    bundle = fake_bundle()
    mask = build_sparse_prefix_mask(
        bundle,
        query_length=2,
        prefix_length=3,
        local_past_length=2,
    )["full_attention"][0, 0]
    assert mask.shape == (2, 7)
    assert torch.all(mask[:, :3] == 0)
    assert mask[0, -1] < -1e20
    assert mask[1, -1] == 0

    sample = make_synthetic_samples(sample_count=1, min_blocks=3, max_blocks=3)[0]
    decision = decide_retrieval(
        sample,
        torch.tensor([5.0, 1.0, 0.0]),
        RetrievalPolicyConfig(
            policy="score_threshold",
            top_n=3,
            score_threshold=0.8,
        ),
    )
    assert decision.selected_indices == (0,)

    rejected = decide_retrieval(
        sample,
        torch.tensor([1.0, 1.0, 1.0]),
        RetrievalPolicyConfig(
            policy="score_threshold",
            top_n=3,
            score_threshold=0.5,
        ),
    )
    assert rejected.selected_indices == ()
    assert rejected.requested_top_n == 0


def test_fake_gemma_collect_train_and_replay_end_to_end(tmp_path):
    bundle = fake_bundle()
    record = SequenceRecord("sequence-1", tuple(range(1, 15)), {"kind": "fake"})
    collection_dir = tmp_path / "collection"
    progress = []
    dataset, manifest = collect_aligned_dataset(
        bundle,
        [record],
        collection_dir,
        AlignedCollectionConfig(
            plan=PlanConfig(
                local_context_length=4,
                block_size=2,
                future_horizon_length=2,
                retrieval_interval=2,
            ),
            student=StudentCollectionConfig(
                local_context_length=4,
                residual_layer=-1,
                query_summary="mean",
                query_summary_length=2,
            ),
            attention=AttentionAggregationConfig(future_reduction="mean"),
        ),
        progress_callback=progress.append,
    )
    assert len(dataset) >= 2
    assert manifest["kv_block_count"] >= 1
    assert progress[-1]["completed"] == progress[-1]["total"] == len(dataset)
    assert dataset[0].per_future_teacher_block_mass.shape == (
        dataset[0].future_horizon_length,
        len(dataset[0].candidate_blocks),
    )
    assert dataset[0].teacher_layer_head_future_block_mass.shape[-2:] == (
        dataset[0].future_horizon_length,
        len(dataset[0].candidate_blocks),
    )

    training_dir = tmp_path / "training"
    fit_router(
        dataset,
        training_dir,
        RouterConfig(residual_dim=dataset.residual_dim, projection_dim=4, hidden_dim=8, depth=1),
        LossConfig(),
        TrainConfig(
            epochs=1,
            batch_size=2,
            validation_fraction=0,
            device="cpu",
            top_n=1,
        ),
    )
    summary = evaluate_retrieval_replay(
        bundle,
        collection_dir,
        training_dir / "best.pt",
        tmp_path / "replay",
        ReplayConfig(
            policy=RetrievalPolicyConfig(policy="fixed", top_n=1),
            router_device="cpu",
        ),
    )

    assert summary["sample_count"] == len(dataset)
    assert set(summary["conditions"]) == {
        "full_context",
        "local_256",
        "oracle_top_n",
        "predicted",
        "recent_top_n",
    }
    assert (tmp_path / "replay" / "samples.jsonl").is_file()
    assert (tmp_path / "replay" / "summary.json").is_file()
