from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cluster_router_validation import (  # noqa: E402
    ClusterCandidate,
    DistributionState,
    EvaluationExample,
    MetricConfig,
    ModelRun,
    ResourceUsage,
    ValidationRunConfig,
    collect_validation_states,
    compact_torch_logits,
    evaluate_validation_states,
    select_clusters,
)
from cluster_router_validation.adapters import JsonlEvaluationDataset  # noqa: E402


class FakeDataset:
    descriptor = {"dataset_id": "fake-memory-qa", "split": "test", "sha256": "fake"}

    def __iter__(self):
        yield EvaluationExample(
            sample_id="sample-1",
            group_id="profile-1",
            reference_answer="The blue bicycle",
            reference_token_ids=(1, 2),
            sequence_length=4_096,
            evidence_distance_tokens=3_000,
            evidence_token_count=8,
            evidence_block_ids=("evidence-block",),
            metadata={"evidence_placement_bin": 2},
        )


class FakeSession:
    def __init__(self, example):
        self.example = example
        self.closed = False

    def cluster_candidates(self):
        return (
            ClusterCandidate(
                layer=0,
                cluster_id="evidence-cluster",
                record_ids=("evidence-record",),
                record_token_count=8,
                latest_position=100,
                learned_probability=0.9,
                learned_log_score=2.0,
                teacher_attention_mass=0.8,
                evidence_record_count=1,
                evidence_token_count=8,
                evidence_block_ids=("evidence-block",),
            ),
            ClusterCandidate(
                layer=0,
                cluster_id="recent-distractor",
                record_ids=("distractor-record",),
                record_token_count=12,
                latest_position=3_000,
                learned_probability=0.1,
                learned_log_score=-2.0,
                teacher_attention_mass=0.2,
            ),
        )

    @staticmethod
    def _resources(historical=0):
        return ResourceUsage(
            historical_tokens_by_layer=({} if historical < 0 else {0: historical}),
            local_tokens_by_layer={0: 256},
            full_history_tokens_by_layer={0: 4_000},
            kv_bytes_visible=max(0, historical) * 16,
            cuda_peak_allocated_bytes=1_000 + max(0, historical),
        )

    def run_full_context(self):
        return ModelRun(
            self.example.reference_answer,
            (1, 2),
            self._resources(4_000),
            state={"quality": "correct"},
        )

    def run_evidence_only(self):
        return ModelRun(
            self.example.reference_answer,
            (1, 2),
            self._resources(32),
            state={"quality": "correct", "upper_bound": True},
        )

    def run_local_only(self):
        return ModelRun("unknown", (9, 9), self._resources(0), state={"quality": "wrong"})

    def run_with_clusters(self, selected_cluster_ids, *, strategy, budget):
        selected = {
            cluster_id
            for cluster_ids in selected_cluster_ids.values()
            for cluster_id in cluster_ids
        }
        correct = "evidence-cluster" in selected
        # Leave historical_tokens_by_layer empty to exercise runner derivation
        # from the selected cluster's actual token counts.
        return ModelRun(
            self.example.reference_answer if correct else "unknown",
            (1, 2) if correct else (9, 9),
            self._resources(-1),
            state={"quality": "correct" if correct else "wrong"},
        )

    def compact_distribution(self, reference, candidate):
        correct = candidate.state["quality"] == "correct"
        return DistributionState(
            token_count=2,
            target_nll_sum=0.2 if correct else 4.0,
            reference_entropy_sum=1.0,
            reference_cross_entropy_sum=1.0 if correct else 3.0,
            argmax_agreement_count=2 if correct else 0,
            target_accuracy_count=2 if correct else 0,
        )

    def close(self):
        self.closed = True


class FakeModel:
    descriptor = {"model_id": "fake-model", "fingerprint": "fake-model-v1"}

    def open(self, example):
        return FakeSession(example)


def test_selection_policies_are_layer_local_and_deterministic():
    candidates = FakeSession(next(iter(FakeDataset()))).cluster_candidates()

    fixed = select_clusters(candidates, strategy="fixed_policy", budget=1)
    learned = select_clusters(candidates, strategy="learned_router", budget=1)
    oracle = select_clusters(candidates, strategy="oracle_cluster", budget=1)

    assert fixed == {0: ("recent-distractor",)}
    assert learned == {0: ("evidence-cluster",)}
    assert oracle == {0: ("evidence-cluster",)}


def test_collect_and_offline_metrics_are_separate_and_replayable(tmp_path):
    state_dir = tmp_path / "state"
    metric_dir = tmp_path / "metrics"
    manifest = collect_validation_states(
        FakeDataset(),
        FakeModel(),
        state_dir,
        ValidationRunConfig(budgets=(1,)),
    )

    assert manifest["status"] == "complete"
    state = json.loads((state_dir / "samples.jsonl").read_text(encoding="utf-8"))
    assert set(state["conditions"]) == {
        "full_context",
        "evidence_only",
        "local_only",
        "fixed_policy@1",
        "learned_router@1",
        "oracle_cluster@1",
    }
    assert state["conditions"]["learned_router@1"]["distribution"]["token_count"] == 2
    assert "distribution_payload" not in (state_dir / "samples.jsonl").read_text(
        encoding="utf-8"
    )

    result = evaluate_validation_states(state_dir, metric_dir, MetricConfig())
    conditions = result["summary"]["conditions"]

    assert conditions["full_context"]["answer_exact_match"] == pytest.approx(1.0)
    assert conditions["evidence_only"]["answer_exact_match"] == pytest.approx(1.0)
    assert conditions["local_only"]["answer_exact_match"] == pytest.approx(0.0)
    assert conditions["fixed_policy@1"]["answer_exact_match"] == pytest.approx(0.0)
    assert conditions["learned_router@1"]["answer_exact_match"] == pytest.approx(1.0)
    assert conditions["oracle_cluster@1"]["answer_exact_match"] == pytest.approx(1.0)
    assert conditions["learned_router@1"]["evidence_record_recall"] == pytest.approx(1.0)
    assert conditions["learned_router@1"]["teacher_attention_mass_coverage"] == pytest.approx(0.8)
    assert conditions["learned_router@1"]["historical_layer_token_kv_ratio"] == pytest.approx(
        8 / 4_000
    )
    assert conditions["learned_router@1"][
        "normalized_quality_recovery/answer_token_f1"
    ] == pytest.approx(1.0)
    assert conditions["learned_router@1"][
        "quality_gap_to_evidence_only/answer_token_f1"
    ] == pytest.approx(0.0)
    assert (metric_dir / "sample_metrics.jsonl").is_file()
    assert (metric_dir / "condition_summary.csv").is_file()

    # Resume is identity-checked and does not duplicate completed samples.
    resumed = collect_validation_states(
        FakeDataset(),
        FakeModel(),
        state_dir,
        ValidationRunConfig(budgets=(1,), resume=True),
    )
    assert resumed["completed_sample_count"] == 1
    assert len((state_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_dataset_adapter_preserves_raw_payload(tmp_path):
    path = tmp_path / "test.jsonl"
    row = {
        "sequence_id": "sequence-1",
        "token_ids": [1, 2, 3, 4],
        "answer": "blue",
        "answer_token_ids": [5],
        "evidence_token_ranges": [[0, 2]],
        "evidence_to_answer_distance_tokens": 2,
        "evidence_block_indices": [0],
        "split_group_id": "profile",
        "evidence_placement_bin": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = JsonlEvaluationDataset(path)
    example = next(iter(dataset))

    assert example.sample_id == "sequence-1"
    assert example.sequence_length == 4
    assert example.evidence_token_count == 2
    assert example.payload == row
    assert dataset.descriptor["sha256"]


def test_validation_package_does_not_import_existing_systems():
    for path in (PACKAGE_ROOT / "cluster_router_validation").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from learnable_index" not in source
        assert "import learnable_index" not in source
        assert "from residual_cache" not in source
        assert "import residual_cache" not in source
        assert "from cluster_router_bridge" not in source
        assert "import cluster_router_bridge" not in source


def test_compact_torch_logits_keeps_exact_sufficient_statistics():
    torch = pytest.importorskip("torch")
    targets = (0, 1)
    reference = ModelRun("", distribution_payload=torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
    candidate = ModelRun("", distribution_payload=torch.tensor([[2.0, 1.0], [1.0, 2.0]]))

    state = compact_torch_logits(reference, candidate, targets)

    assert state.token_count == 2
    assert state.argmax_agreement_count == 2
    assert state.target_accuracy_count == 2
    assert state.reference_cross_entropy_sum > state.reference_entropy_sum
