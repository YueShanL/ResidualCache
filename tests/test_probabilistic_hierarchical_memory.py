from pathlib import Path
import math
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residual_cache.memory_benchmark import (  # noqa: E402
    assert_effective_baseline,
    default_five_way_benchmark,
    evaluate_scenario,
    make_static_cluster_scenario,
    make_temporal_conflict_scenario,
    sweep_camelot_hard,
)
from residual_cache.probabilistic_hierarchical_memory import (  # noqa: E402
    AdaptiveQuantileMemory,
    CamelotHardMemory,
    HierarchicalVMFMemory,
    MemoryConfig,
    RetrievalResult,
    TemporalVMFMemory,
    VMFPosteriorMemory,
    attention_readout,
    cosine_similarity,
    estimate_vmf_concentration,
    vmf_log_density,
    vmf_log_normalizer,
)


class VectorAndVMFTests(unittest.TestCase):
    def test_cosine_rejects_zero_and_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            cosine_similarity((0.0, 0.0), (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))

    def test_vmf_zero_concentration_is_uniform_and_alignment_is_likely(self):
        expected_uniform = -math.log(2.0 * math.pi)
        self.assertAlmostEqual(vmf_log_normalizer(2, 0.0), expected_uniform, places=12)
        aligned = vmf_log_density((1.0, 0.0), (1.0, 0.0), 4.0)
        orthogonal = vmf_log_density((0.0, 1.0), (1.0, 0.0), 4.0)
        opposite = vmf_log_density((-1.0, 0.0), (1.0, 0.0), 4.0)
        self.assertAlmostEqual(aligned - orthogonal, 4.0, places=10)
        self.assertAlmostEqual(orthogonal - opposite, 4.0, places=10)

    def test_three_dimensional_normalizer_matches_closed_form(self):
        for concentration in (0.1, 1.0, 10.0, 100.0):
            expected = (
                math.log(concentration)
                - math.log(4.0 * math.pi)
                - math.log(math.sinh(concentration))
            )
            self.assertAlmostEqual(
                vmf_log_normalizer(3, concentration), expected, places=10
            )

    def test_concentration_estimate_is_monotone_and_bounded(self):
        values = [
            estimate_vmf_concentration(resultant, 16, maximum=100.0)
            for resultant in (0.0, 0.2, 0.5, 0.9, 0.999999)
        ]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 100.0)


class BaselineTests(unittest.TestCase):
    def test_camelot_threshold_controls_slot_creation(self):
        memory = CamelotHardMemory(threshold=0.9, record_top_k=1)
        first = memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)
        near = memory.write((0.99, 0.05), (0.99, 0.05), (1.0,), time=2.0)
        far = memory.write((0.0, 1.0), (0.0, 1.0), (2.0,), time=3.0)

        self.assertTrue(first.created_new_slot)
        self.assertFalse(near.created_new_slot)
        self.assertTrue(far.created_new_slot)
        self.assertEqual(memory.slot_count, 2)

    def test_adaptive_quantile_changes_after_warmup(self):
        memory = AdaptiveQuantileMemory(
            initial_threshold=0.5,
            quantile=0.5,
            minimum_history=2,
        )
        memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)
        memory.write((1.0, 0.1), (1.0, 0.1), (1.0,), time=2.0)
        memory.write((1.0, -0.1), (1.0, -0.1), (1.0,), time=3.0)

        self.assertGreater(memory.threshold, 0.9)
        self.assertNotEqual(memory.threshold, memory.fixed_threshold)

    def test_exact_duplicate_count_correction_matches_uncompressed_attention(self):
        compressed = RetrievalResult(
            record_id="a",
            slot_id="a",
            score=0.0,
            original_key=(1.0, 0.0),
            original_value=(1.0, 0.0),
            source_token_or_span="a",
            write_time=1.0,
            source_authority=0.0,
            multiplicity=2,
        )
        single_a = RetrievalResult(
            **{**compressed.__dict__, "record_id": "a1", "multiplicity": 1}
        )
        single_a2 = RetrievalResult(
            **{**compressed.__dict__, "record_id": "a2", "multiplicity": 1}
        )
        other = RetrievalResult(
            record_id="b",
            slot_id="b",
            score=0.0,
            original_key=(0.0, 1.0),
            original_value=(0.0, 1.0),
            source_token_or_span="b",
            write_time=1.0,
            source_authority=0.0,
        )

        compressed_output, _ = attention_readout((1.0, 0.0), [compressed, other])
        full_output, _ = attention_readout((1.0, 0.0), [single_a, single_a2, other])

        for compressed_value, full_value in zip(compressed_output, full_output):
            self.assertAlmostEqual(compressed_value, full_value, places=12)

    def test_hard_threshold_sweep_is_non_degenerate_and_effective(self):
        rows = sweep_camelot_hard(
            make_static_cluster_scenario(),
            thresholds=(0.5, 0.9, 0.9999),
        )
        best = assert_effective_baseline(
            rows, minimum_recall=1.0, minimum_top1=1.0
        )

        self.assertEqual(best.recall_at_k, 1.0)
        self.assertGreater(len({row.slot_count for row in rows}), 1)
        self.assertTrue(all(row.memory_bytes > 0 for row in rows))


class ProbabilisticWriteTests(unittest.TestCase):
    def test_near_record_joins_and_far_record_gets_new_posterior_slot(self):
        memory = VMFPosteriorMemory(
            MemoryConfig(
                alpha=0.2,
                tau_new=0.4,
                count_exponent=0.25,
                concentration_prior_mass=0.5,
            )
        )
        first = memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)
        near = memory.write((1.0, 0.03), (1.0, 0.03), (1.0,), time=2.0)
        far = memory.write((0.0, 1.0), (0.0, 1.0), (2.0,), time=3.0)

        self.assertTrue(first.created_new_slot)
        self.assertFalse(near.created_new_slot)
        self.assertTrue(far.created_new_slot)
        self.assertGreater(far.probability_new, near.probability_new)
        self.assertAlmostEqual(
            far.probability_new
            + sum(probability for _slot, probability in far.existing_probabilities),
            1.0,
            places=12,
        )

    def test_tempered_count_reduces_large_cluster_prior_advantage(self):
        low_gamma = VMFPosteriorMemory(
            MemoryConfig(alpha=0.1, tau_new=1.0, count_exponent=0.0)
        )
        high_gamma = VMFPosteriorMemory(
            MemoryConfig(alpha=0.1, tau_new=1.0, count_exponent=0.9)
        )
        for memory in (low_gamma, high_gamma):
            for index in range(8):
                memory.write(
                    (1.0, 0.01 * ((index % 2) * 2 - 1)),
                    (1.0, 0.0),
                    (1.0,),
                    time=float(index + 1),
                )

        low_probability = low_gamma.write(
            (0.0, 1.0), (0.0, 1.0), (2.0,), time=9.0
        ).probability_new
        high_probability = high_gamma.write(
            (0.0, 1.0), (0.0, 1.0), (2.0,), time=9.0
        ).probability_new

        self.assertGreater(low_probability, high_probability)


class TemporalAndRetrievalTests(unittest.TestCase):
    @staticmethod
    def _one_slot_config(**changes):
        defaults = dict(
            alpha=1e-9,
            tau_new=1.0,
            count_exponent=0.0,
            enable_split_merge=False,
            route_top_k=1,
            record_top_k=2,
        )
        defaults.update(changes)
        return MemoryConfig(**defaults)

    def test_current_and_historical_queries_respect_supersession_time(self):
        memory = TemporalVMFMemory(self._one_slot_config())
        old = memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            source_token_or_span="Paris",
            conflict_group="alice-location",
            time=1.0,
        )
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            source_token_or_span="London",
            conflict_group="alice-location",
            supersedes=old.record_id,
            time=10.0,
        )

        current = memory.retrieve(
            (1.0, 0.0), query_key=(1.0, 0.0), time=11.0, temporal_policy="current"
        )
        historical = memory.retrieve(
            (1.0, 0.0),
            query_key=(1.0, 0.0),
            time=12.0,
            temporal_policy="historical",
            as_of=5.0,
        )

        self.assertEqual([result.source_token_or_span for result in current], ["London"])
        self.assertEqual([result.source_token_or_span for result in historical], ["Paris"])

    def test_native_key_reranking_selects_record_not_centroid_payload(self):
        memory = TemporalVMFMemory(self._one_slot_config(record_top_k=1))
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            source_token_or_span="x",
            time=1.0,
        )
        memory.write(
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            source_token_or_span="y",
            time=2.0,
        )

        result = memory.retrieve(
            (1.0, 1.0),
            query_key=(0.0, 1.0),
            time=3.0,
            top_k=1,
        )

        self.assertEqual(result[0].source_token_or_span, "y")
        self.assertEqual(result[0].original_value, (0.0, 1.0))

    def test_minimum_native_score_supports_absent_fact_abstention(self):
        memory = TemporalVMFMemory(
            self._one_slot_config(
                record_top_k=1,
                minimum_native_score=0.25,
            )
        )
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0,),
            source_token_or_span="known",
            time=1.0,
        )

        result = memory.retrieve(
            (0.0, 1.0),
            query_key=(0.0, 1.0),
            time=2.0,
        )

        self.assertEqual(result, [])

    def test_layer_and_kv_group_statistics_are_isolated(self):
        memory = TemporalVMFMemory(self._one_slot_config(record_top_k=1))
        first = memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0,),
            layer=0,
            head_or_kv_group="a",
            source_token_or_span="layer-0",
            time=1.0,
        )
        second = memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (2.0,),
            layer=1,
            head_or_kv_group="a",
            source_token_or_span="layer-1",
            time=2.0,
        )

        self.assertTrue(first.created_new_slot)
        self.assertTrue(second.created_new_slot)
        layer_zero = memory.retrieve(
            (1.0, 0.0),
            query_key=(1.0, 0.0),
            layer=0,
            head_or_kv_group="a",
            time=3.0,
        )
        layer_one = memory.retrieve(
            (1.0, 0.0),
            query_key=(1.0, 0.0),
            layer=1,
            head_or_kv_group="a",
            time=4.0,
        )
        self.assertEqual(layer_zero[0].source_token_or_span, "layer-0")
        self.assertEqual(layer_one[0].source_token_or_span, "layer-1")

    def test_authority_prior_breaks_equal_native_key_tie(self):
        memory = TemporalVMFMemory(
            self._one_slot_config(record_top_k=1, authority_bias=1.0)
        )
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0,),
            source_token_or_span="authoritative",
            source_authority=1.0,
            time=1.0,
        )
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (2.0,),
            source_token_or_span="untrusted",
            source_authority=0.0,
            time=2.0,
        )

        result = memory.retrieve(
            (1.0, 0.0), query_key=(1.0, 0.0), time=3.0
        )

        self.assertEqual(result[0].source_token_or_span, "authoritative")

    def test_decay_can_evict_an_individual_record_and_rebuild_statistics(self):
        memory = TemporalVMFMemory(
            self._one_slot_config(
                age_decay=1.0,
                minimum_effective_weight=0.1,
            )
        )
        memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)

        report = memory.maintain(time=4.0)

        self.assertEqual(len(report.evicted_record_ids), 1)
        self.assertEqual(memory.record_count, 0)
        self.assertEqual(memory.slot_count, 0)

    def test_budget_controller_respects_new_record_protection(self):
        memory = TemporalVMFMemory(
            self._one_slot_config(
                memory_budget_bytes=1,
                budget_step_size=0.01,
                new_record_protection=5.0,
            )
        )
        memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)

        protected = memory.maintain(time=2.0)
        expired = memory.maintain(time=7.0)

        self.assertEqual(protected.evicted_record_ids, ())
        self.assertEqual(len(expired.evicted_record_ids), 1)
        self.assertEqual(memory.record_count, 0)


class SplitMergeTests(unittest.TestCase):
    def test_bimodal_slot_splits_and_routes_to_each_child(self):
        config = MemoryConfig(
            alpha=1e-9,
            tau_new=1.0,
            count_exponent=0.0,
            minimum_child_mass=1.5,
            split_minimum_resultant=0.8,
            split_minimum_gain=0.1,
            split_patience=1,
            slot_penalty=0.01,
            split_cooldown=5.0,
            merge_minimum_similarity=0.99,
            route_top_k=1,
            child_top_k=1,
            record_top_k=1,
        )
        memory = HierarchicalVMFMemory(config)
        samples = (
            ((1.0, 0.05), "positive"),
            ((1.0, -0.05), "positive"),
            ((-1.0, 0.05), "negative"),
            ((-1.0, -0.05), "negative"),
        )
        for index, (vector, label) in enumerate(samples, start=1):
            memory.write(
                vector,
                vector,
                (1.0,) if label == "positive" else (-1.0,),
                source_token_or_span=label,
                time=float(index),
            )

        report = memory.maintain(time=5.0)
        positive = memory.retrieve((1.0, 0.0), query_key=(1.0, 0.0), time=6.0)
        negative = memory.retrieve((-1.0, 0.0), query_key=(-1.0, 0.0), time=7.0)

        self.assertEqual(len(report.split_slot_ids), 1)
        self.assertEqual(memory.leaf_count, 2)
        self.assertEqual(len(memory.parents), 1)
        self.assertEqual(positive[0].source_token_or_span, "positive")
        self.assertEqual(negative[0].source_token_or_span, "negative")

    def test_large_tight_slot_does_not_split_merely_due_to_count(self):
        config = MemoryConfig(
            alpha=1e-9,
            tau_new=1.0,
            count_exponent=0.0,
            minimum_child_mass=2.0,
            split_minimum_resultant=0.8,
            split_minimum_gain=0.2,
            split_patience=1,
            slot_penalty=0.01,
        )
        memory = HierarchicalVMFMemory(config)
        for index in range(20):
            vector = (1.0, 0.001 * ((index % 3) - 1))
            memory.write(vector, vector, (1.0,), time=float(index + 1))

        report = memory.maintain(time=21.0)

        self.assertFalse(report.split_slot_ids)
        self.assertEqual(memory.leaf_count, 1)
        self.assertEqual(len(memory.parents), 0)

    def test_fragmented_similar_slots_merge_when_savings_exceed_distortion(self):
        config = MemoryConfig(
            alpha=1.0,
            tau_new=0.0,
            count_exponent=0.0,
            minimum_child_mass=10.0,
            split_patience=1,
            split_cooldown=0.0,
            merge_cooldown=0.0,
            merge_minimum_similarity=0.99,
            slot_penalty=0.1,
        )
        memory = HierarchicalVMFMemory(config)
        memory.write((1.0, 0.0), (1.0, 0.0), (1.0,), time=1.0)
        memory.write((1.0, 0.01), (1.0, 0.01), (1.0,), time=2.0)
        self.assertEqual(memory.leaf_count, 2)

        report = memory.maintain(time=3.0)

        self.assertEqual(len(report.merged_slot_pairs), 1)
        self.assertEqual(memory.leaf_count, 1)


class BenchmarkTests(unittest.TestCase):
    def test_five_mandatory_variants_share_the_same_scenario(self):
        rows = default_five_way_benchmark(make_static_cluster_scenario())

        self.assertEqual(
            {row.method for row in rows},
            {
                "CAMELoT-Hard",
                "Adaptive-Quantile",
                "vMF-Posterior",
                "Temporal-vMF",
                "Hierarchical-vMF",
            },
        )
        self.assertTrue(all(row.scenario == "static_two_cluster" for row in rows))
        self.assertTrue(all(row.queries == 2 for row in rows))

    def test_temporal_vmf_avoids_stale_fact_that_centroid_baseline_injects(self):
        scenario = make_temporal_conflict_scenario()
        config = MemoryConfig(
            alpha=1e-9,
            tau_new=1.0,
            count_exponent=0.0,
            enable_split_merge=False,
            record_top_k=1,
        )
        temporal = evaluate_scenario(
            TemporalVMFMemory(config), scenario, method_name="Temporal-vMF"
        )
        baseline = evaluate_scenario(
            CamelotHardMemory(0.9, record_top_k=1),
            scenario,
            method_name="CAMELoT-Hard",
        )
        posterior_only = evaluate_scenario(
            VMFPosteriorMemory(config),
            scenario,
            method_name="vMF-Posterior",
        )

        self.assertEqual(temporal.top1_accuracy, 1.0)
        self.assertEqual(temporal.stale_fact_injection_rate, 0.0)
        self.assertGreater(baseline.stale_fact_injection_rate, 0.0)
        self.assertGreater(posterior_only.stale_fact_injection_rate, 0.0)
        self.assertLess(posterior_only.top1_accuracy, temporal.top1_accuracy)


if __name__ == "__main__":
    unittest.main()
