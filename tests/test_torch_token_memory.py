from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - lets the stdlib-only suite still run.
    torch = None


class EvaluationCliTests(unittest.TestCase):
    def test_progress_is_enabled_by_default_and_can_be_disabled(self):
        from residual_cache.camelot_model_eval import build_parser

        parser = build_parser()
        default = parser.parse_args(["--output-dir", "output"])
        disabled = parser.parse_args(
            ["--output-dir", "output", "--no-progress"]
        )
        explicit = parser.parse_args(
            ["--output-dir", "output", "--progress"]
        )
        chunk_override = parser.parse_args(
            [
                "--output-dir",
                "output",
                "--vmf-write-chunk-size",
                "1",
            ]
        )
        self.assertTrue(default.show_progress)
        self.assertFalse(disabled.show_progress)
        self.assertTrue(explicit.show_progress)
        self.assertEqual(default.vmf_write_chunk_size, 32)
        self.assertEqual(chunk_override.vmf_write_chunk_size, 1)


@unittest.skipIf(torch is None, "torch is not installed")
class TorchCamelotMemoryTests(unittest.TestCase):
    def setUp(self):
        from residual_cache.torch_token_memory import TokenMemoryConfig

        self.config = TokenMemoryConfig(
            method="camelot",
            slot_capacity=3,
            record_capacity=4,
            camelot_threshold=0.93,
        )

    def _bank(self):
        from residual_cache.torch_token_memory import TorchCamelotMemoryBank

        return TorchCamelotMemoryBank(
            batch_size=1,
            kv_heads=1,
            head_dim=32,
            device="cpu",
            dtype=torch.float32,
            config=self.config,
        )

    @staticmethod
    def _kv(vectors):
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return tensor.view(1, 1, len(vectors), 32)

    def test_empty_memory_returns_invalid_retrievals(self):
        bank = self._bank()
        keys = self._kv([[1.0] + [0.0] * 31])
        self.assertEqual(bank.allocated_capacity, 0)
        retrieved_keys, retrieved_values, valid = bank.retrieve(keys)
        self.assertEqual(tuple(retrieved_keys.shape), (1, 1, 1, 32))
        self.assertEqual(tuple(retrieved_values.shape), (1, 1, 1, 32))
        self.assertFalse(bool(valid.any()))

    def test_backing_storage_grows_only_with_novel_writes(self):
        bank = self._bank()
        key_a = self._kv([[1.0] + [0.0] * 31])
        key_b = self._kv([[0.99, 0.01] + [0.0] * 30])
        bank.write(key_a, key_a)
        self.assertEqual(bank.allocated_capacity, 1)
        bank.retrieve(key_b)
        bank.write(key_b, key_b)
        self.assertEqual(bank.allocated_capacity, 1)
        self.assertEqual(bank.snapshot()["maximum_slots_per_stream"], 3)

    def test_count_weighted_mean_matches_camelot(self):
        bank = self._bank()
        key_a = [1.0] + [0.0] * 31
        key_b = [0.99, 0.01] + [0.0] * 30
        value_a = [2.0] + [0.0] * 31
        value_b = [4.0] + [0.0] * 31
        bank.write(self._kv([key_a]), self._kv([value_a]))
        bank.write(self._kv([key_b]), self._kv([value_b]))
        self.assertEqual(int(bank.active.sum()), 1)
        self.assertEqual(int(bank.counts.sum()), 2)
        self.assertAlmostEqual(float(bank.keys[0, 0, 0]), 0.995, places=6)
        self.assertAlmostEqual(float(bank.values[0, 0, 0, 0]), 3.0, places=6)

    def test_novel_writes_fill_then_replace_oldest(self):
        bank = self._bank()
        basis = []
        for position in range(4):
            vector = [0.0] * 32
            vector[position] = 1.0
            basis.append(vector)
        bank.write(self._kv(basis[:3]), self._kv(basis[:3]))
        self.assertEqual(int(bank.active.sum()), 3)
        bank.write(self._kv([basis[3]]), self._kv([basis[3]]))
        self.assertEqual(int(bank.active.sum()), 3)
        self.assertEqual(bank.replacements, 1)

    def test_one_read_assignment_is_reused_by_following_write(self):
        bank = self._bank()
        first = self._kv([[1.0] + [0.0] * 31])
        second = self._kv([[0.99, 0.01] + [0.0] * 30])
        bank.write(first, first)
        self.assertEqual(bank.assignment_searches, 1)
        bank.retrieve(second)
        self.assertEqual(bank.assignment_searches, 2)
        bank.write(second, second)
        self.assertEqual(bank.assignment_searches, 2)

    def test_same_slot_writes_are_grouped_into_exact_weighted_mean(self):
        bank = self._bank()
        original_key = [1.0] + [0.0] * 31
        original_value = [3.0] + [0.0] * 31
        bank.write(
            self._kv([original_key]), self._kv([original_value])
        )
        new_keys = [
            [0.99, 0.01] + [0.0] * 30,
            [0.98, 0.02] + [0.0] * 30,
        ]
        new_values = [
            [6.0] + [0.0] * 31,
            [9.0] + [0.0] * 31,
        ]
        bank.retrieve(self._kv(new_keys))
        bank.write(self._kv(new_keys), self._kv(new_values))
        self.assertEqual(int(bank.active.sum()), 1)
        self.assertEqual(int(bank.counts.sum()), 3)
        self.assertAlmostEqual(
            float(bank.keys[0, 0, 0]),
            (1.0 + 0.99 + 0.98) / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(bank.values[0, 0, 0, 0]),
            6.0,
            places=6,
        )

@unittest.skipIf(torch is None, "torch is not installed")
class TorchVMFRecordMemoryTests(unittest.TestCase):
    def _bank(self, **overrides):
        from residual_cache.torch_token_memory import (
            TokenMemoryConfig,
            TorchVMFRecordMemoryBank,
        )

        values = {
            "method": "vmf_records",
            "slot_capacity": 4,
            "record_capacity": 5,
            "route_top_k": 2,
            "tau_new": 1.0,
        }
        values.update(overrides)
        return TorchVMFRecordMemoryBank(
            batch_size=1,
            kv_heads=1,
            head_dim=32,
            device="cpu",
            dtype=torch.float32,
            config=TokenMemoryConfig(**values),
        )

    @staticmethod
    def _kv(vectors):
        return torch.tensor(vectors, dtype=torch.float32).view(
            1, 1, len(vectors), 32
        )

    def test_multiple_original_records_share_one_slot(self):
        bank = self._bank()
        self.assertEqual(bank.allocated_slot_capacity, 0)
        self.assertEqual(bank.allocated_record_capacity, 0)
        first = [1.0] + [0.0] * 31
        second = [0.8, 0.6] + [0.0] * 30
        first_value = [3.0] + [0.0] * 31
        second_value = [7.0] + [0.0] * 31
        bank.write(self._kv([first]), self._kv([first_value]))
        self.assertEqual(bank.allocated_slot_capacity, 1)
        self.assertEqual(bank.allocated_record_capacity, 1)
        bank.write(self._kv([second]), self._kv([second_value]))
        self.assertEqual(int(bank.slot_active.sum()), 1)
        self.assertEqual(int(bank.record_active.sum()), 2)
        retrieved_key, retrieved_value, valid = bank.retrieve(
            self._kv([second])
        )
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.equal(retrieved_key, self._kv([second])))
        self.assertTrue(torch.equal(retrieved_value, self._kv([second_value])))

    def test_record_budget_evicts_without_corrupting_slot_counts(self):
        bank = self._bank(record_capacity=2)
        vectors = []
        for first in (1.0, 0.9, 0.8):
            vector = [first, math.sqrt(1.0 - first * first)] + [0.0] * 30
            vectors.append(vector)
        for vector in vectors:
            bank.write(self._kv([vector]), self._kv([vector]))
        self.assertEqual(int(bank.record_active.sum()), 2)
        self.assertEqual(int(bank.slot_counts.sum()), 2)
        self.assertEqual(bank.evicted_records, 1)

    def test_chunk_size_one_matches_individual_token_writes(self):
        whole = self._bank(
            tau_new=0.5,
            vmf_write_chunk_size=1,
            slot_capacity=6,
            record_capacity=6,
        )
        individual = self._bank(
            tau_new=0.5,
            vmf_write_chunk_size=1,
            slot_capacity=6,
            record_capacity=6,
        )
        vectors = [
            [1.0] + [0.0] * 31,
            [0.98, 0.02] + [0.0] * 30,
            [0.0, 1.0] + [0.0] * 30,
        ]
        window = self._kv(vectors)
        whole.write(window, window)
        for vector in vectors:
            token = self._kv([vector])
            individual.write(token, token)
        torch.testing.assert_close(
            whole.slot_resultants,
            individual.slot_resultants,
        )
        self.assertTrue(torch.equal(whole.slot_counts, individual.slot_counts))
        self.assertTrue(torch.equal(whole.slot_active, individual.slot_active))
        self.assertTrue(torch.equal(whole.record_keys, individual.record_keys))
        self.assertTrue(torch.equal(whole.record_slots, individual.record_slots))
        self.assertEqual(whole.created_slots, individual.created_slots)
        self.assertEqual(
            whole.assigned_existing, individual.assigned_existing
        )
        self.assertAlmostEqual(
            whole.posterior_new_sum,
            individual.posterior_new_sum,
            places=10,
        )

    def test_chunked_write_respects_slot_and_record_budgets(self):
        bank = self._bank(
            slot_capacity=2,
            record_capacity=3,
            vmf_write_chunk_size=32,
            tau_new=1.0,
        )
        vectors = []
        for first in (1.0, 0.98, 0.96, 0.94, 0.92):
            vectors.append(
                [first, math.sqrt(1.0 - first * first)]
                + [0.0] * 30
            )
        window = self._kv(vectors)
        bank.write(window, window)
        self.assertLessEqual(int(bank.slot_active.sum()), 2)
        self.assertEqual(int(bank.record_active.sum()), 3)
        self.assertEqual(int(bank.slot_counts.sum()), 3)
        self.assertEqual(bank.evicted_records, 2)
        self.assertEqual(bank.allocated_slot_capacity, 2)
        self.assertEqual(bank.allocated_record_capacity, 3)

    def test_high_dimensional_log_normalizer_matches_exact_oracle(self):
        from residual_cache.probabilistic_hierarchical_memory import (
            vmf_log_normalizer,
        )
        from residual_cache.torch_token_memory import (
            vmf_log_normalizer_high_dim_torch,
        )

        for dimension, concentrations in (
            (32, (0.0, 1.0, 10.0, 100.0)),
            (64, (0.0, 1.0, 10.0, 100.0, 500.0)),
        ):
            approximate = vmf_log_normalizer_high_dim_torch(
                torch.tensor(concentrations), dimension
            )
            for concentration, observed in zip(concentrations, approximate):
                expected = vmf_log_normalizer(dimension, concentration)
                self.assertAlmostEqual(
                    float(observed), expected, delta=2e-4
                )


@unittest.skipIf(torch is None, "torch is not installed")
class LayerOnlineAdapterTests(unittest.TestCase):
    class FakeAttention:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.head_dim = 32
            self.num_key_value_groups = 2
            self.training = False

    @staticmethod
    def _causal_mask(tokens):
        minimum = torch.finfo(torch.float32).min
        upper = torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1
        )
        mask = torch.zeros((1, 1, tokens, tokens), dtype=torch.float32)
        return mask.masked_fill(upper[None, None], minimum)

    def test_first_window_is_exact_native_attention_then_written(self):
        from residual_cache.gemma4_memory_adapter import (
            Gemma4MemoryController,
            eager_attention_reference,
        )
        from residual_cache.torch_token_memory import TokenMemoryConfig

        torch.manual_seed(7)
        module = self.FakeAttention(0)
        query = torch.randn(1, 2, 3, 32)
        key = torch.randn(1, 1, 3, 32)
        value = torch.randn(1, 1, 3, 32)
        mask = self._causal_mask(3)
        expected_output, expected_weights = eager_attention_reference(
            module, query, key, value, mask, scaling=1.0
        )
        controller = Gemma4MemoryController(
            TokenMemoryConfig(
                method="camelot", slot_capacity=8, record_capacity=8
            )
        )
        actual_output, actual_weights = controller.attend(
            module,
            query,
            key,
            value,
            mask,
            dropout=0.0,
            scaling=1.0,
            softcap=None,
        )
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(actual_weights, expected_weights)
        self.assertEqual(controller.written_tokens, 3)
        self.assertEqual(controller.retrieved_tokens, 0)

    def test_registered_backend_keeps_eager_causal_mask_creation(self):
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            eager_mask,
        )
        from residual_cache.gemma4_memory_adapter import (
            ATTENTION_BACKEND_NAME,
            register_attention_backend,
        )

        register_attention_backend()
        self.assertIs(
            ALL_MASK_ATTENTION_FUNCTIONS[ATTENTION_BACKEND_NAME], eager_mask
        )

    def test_second_window_prepends_one_retrieval_per_token(self):
        from residual_cache.gemma4_memory_adapter import Gemma4MemoryController
        from residual_cache.torch_token_memory import TokenMemoryConfig

        torch.manual_seed(8)
        module = self.FakeAttention(0)
        controller = Gemma4MemoryController(
            TokenMemoryConfig(
                method="camelot", slot_capacity=8, record_capacity=8
            )
        )
        query = torch.randn(1, 2, 3, 32)
        key = torch.randn(1, 1, 3, 32)
        value = torch.randn(1, 1, 3, 32)
        mask = self._causal_mask(3)
        controller.attend(
            module, query, key, value, mask, dropout=0.0, scaling=1.0, softcap=None
        )
        _output, weights = controller.attend(
            module, query, key, value, mask, dropout=0.0, scaling=1.0, softcap=None
        )
        self.assertEqual(tuple(weights.shape), (1, 2, 3, 6))
        self.assertEqual(controller.retrieved_tokens, 3)
        self.assertEqual(controller.written_tokens, 6)

    def test_layers_have_independent_banks(self):
        from residual_cache.gemma4_memory_adapter import Gemma4MemoryController
        from residual_cache.torch_token_memory import TokenMemoryConfig

        controller = Gemma4MemoryController(
            TokenMemoryConfig(
                method="camelot", slot_capacity=4, record_capacity=4
            )
        )
        query = torch.ones(1, 2, 2, 32)
        key = torch.ones(1, 1, 2, 32)
        value = torch.ones(1, 1, 2, 32)
        mask = self._causal_mask(2)
        for layer in (0, 1):
            controller.attend(
                self.FakeAttention(layer),
                query,
                key,
                value,
                mask,
                dropout=0.0,
                scaling=1.0,
                softcap=None,
            )
        self.assertEqual(set(controller.banks), {0, 1})
        self.assertIsNot(controller.banks[0], controller.banks[1])


@unittest.skipIf(torch is None, "torch is not installed")
class SequentialProtocolTests(unittest.TestCase):
    def test_batchify_and_targets_match_sequential_stream_protocol(self):
        from residual_cache.camelot_model_eval import (
            contiguous_stream_batchify,
            sequential_windows,
        )

        streams = contiguous_stream_batchify(torch.arange(12), batch_size=2)
        self.assertEqual(streams.tolist(), [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        windows = list(sequential_windows(streams, window_size=2))
        self.assertEqual(windows[0][2].tolist(), [[0, 1], [6, 7]])
        self.assertEqual(windows[0][3].tolist(), [[1, 2], [7, 8]])
        self.assertEqual(windows[-1][2].tolist(), [[4], [10]])
        self.assertEqual(windows[-1][3].tolist(), [[5], [11]])

    def test_first_window_equivalence_guard_accepts_equal_methods(self):
        from residual_cache.camelot_model_eval import (
            validate_first_window_equivalence,
        )

        rows = [
            {
                "method": "base",
                "window_size": 8,
                "window_index": 0,
                "negative_log_likelihood": 12.0,
            },
            {
                "method": "camelot",
                "window_size": 8,
                "window_index": 0,
                "negative_log_likelihood": 12.0,
            },
        ]
        report = validate_first_window_equivalence(rows)
        self.assertTrue(report["checked"])
        self.assertEqual(report["maximum_absolute_nll_difference"], 0.0)

    def test_first_window_equivalence_guard_rejects_causal_leak(self):
        from residual_cache.camelot_model_eval import (
            validate_first_window_equivalence,
        )

        rows = [
            {
                "method": "base",
                "window_size": 8,
                "window_index": 0,
                "negative_log_likelihood": 12.0,
            },
            {
                "method": "camelot",
                "window_size": 8,
                "window_index": 0,
                "negative_log_likelihood": 4.0,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            validate_first_window_equivalence(rows)

    def test_cache_relative_position_policy_matches_past_cache_semantics(self):
        from residual_cache.camelot_model_eval import window_position_offset

        self.assertEqual(
            window_position_offset(
                method="camelot",
                policy="cache_relative",
                window_index=0,
                stream_position=0,
                window_size=512,
            ),
            0,
        )
        self.assertEqual(
            window_position_offset(
                method="camelot",
                policy="cache_relative",
                window_index=3,
                stream_position=1536,
                window_size=512,
            ),
            512,
        )
        self.assertEqual(
            window_position_offset(
                method="base",
                policy="cache_relative",
                window_index=3,
                stream_position=1536,
                window_size=512,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
