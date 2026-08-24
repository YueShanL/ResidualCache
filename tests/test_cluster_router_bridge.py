from __future__ import annotations

from pathlib import Path
import sys
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cluster_router_bridge import (  # noqa: E402
    BlockInput,
    BlockMemoryIngestor,
    ClusterSelection,
    HierarchicalLayerMemoryAdapter,
    KVPayloadStore,
    KVRecordPayload,
    LayerKVCacheBuilder,
    LearnableRouterEncoder,
    MemoryRecordInput,
    RecordRef,
    TorchKVBlockInputBuilder,
)
from residual_cache.probabilistic_hierarchical_memory import (  # noqa: E402
    HierarchicalVMFMemory,
    MemoryConfig,
    TemporalVMFMemory,
    VMFPosteriorMemory,
)

try:
    import torch
except ImportError:  # pragma: no cover - permits the stdlib-only suite.
    torch = None


def _record(
    layer: int,
    position: int,
    *,
    direction=(1.0, 0.0),
    value=(1.0,),
    time: float | None = None,
    payload: KVRecordPayload | None = None,
) -> MemoryRecordInput:
    return MemoryRecordInput(
        layer=layer,
        logical_positions=(position,),
        memory_index=direction,
        original_key=direction,
        original_value=value,
        kv_payload=payload,
        time=time,
    )


def _native_leaf_state(memory):
    return {
        slot.id: (
            tuple(slot.record_ids),
            slot.centroid,
            slot.effective_count,
            slot.resultant_length,
            slot.concentration,
            slot.weighted_scatter,
            slot.value_conflict_score,
            slot.parent_id,
        )
        for slot in memory.leaves.values()
    }


class IsolationAndIngestionTests(unittest.TestCase):
    @staticmethod
    def _memory(**overrides):
        values = dict(
            alpha=1e-9,
            tau_new=1.0,
            count_exponent=0.0,
            use_temporal_weights=False,
            enable_split_merge=False,
        )
        values.update(overrides)
        return VMFPosteriorMemory(MemoryConfig(**values))

    def test_existing_packages_remain_independent(self):
        for path in (PACKAGE_ROOT / "learnable_index").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("import cluster_router_bridge", source)
            self.assertNotIn("from cluster_router_bridge", source)
            self.assertNotIn("import residual_cache", source)
            self.assertNotIn("from residual_cache", source)
        for path in (PACKAGE_ROOT / "residual_cache").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("import cluster_router_bridge", source)
            self.assertNotIn("from cluster_router_bridge", source)
            self.assertNotIn("import learnable_index", source)
            self.assertNotIn("from learnable_index", source)

    def test_router_metadata_does_not_change_native_write_state(self):
        control = self._memory()
        bridged = self._memory()
        records = (
            _record(0, 0, direction=(1.0, 0.0), value=(1.0,), time=1.0),
            _record(0, 1, direction=(0.99, 0.01), value=(2.0,), time=2.0),
        )
        control_decisions = []
        for record in records:
            control_decisions.append(
                control.write(
                    record.memory_index,
                    record.original_key,
                    record.original_value,
                    layer=0,
                    head_or_kv_group=record.head_or_kv_group,
                    source_token_or_span=record.logical_positions,
                    time=record.time,
                )
            )

        writer = HierarchicalLayerMemoryAdapter({0: bridged})
        result = BlockMemoryIngestor(writer).ingest(
            BlockInput(
                block_id="block-0",
                start_position=0,
                end_position=2,
                router_key=(1.0, 0.0),
                records=records,
            )
        )

        self.assertEqual(
            [decision.slot_id for decision in control_decisions],
            [placement.cluster_id for placement in result.placements],
        )
        self.assertEqual(_native_leaf_state(control), _native_leaf_state(bridged))
        self.assertEqual(control.memory_bytes, bridged.memory_bytes)
        leaf = next(iter(bridged.leaves.values()))
        self.assertEqual(leaf.router_record_count, 2)
        self.assertEqual(leaf.router_block_count, 1)
        self.assertAlmostEqual(leaf.router_mass, 1.0)
        self.assertTrue(
            all(record.router_key == (1.0, 0.0) for record in bridged.records.values())
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_block_builder_splits_layers_without_exposing_router_key_to_indexer(self):
        seen = []

        def indexer(layer, key, value):
            seen.append((layer, key.clone(), value.clone()))
            return key.detach().float().reshape(-1)

        layer_kv = {
            0: (
                torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4),
                torch.arange(8, 16, dtype=torch.float32).view(1, 1, 2, 4),
            ),
            3: (
                torch.arange(16, 24, dtype=torch.float32).view(1, 1, 2, 4),
                torch.arange(24, 32, dtype=torch.float32).view(1, 1, 2, 4),
            ),
        }
        block = TorchKVBlockInputBuilder(indexer).build(
            block_id="block",
            start_position=10,
            router_key=(0.25, 0.75),
            layer_kv=layer_kv,
        )

        self.assertEqual(len(block.records), 4)
        self.assertEqual(block.expected_records_by_layer, {0: 2, 3: 2})
        self.assertEqual(
            [record.logical_positions for record in block.records],
            [(10,), (11,), (10,), (11,)],
        )
        self.assertEqual([row[0] for row in seen], [0, 0, 3, 3])
        self.assertTrue(all(tuple(row[1].shape) == (1, 1, 1, 4) for row in seen))
        self.assertEqual(block.router_key, (0.25, 0.75))


class MemoryOwnedRouterDistributionTests(unittest.TestCase):
    def test_failed_record_never_contributes_router_mass(self):
        memory = VMFPosteriorMemory(
            MemoryConfig(alpha=1e-9, tau_new=1.0, count_exponent=0.0)
        )
        memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0,),
            router_key=(1.0, 0.0),
            router_block_id="partial",
            router_block_size=2,
        )
        with self.assertRaises(ValueError):
            memory.write(
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 2.0),
                router_key=(1.0, 0.0),
                router_block_id="partial",
                router_block_size=2,
            )

        leaf = next(iter(memory.leaves.values()))
        self.assertEqual(memory.record_count, 1)
        self.assertEqual(leaf.router_record_count, 1)
        self.assertAlmostEqual(leaf.router_mass, 0.5)

    def test_split_rebuilds_mass_from_each_child_actual_records(self):
        memory = HierarchicalVMFMemory(
            MemoryConfig(
                alpha=1e-9,
                tau_new=1.0,
                count_exponent=0.0,
                minimum_child_mass=0.5,
                split_minimum_resultant=0.9,
                split_minimum_gain=0.01,
                split_patience=1,
                slot_penalty=0.001,
                split_cooldown=5.0,
                merge_minimum_similarity=0.99,
            )
        )
        directions = ((1.0, 0.0), (-1.0, 0.0), (-1.0, 0.01), (-1.0, -0.01))
        for position, direction in enumerate(directions):
            memory.write(
                direction,
                direction,
                (1.0,),
                time=float(position + 1),
                router_key=(1.0, 0.0),
                router_block_id="block-x",
                router_block_size=4,
            )

        report = memory.maintain(time=5.0)
        masses = sorted(slot.router_mass for slot in memory.leaves.values())

        self.assertEqual(len(report.split_slot_ids), 1)
        self.assertEqual(masses, [0.25, 0.75])
        self.assertAlmostEqual(sum(masses), 1.0)
        for slot in memory.leaves.values():
            self.assertEqual(slot.router_record_count, len(slot.record_ids))
            self.assertEqual(slot.router_block_weights, (("block-x", slot.router_mass),))

    def test_merge_rebuilds_one_distribution_from_combined_records(self):
        memory = HierarchicalVMFMemory(
            MemoryConfig(
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
        )
        for index, native in enumerate(((1.0, 0.0), (1.0, 0.01))):
            memory.write(
                native,
                native,
                (1.0,),
                time=float(index + 1),
                router_key=(1.0, 0.0),
                router_block_id=f"block-{index}",
                router_block_size=1,
            )
        self.assertEqual(memory.leaf_count, 2)

        report = memory.maintain(time=3.0)
        leaf = next(iter(memory.leaves.values()))

        self.assertEqual(len(report.merged_slot_pairs), 1)
        self.assertEqual(leaf.router_record_count, 2)
        self.assertEqual(leaf.router_block_count, 2)
        self.assertAlmostEqual(leaf.router_mass, 2.0)

    def test_vmf_router_selects_aligned_cluster_and_returns_actual_members(self):
        memory = HierarchicalVMFMemory(
            MemoryConfig(
                alpha=1.0,
                tau_new=0.0,
                count_exponent=0.0,
                enable_split_merge=False,
                router_count_exponent=0.0,
                router_concentration_prior_mass=0.25,
            )
        )
        first = memory.write(
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0,),
            router_key=(1.0, 0.0),
            router_block_id="x",
            router_block_size=1,
        )
        memory.write(
            (0.0, 1.0),
            (0.0, 1.0),
            (1.0,),
            router_key=(0.0, 1.0),
            router_block_id="y",
            router_block_size=1,
        )

        selected = memory.retrieve_router_clusters((1.0, 0.0), top_n=1)

        self.assertEqual(selected[0].slot_id, first.slot_id)
        self.assertEqual(selected[0].record_ids, (first.record_id,))
        self.assertGreater(selected[0].probability, 0.5)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_encoder_accepts_an_existing_router_without_import_coupling(self):
        class Tower(torch.nn.Module):
            def forward(self, values):
                return torch.nn.functional.normalize(values, dim=-1)

        class Router:
            query_network = Tower()
            key_network = Tower()

        encoder = LearnableRouterEncoder(Router(), torch.device("cpu"))
        self.assertEqual(
            encoder.encode_block((3.0, 4.0)),
            (0.6000000238418579, 0.800000011920929),
        )
        self.assertEqual(encoder.encode_query((0.0, 2.0)), (0.0, 1.0))


@unittest.skipIf(torch is None, "torch is not installed")
class MaintenanceAndDynamicKVTests(unittest.TestCase):
    @staticmethod
    def _payload(position: int, value: float) -> KVRecordPayload:
        key = torch.full((1, 1, 1, 2), value, dtype=torch.float32)
        return KVRecordPayload(key=key, value=key + 10.0, logical_positions=(position,))

    def test_eviction_removes_record_from_router_vmf_and_payload_store(self):
        memory = TemporalVMFMemory(
            MemoryConfig(
                alpha=1e-9,
                tau_new=1.0,
                count_exponent=0.0,
                enable_split_merge=False,
                age_decay=1.0,
                minimum_effective_weight=0.5,
                new_record_protection=0.0,
            )
        )
        payload_store = KVPayloadStore()
        writer = HierarchicalLayerMemoryAdapter({0: memory})
        ingestor = BlockMemoryIngestor(writer, payload_store=payload_store)
        ingestor.ingest(
            BlockInput(
                "old",
                0,
                1,
                (1.0, 0.0),
                (_record(0, 0, time=1.0, payload=self._payload(0, 1.0)),),
            )
        )
        self.assertEqual(len(memory.retrieve_router_clusters((1.0, 0.0))), 1)
        self.assertEqual(len(payload_store), 1)

        report = memory.maintain(time=2.0)
        ingestor.synchronize()

        self.assertEqual(len(report.evicted_record_ids), 1)
        self.assertEqual(memory.retrieve_router_clusters((1.0, 0.0)), [])
        self.assertEqual(len(payload_store), 0)

    def test_internal_selection_replays_whole_cluster_and_packs_dynamic_lengths(self):
        memories = {
            layer: VMFPosteriorMemory(
                MemoryConfig(alpha=1e-9, tau_new=1.0, count_exponent=0.0)
            )
            for layer in (0, 1)
        }
        payload_store = KVPayloadStore()
        writer = HierarchicalLayerMemoryAdapter(memories)
        ingestor = BlockMemoryIngestor(writer, payload_store=payload_store)
        ingestor.ingest(
            BlockInput(
                "block",
                2,
                9,
                (1.0, 0.0),
                (
                    _record(0, 8, payload=self._payload(8, 8.0)),
                    _record(0, 2, payload=self._payload(2, 2.0)),
                    _record(1, 5, payload=self._payload(5, 5.0)),
                ),
            )
        )
        selections = ingestor.select((1.0, 0.0), top_n=1)
        records = ingestor.records_for_selections(selections)
        views = LayerKVCacheBuilder(payload_store).build(
            records, selected_clusters=selections
        )

        self.assertEqual(views[0].logical_positions, (2, 8))
        self.assertEqual(views[1].logical_positions, (5,))
        self.assertEqual(tuple(views[0].key.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(views[1].key.shape), (1, 1, 1, 2))
        self.assertEqual(len(selections[0][0].record_refs), 2)
        query0 = torch.ones((1, 1, 1, 2))
        query1 = torch.ones((1, 1, 1, 2))
        self.assertEqual(
            tuple(torch.matmul(query0, views[0].key.transpose(2, 3)).shape),
            (1, 1, 1, 2),
        )
        self.assertEqual(
            tuple(torch.matmul(query1, views[1].key.transpose(2, 3)).shape),
            (1, 1, 1, 1),
        )

    def test_manual_selection_contract_remains_usable_for_cache_packing(self):
        payload_store = KVPayloadStore()
        ref = RecordRef(0, "record")
        payload_store.register(ref, self._payload(3, 3.0))
        selection = ClusterSelection(
            0, "cluster", 1.0, 0.0, 1, 1.0, record_refs=(ref,)
        )

        view = LayerKVCacheBuilder(payload_store).build(
            {0: (ref,)}, selected_clusters={0: (selection,)}
        )[0]

        self.assertEqual(view.selected_cluster_ids, ("cluster",))
        self.assertEqual(view.record_refs, (ref,))


if __name__ == "__main__":
    unittest.main()
