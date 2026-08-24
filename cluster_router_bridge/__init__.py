"""Independent bridge between learned block routing and clustered KV memory.

Neither :mod:`learnable_index` nor :mod:`residual_cache` imports this package.
The bridge only transports block keys and packs dynamic K/V; each memory owns
the record-bound router metadata and its derived leaf vMF distributions.
"""

from .bridge import (
    BlockMemoryIngestor,
    HierarchicalLayerMemoryAdapter,
    MemoryWriter,
)
from .block_input import TorchKVBlockInputBuilder, flattened_key_index, mean_kv_index
from .contracts import (
    BlockInput,
    BlockWriteResult,
    ClusterMembership,
    ClusterSelection,
    KVRecordPayload,
    LayerKVView,
    MemoryRecordInput,
    MemoryWritePlacement,
    RecordRef,
)
from .kv_cache import KVPayloadStore, LayerKVCacheBuilder
from .learnable_router import LearnableRouterEncoder

__all__ = [
    "BlockInput",
    "BlockMemoryIngestor",
    "BlockWriteResult",
    "ClusterMembership",
    "ClusterSelection",
    "HierarchicalLayerMemoryAdapter",
    "KVPayloadStore",
    "KVRecordPayload",
    "LayerKVCacheBuilder",
    "LayerKVView",
    "LearnableRouterEncoder",
    "MemoryRecordInput",
    "MemoryWritePlacement",
    "MemoryWriter",
    "RecordRef",
    "TorchKVBlockInputBuilder",
    "flattened_key_index",
    "mean_kv_index",
]
