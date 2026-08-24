# Cluster Router Bridge

This package is the integration boundary between the two standalone systems:

```text
learnable_index -> query/block embeddings
residual_cache  -> autonomous K/V record clustering and maintenance
```

Neither existing package imports this bridge or the other package. The bridge
uses a learned block key only as an opaque read index. Memory owns that key as
record metadata, while writing and classifying the record solely through its
existing `memory_index`, `original_key`, and `original_value` fields.

## Data flow

1. Use `TorchKVBlockInputBuilder` to split an all-layer block into layer-local
   token records, deriving the memory classification feature only from each
   record's own K/V.
2. Write each layer-local K/V record together with the completed block's key,
   block identity, and layer-local normalization count through memory's public
   `write` API.
3. On every leaf refresh, memory derives the router-key vMF only from records
   that are currently present in that leaf.
4. At retrieval, encode one layer-40 query and independently select the
   highest posterior clusters for every layer.
5. Pack all active records from the selected clusters into a variable-length
   `LayerKVView` for the next eligible forward.

The classes above remain the reference/public bridge for the existing
hierarchical memory. The concrete 4096-token validation runner uses the same
ownership contract through `GpuLocalClusterMemory`: the model advances once
with a 256-token retained cache. Appending the next mechanical block grows the
forward context to 320, after which the learned key is prepared and the oldest
64-token block is unloaded. CPU keeps only a compact locality-code-to-slot map
and prefetches a bounded list of slot IDs. Native K/V classification, exact vMF
posterior scoring, record writes, eviction subtraction, and sufficient-
statistic refresh all run on the GPU over that local list. Every unloaded
record obtains its own candidate posterior from one shared pre-commit memory
state; all records in the block then commit together. The learned block key
remains metadata for the independent router-vMF and never enters native cluster
assignment.

For a block producing `N` records in one layer, every record contributes
`1 / N` to the cluster's router-key distribution. This normalization affects
only the derived read index. It never changes memory counts, retention,
split/merge, utility, or attention multiplicity.

The vMF cache is a `LeafSlot` derivative, not adapter-owned state. A failed
write contributes nothing. Eviction removes the contribution on the same
maintenance refresh, and split/merge recomputes each new leaf from its actual
`record_ids`. `retrieve_router_clusters` is a pure read and returns the current
record IDs of the whole selected leaf for replay. Router posterior parameters
live in `MemoryConfig` under the `router_*` names.

`BlockMemoryIngestor` accepts any implementation of the small `MemoryWriter`
protocol. `HierarchicalLayerMemoryAdapter` is a duck-typed adapter for one
existing `ProbabilisticHierarchicalMemory` per layer; it retains no router keys
or probability distributions. `LearnableRouterEncoder` loads an existing
checkpoint lazily, so importing the bridge does not import either standalone
system.

`TorchKVBlockInputBuilder` defaults to the record's flattened native K as the
memory classification feature. A different existing memory statistic can be
passed as `memory_indexer`; `mean_kv_index` is provided as an explicit
K/V-mean alternative. The callback never receives the learned block key.
