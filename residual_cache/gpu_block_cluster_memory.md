# Router-key block-record GPU memory

`gpu_block_cluster_memory.py` is an independent memory implementation. It does
not add a block mode to `GpuLocalClusterMemory` and it does not subclass that
token-record implementation.

## Record contract

One memory instance still owns one physical decoder layer. One call to
`ingest_block` creates exactly one record containing:

- one learned router block-key direction used for native cluster assignment;
- the original complete layer-local K/V block as replay payload;
- the block identity and contiguous logical positions;
- one cluster-local attention-usage EMA value.

The configured `block_size` fixes both the record contract and GPU backing
storage. Partial blocks are rejected. A block that has only partly crossed the
local-context boundary must remain local until the complete block can be
unloaded atomically; a dedicated streaming adapter can therefore retain between
`local_context_length` and `local_context_length + block_size - 1` tokens.

## Classification and retrieval

The learned block key is normalized and passed once through the same bounded
locality candidate lookup and explicit new-vs-existing vMF posterior used by
the token memory. A block is then committed as one indivisible record.

The same cluster sufficient statistics rank clusters for a learned query key.
There is no second metadata-only router vMF because the router key is now the
native classification feature. Original K/V never participates in assignment.

The two memory implementations share only the functions in
`gpu_vmf_posterior.py`:

- bounded CPU locality buckets;
- random-hyperplane locality codes;
- vMF concentration/log-base refresh;
- new-vs-existing posterior scoring.

Payload storage, record identity, packing, usage tracking, eviction, capacities,
and public APIs remain separately implemented.

## Public entry

```python
from residual_cache.gpu_block_cluster_memory import (
    GpuBlockClusterMemory,
    GpuBlockClusterMemoryConfig,
)

memory = GpuBlockClusterMemory(
    kv_heads=4,
    head_dim=256,
    router_dim=128,
    device="cuda",
    dtype=torch.bfloat16,
    config=GpuBlockClusterMemoryConfig(block_size=64),
)

record_id = memory.ingest_block(
    key,
    value,
    router_key=learned_block_key,
    block_id=block_id,
    logical_positions=positions,
)
```

`selected_kv_blocks` returns `PackedBlockKV`. Its `record_slices` and
`token_record_ids` explicitly map packed attention positions back to block
records, allowing a later model adapter to aggregate token attention into one
usage value per block before calling `observe_recall_usage`.
