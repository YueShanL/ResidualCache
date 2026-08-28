# Cluster Router End-to-End Validation

This package separates expensive model execution from metric computation:

```text
dataset adapter + model adapter
            |
            v
        collect runner
            |
            +-- run_manifest.json
            +-- samples.jsonl
            +-- errors.jsonl
            |
            v
       offline metrics
            |
            +-- metrics.json
            +-- sample_metrics.jsonl
            +-- condition_summary.csv
```

The package imports none of `learnable_index`, `residual_cache`, or
`cluster_router_bridge`. A concrete model adapter may import all three systems
and is loaded only through a configured factory.

## Built-in comparisons

For every sample and every configured per-layer cluster budget, collection runs:

- `full_context`: quality reference;
- `evidence_only`: unconstrained upper bound where the correct source context
  is the only non-local history; this is not limited by memory retention,
  clustering, or Top-N;
- `local_only`: no historical memory;
- `fixed_policy@N`: most recent current leaves;
- `learned_router@N`: highest learned router score/probability;
- `oracle_cluster@N`: highest teacher-attention mass, with evidence overlap as
  the configured fallback.

The three cluster policies are implemented by the runner and select independently
per decoder layer. The model adapter only exposes current leaf candidates and
executes a supplied selection. This keeps policy comparisons identical across
model implementations.

## Dataset interface

An `EvaluationDataset` exposes a JSON-compatible `descriptor` and yields
`EvaluationExample`. The included `JsonlEvaluationDataset` understands the
existing ConvoMem synthesis fields:

- `sequence_id`, `token_ids`;
- `answer`, `answer_token_ids`;
- `evidence_token_ranges`, `evidence_block_indices`;
- `evidence_to_answer_distance_tokens`;
- `split_group_id`, `evidence_placement_bin`.

The complete original JSON row remains available as `example.payload` to the
model adapter. A different dataset only needs another factory returning the same
contract.

## Model interface

An `EvaluationModel` has a JSON-compatible `descriptor` and an `open(example)`
method returning an `EvaluationSession`. The session implements:

```python
cluster_candidates() -> Sequence[ClusterCandidate]
run_full_context() -> ModelRun
run_evidence_only() -> ModelRun
run_local_only() -> ModelRun
run_with_clusters(selection, strategy=..., budget=...) -> ModelRun
compact_distribution(full_run, candidate_run) -> DistributionState | None
```

`ClusterCandidate.record_ids` must be the leaf's current actual records. Expose
all current leaves, including leaves without a learned score; this lets oracle
coverage reveal evidence that the learned index cannot select. Each candidate
also carries the layer, replay token count, recency, learned score, teacher
attention mass, and evidence overlap.

`ModelRun.distribution_payload` is transient and never serialized. A PyTorch
adapter can place exact `[answer_tokens, vocabulary]` logits there and call
`compact_torch_logits` from `compact_distribution`. Only sufficient statistics
for exact NLL, full-context KL, argmax agreement, and target accuracy enter the
state file.

## Configuration and commands

See `configs/cluster_router_validation.example.json`. Factories use
`package.module:callable` syntax and receive the configured keyword arguments.
The concrete local 4096-token/256-example Gemma 4 run is fixed in
`configs/cluster_router_validation_convomem4096_512.json`. It keeps Gemma 4's
native 512-token sliding-attention span and grows to 576 tokens for each block
transaction. Its ConvoMem rows
are synthesized in process memory and are never written as an intermediate
dataset.

```bash
python -m cluster_router_validation collect \
  --config configs/cluster_router_validation.example.json

python -m cluster_router_validation metrics \
  --state-dir outputs/cluster_router_validation/state \
  --output-dir outputs/cluster_router_validation/metrics \
  --config configs/cluster_router_metrics.example.json
```

For an HPC allocation, use the fixed task runner and its dedicated config:

```bash
python -u scripts/run_cluster_router_validation_hpc.py \
  --config configs/cluster_router_validation_convomem4096_hpc.json
```

`router.path` is the only model artifact path added by this validation stage;
`model.name` and `data.dataset_name` remain Hugging Face repository IDs. With
`paths.use_tmp_workspace=true`, Hugging Face caches, generated configs, state
JSONL, and all transient memory-test artifacts are placed below the configurable
`paths.tmp_workspace_root`. The pipeline atomically exports only
`metrics/metrics.json`, `metrics/sample_metrics.jsonl`, and
`metrics/condition_summary.csv`. When `paths.cleanup_tmp_workspace=true`, the
verified job workspace is deleted after a successful export. A failed run keeps
the temporary workspace for diagnosis.

The configuration can be checked without loading Gemma or ConvoMem:

```bash
python -u scripts/run_cluster_router_validation_hpc.py \
  --config configs/cluster_router_validation_convomem4096_hpc.json \
  --validate-only
```

Collection is streamed per sample. Setting `run.resume=true` validates the
dataset/model/config identity hash and skips completed sample IDs. A failed
sample is written to `errors.jsonl`; clean metric evaluation requires manifest
status `complete` unless `allow_incomplete_state` is explicitly enabled.
The committed 256-example local configuration enables resume. After bounded
GPU-local assignment and block-transaction ingestion were added, the current
4096-token reference smoke reports about 23.9 seconds of session ingestion;
end-to-end scheduler time still depends on baseline forwards, answer length,
GPU type, cache warmth, and Hugging Face downloads, so the runner records actual
per-condition latency instead of embedding a production runtime estimate.

## Offline metrics

The metric command computes:

- answer exact match and token F1;
- normalized quality recovery from local-only toward full-context;
- absolute quality gap and normalized recovery from local-only toward the
  evidence-only upper bound;
- teacher-forced NLL, exact KL from full context, argmax agreement, accuracy;
- evidence record/token/block recall and any/all evidence hit rate;
- teacher attention mass coverage;
- retrieval precision and cluster amplification;
- historical and total-visible layer-token KV ratios;
- mean, maximum, and p95 retrieved tokens per layer;
- KV bytes, attention pairs, CUDA allocated/reserved peaks and latency;
- distance, sequence-length, and evidence-placement breakdowns;
- quality-memory Pareto points and quality at configured KV-ratio thresholds.

Latency is retained as a diagnostic, not treated as the primary objective.
