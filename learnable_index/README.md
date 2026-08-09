# Learnable Block-Attention Index

This directory is the standalone implementation of
`../learned_block_attention_index.md`. It deliberately has no imports from the
existing `residual_cache` package. The frozen Gemma 4 adapter, aligned
teacher/student collection, router training, KV block store, retrieval policy,
and restricted replay are all contained in this package.

## Implemented boundary

The pipeline provides:

- a versioned retrieval-sample contract with logical positions and explicit
  `student_restricted` provenance for router inputs;
- configurable teacher attention aggregation from
  `[layer, head, future query, key]` tensors;
- preservation of both absolute historical attention mass and the conditional
  distribution across candidate blocks, including per-future-distance mass;
- one-time tokenization and aligned logical positions shared by teacher and
  student branches;
- eager full-context Gemma 4 teacher attention collection;
- strict local-window student query/block residual and complete physical KV
  collection, including Gemma 4 shared-KV metadata;
- a versioned on-disk KV block store with model/config fingerprint checks;
- separate query/key MLP towers trained only with soft conditional
  cross-entropy;
- variable candidate counts through masked batching;
- prediction KL/cross-entropy, top-1 recall, predicted/oracle coverage, and
  entropy;
- checkpointed training/evaluation with a JSON run manifest;
- fixed Top-N or an explicit query-key probability-threshold policy;
- schedule-correct teacher-forced replay for full context, local only,
  predicted, oracle, and recent-block conditions;
- deterministic synthetic data for an offline end-to-end smoke test.

The language model remains frozen by construction. Teacher attention crosses
the branch boundary only as labels; teacher residuals never enter router
features. Router training is the only autograd-enabled model update.

The current model deliberately has no learned retrieval-demand head. Absolute
historical attention mass remains in the dataset contract so that conditional
teacher distributions are well-defined, but it is not a prediction target.

## Dataset layout

Each dataset directory contains:

```text
manifest.json
samples.pt
```

Each sample includes the fields required by the design document, including the
first future position that can observe a retrieval decision.  Historical block
ranges must end before the current local window starts.  Query and block
summaries are rejected unless their provenance is `student_restricted`.

A real collection directory additionally contains:

```text
collection_manifest.json
sequences.pt
plans.jsonl
dataset/
kv_store/manifest.json
kv_store/blocks/*.pt
```

Validation splits are grouped by `sequence_id`; retrieval points from the same
sequence are never split across train and validation.

## Complete Gemma 4 run

Input is JSONL. Each row must contain a unique `sequence_id` and exactly one of
`token_ids`, `text`, or `messages`. Text/chat rows are tokenized once and the
same IDs are sliced for both branches.

From `ResidualCache`, using the parent repository environment:

```powershell
..\venv\Scripts\python.exe -m learnable_index run `
  --model-name PATH_TO_LOCAL_GEMMA4_SNAPSHOT `
  --input-jsonl data/long_sequences.jsonl `
  --output-dir outputs/learnable_index_gemma4 `
  --local-context-length 256 `
  --block-size 32 `
  --future-horizon 16 `
  --retrieval-interval 32 `
  --teacher-layers 36-41 `
  --residual-layer 40 `
  --query-summary mean `
  --query-summary-length 16 `
  --epochs 20 `
  --top-n 4 `
  --policy fixed `
  --replay-top-n 4
```

The command is local-files-only by default. `--allow-network` is an explicit
opt-in. Teacher attention requires eager attention and the loader rejects other
backends/model families.

For a manually controlled budget, use fixed Top-N. For an explicit rule-based
filter, use `--policy score_threshold --score-threshold P`; this keeps at most
`--replay-top-n` blocks whose query-key softmax probability is at least `P`.
No learned demand value participates in either policy.

The output root contains `collection/`, `training/`, `replay/`, and
`pipeline_manifest.json`. Replay records token predictions, NLL, KL from full
context, full-context argmax agreement, visible KV bytes, latency, and logical
attention query/key pairs.

## Offline smoke run

Run from `ResidualCache` with the parent repository environment:

```powershell
..\venv\Scripts\python.exe -m learnable_index make-smoke-data `
  --output-dir outputs/learnable_index_smoke/data `
  --samples 128 `
  --residual-dim 16

..\venv\Scripts\python.exe -m learnable_index train `
  --dataset-dir outputs/learnable_index_smoke/data `
  --output-dir outputs/learnable_index_smoke/run `
  --projection-dim 32 `
  --hidden-dim 64 `
  --epochs 10 `
  --device cpu

..\venv\Scripts\python.exe -m learnable_index evaluate `
  --dataset-dir outputs/learnable_index_smoke/data `
  --checkpoint outputs/learnable_index_smoke/run/best.pt `
  --output outputs/learnable_index_smoke/evaluation.json `
  --device cpu
```

Training writes `run_config.json`, `metrics.jsonl`, `best.pt`, `final.pt`, and
`summary.json`.  The run config records the information boundary explicitly.

## Slurm/HPC run: WikiText-103, 4096 training documents

The HPC stage is driven by
`configs/learnable_index_wikitext4096_hpc.json`. It prepares 4096 train
articles, yielding 20,480 retrieval samples at five retrieval points per
article. Training uses a sequence-grouped 10% internal holdout (410 articles,
2050 retrieval samples), a maximum of 10 epochs, and early stopping with
patience 2. The independent official WikiText validation/test inputs remain at
59/58 articles (295/290 retrieval samples), and replay covers all 290 official
test samples. The earlier 1024-document JSON remains available as the
single-variable baseline.

The 4096-document config enables node-local temporary storage with:

```json
"paths": {
  "output_root": "outputs/learnable_index_wikitext103_4096docs_query_key_v1",
  "use_tmp_workspace": true,
  "tmp_workspace_root": "/tmp"
}
```

When enabled, prepared inputs, aligned datasets, checkpoints, evaluation,
replay, and Hugging Face/Datasets/Torch caches are placed below the configured
temporary root. Only `training/best.pt`, `training/metrics.jsonl`, and
`training/summary.json` are copied atomically to `output_root` after every
stage succeeds. Change `tmp_workspace_root` to another absolute node-local
scratch path when required. Temporary mode intentionally does not resume data
collection across jobs because node-local storage is ephemeral.

The JSON uses Hugging Face repository IDs (`google/gemma-4-E4B-it` and
`Salesforce/wikitext`) rather than model or dataset filesystem paths.
Transformers and Datasets use their normal Hugging Face cache. Optionally set
the cache root and authentication in the job environment:

```bash
export HF_HOME=/path/to/huggingface/cache
export HF_TOKEN=your_read_token
export PYTHON_BIN=/path/to/venv/bin/python
```

`HF_TOKEN` is needed when the model repository requires accepted terms or
authentication. Validation checks only the JSON contract and does not download
the model or dataset:

```bash
"${PYTHON_BIN}" -m learnable_index.hpc \
  --config configs/learnable_index_wikitext4096_hpc.json \
  --validate-only
```

Submit the fixed Slurm task directly:

```bash
sbatch scripts/submit_learnable_index_hpc.slurm
```

The `#SBATCH` header is fixed in the task script; edit its partition/account or
add cluster-specific `module load` lines there if required. The script directly
starts `scripts/run_learnable_index_hpc.py`; it does not generate or recursively
submit another job. An alternative JSON file may be passed as the first script
argument. Completed stages are validated and skipped on resubmission; a changed
experiment config must use a different `output_root`.

## Current research boundary

- Collection is teacher-forced; it does not synthesize autoregressive training
  trajectories.
- Retrieval points follow a fixed mechanical interval. Fixed Top-N and manual
  query-key score thresholds do not reschedule the next retrieval point.
- Student collection uses local-only history without recurrently exposing
  previously retrieved blocks. Replay does expose selected blocks to future
  forwards and records this policy explicitly.
- The model adapter currently supports Transformers 5.12-compatible Gemma 4
  text backbones. Other model families must add and validate their own cache,
  position, and mask adapter.
