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

After training, reuse an official-test collection to compare fixed retrieval
budgets without retraining:

```powershell
..\venv\Scripts\python.exe -m learnable_index topn-sweep `
  --model-name google/gemma-4-E4B-it `
  --model-device cuda `
  --dtype bfloat16 `
  --collection-dir outputs/learnable_index_topn_validation/collection/test `
  --checkpoint outputs/learnable_index_wikitext103_4096docs_query_key_v1/best.pt `
  --output-dir outputs/learnable_index_topn_validation/sweep `
  --budgets 1,2,4,8 `
  --router-device cuda
```

The sweep evaluates full context and local-256 once per sample, then compares
learned, recent, deterministic-random, and oracle selection at each budget. It
writes per-sample paired results, bootstrap confidence intervals, exact visible
KV bytes, attention pairs, and CUDA allocated/reserved peak measurements.

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

## Controlled long-distance ConvoMem sequences

WikiText preserves natural document order, so relevance and recency are
correlated. The ConvoMem preparer creates a controlled sequence instead:

```text
target evidence -> unrelated ConvoMem conversations -> final question
-> generation bridge -> gold answer
```

The output has an exact token length. Distractors come from other examples in
the same source-file-grouped split and are rejected if they contain the target
answer. Each row records evidence, distractor, question, and answer token
ranges plus the exact evidence-to-answer distance. Train/validation/test use a
deterministic 80/10/10 hash split grouped by source file/profile.

```powershell
python -m learnable_index.prepare_convomem `
  --dataset-name Salesforce/ConvoMem `
  --tokenizer google/gemma-4-E4B-it `
  --output outputs/convomem_long/test_64x4096.jsonl `
  --split test --sequence-length 4096 --sequences 64 `
  --maximum-answer-tokens 64 --maximum-future-horizon 16

python -m learnable_index collect `
  --model-name google/gemma-4-E4B-it `
  --model-device cuda --dtype bfloat16 `
  --input-jsonl outputs/convomem_long/test_64x4096.jsonl `
  --output-dir outputs/convomem_long/collection/test `
  --local-context-length 256 --block-size 64 `
  --future-horizon 16 --retrieval-interval 128 `
  --retrieval-point-policy metadata `
  --minimum-candidate-blocks 2 `
  --teacher-prefill-chunk-size 256 `
  --residual-layer 40 --query-summary mean --query-summary-length 16 `
  --teacher-layers 29,35,41 --teacher-heads all
```

Chunked teacher prefill captures attention only for the answer horizon, so the
transient attention tensor scales with `horizon * context_length` instead of a
full eager attention square. The staged length sweep should be 2K, 4K, 8K,
then 16K while keeping target examples, split hash, local window, block size,
teacher layers, and checkpoint fixed. Report exact distance and candidate
count, not only nominal sequence length. Full autoregressive answer EM/F1 is
the next evaluation layer; teacher-forced NLL remains a diagnostic.

The first 4K ConvoMem training run is defined by
`configs/learnable_index_convomem4096_hpc.json`: 4096 training sequences, a
profile-grouped 10% internal holdout, 295 independent validation sequences,
and 290 independent test/replay sequences. It preserves the WikiText router,
optimizer, regularization, epoch, and early-stopping settings. Candidate blocks
are unbounded so early evidence remains eligible.

This config uses `"persist_prepared_inputs": false`. The runner processes each
split as an ephemeral dynamic stage:

```text
synthesize split JSONL -> collect aligned samples -> delete JSONL + manifest
```

Train and validation omit KV payload persistence because query-key training and
evaluation need only summaries and labels. Test retains KV payloads for the
complete 290-sample replay. With node-local temporary mode enabled, successful
completion exports only `best.pt`, `metrics.jsonl`, and `summary.json`, then
removes the remaining task workspace. Run the fixed task script with the
ConvoMem config as its argument:

```bash
sbatch scripts/submit_learnable_index_hpc.slurm \
  configs/learnable_index_convomem4096_hpc.json
```

## Current research boundary

- Collection is teacher-forced; it does not synthesize autoregressive training
  trajectories.
- Retrieval points follow a fixed mechanical interval for natural documents,
  or explicit answer-aligned input metadata for controlled QA. Fixed Top-N and
  manual query-key score thresholds do not reschedule the next retrieval point.
- Student collection uses local-only history without recurrently exposing
  previously retrieved blocks. Replay does expose selected blocks to future
  forwards and records this policy explicitly.
- The model adapter currently supports Transformers 5.12-compatible Gemma 4
  text backbones. Other model families must add and validate their own cache,
  position, and mask adapter.
