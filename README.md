# ResidualCache Pre-Research Pipeline

Small pipeline for the pre-research stage in `pre_research_hidden_state_reuse.md`.

It creates controlled prompt variants, collects residual stream states from a
decoder-only Hugging Face model, and compares whether states retrieve the same
target fact better than chance.

## Layout

- `residual_cache/data_process.py`: build controlled JSONL datasets.
- `residual_cache/residual_collect.py`: collect pre-attention residual,
  self-attention `q_proj` output, and block-output states. By default the token
  mask is the last input/prompt token.
- `residual_cache/analysis.py`: cosine retrieval and layer/position reports.
- `residual_cache/runner.py`: one-command pipeline runner.
- `learnable_index/`: standalone learned block-attention index contracts,
  router, trainer, metrics, and offline smoke CLI. It does not import the
  existing `residual_cache` package.
- `block_probability_router/`: independent positive two-tower successor that
  reuses only the aligned collection contract. It learns normalized
  next-block memory probabilities with an explicit `q dot sum(keys)`
  denominator and supports probability-threshold retrieval without changing
  the legacy router.
- `cluster_router_bridge/`: independent integration which splits blocks,
  transports learned keys into memory-owned record metadata, invokes the
  memory-owned per-leaf router-vMF index, and packs selected clusters into
  variable-length per-layer K/V views. Neither existing package imports this
  bridge.
- `cluster_router_validation/`: independent state-collection runner and offline
  metrics for full-context, local-only, recent, learned-router, and oracle-cluster
  comparisons. Dataset and model implementations are injected through factory
  interfaces, while the intermediate JSONL schema keeps expensive replay
  separate from metric iteration. Its config-driven HPC wrapper can keep all
  collection state in node-local temporary storage and persist only metrics.
- `cluster_router_experiment/`: concrete Gemma 4/ConvoMem composition. It runs a
  512-token retained cache that grows to 576 while one new 64-token block is
  evaluated. It prepares the learned block key, unloads the oldest block, and
  writes that block as one pre-commit transaction.
- `residual_cache/gpu_local_cluster_memory.py`: dynamically growing GPU K/V
  records and GPU vMF sufficient statistics. Cluster and record tensor storage
  grows on demand; no global byte or cluster-count cap evicts data. During
  ingestion, CPU locality lookup returns
  at most `candidate_capacity` slot IDs; exact posterior assignment and every
  numerical update are executed only on those GPU slots. Every record has its
  own posterior, but records from one unloaded block share the same pre-commit
  state and commit together. No ingestion path scans every active cluster.
  Eviction is optional and occurs only after a complete cluster recall, using
  per-record attention-usage EMA within that same cluster.
- `residual_cache/gpu_block_cluster_memory.py`: independent block-record memory.
  Each layer-local block is one indivisible record; its learned router block key
  directly drives locality lookup, posterior cluster assignment, and query-time
  cluster scoring, while the original complete-block K/V is replay payload
  only. It shares the vMF classifier functions with the token memory but has
  separate storage, packing, usage, eviction, config, and public entry points.
- `memory_replay_calibration/`: focused full-context vs full-memory-replay
  calibration. It collects post-recall record usage once and sweeps cluster-local
  eviction thresholds without router Top-N selection or repeated ingestion.
- `memory_cluster_calibration/`: static clustering-parameter sweep over one fixed
  GPU K/V capture per sample. Every candidate starts from an empty memory and sees
  identical records in identical order; fact labels remain offline-only. The
  report rejects singleton/oversized allocation degeneracies and emits
  `possible_structural_limit` when repeated sample/layer conditions cannot beat a
  deterministic fact-label permutation baseline.
- `block_memory_cluster_calibration/`: independent static sweep for
  `GpuBlockClusterMemory`. It uses a block-aligned dynamic 512--575 local cache,
  admits only complete 64-token records, reuses one fixed router-key/KV capture
  across every variant, and measures block-level fact separation plus Top-N
  target recall and selected-block ratio. Because K/V is not a classification
  feature, one representative full-attention payload layer exactly represents
  the shared assignment produced for all physical memory layers.
- `learned_block_attention_index.md`: development specification for the
  learned prompt-free block-attention index over historical KV-cache records.

Analysis also writes matplotlib plots to:

```text
<output-dir>/analysis/plots/
```

Current plots:

- `same_target_cosine_margin.png`
- `same_target_distance_margin.png`
- `topK_same_target_recall.png`
- `conflict_false_positive_rate.png`
- `residual_shift_distance.png`
- `same_question_fact_separation_margin.png`
- `topK_query_to_fact_recall.png`
- `query_to_fact_margin.png`

## Quick Run

From this folder, using the parent repository environment:

```powershell
..\venv\Scripts\python.exe -m residual_cache.runner `
  --model-name Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir outputs/qwen25_15b_smoke `
  --max-facts 8 `
  --max-new-tokens 8 `
  --device auto `
  --dtype auto
```

The default collection position is:

```text
--position-family final_prompt
```

This masks the residual readout to the final token of the input prompt. Use
`--position-family both` only when answer-token teacher-forced states are needed.

For an offline check that does not load a model:

```powershell
..\venv\Scripts\python.exe -m pytest tests
```

## LUFY Run

LUFY uses `RuiSumida/LUFY` from Hugging Face `datasets`.

```powershell
..\venv\Scripts\python.exe -m residual_cache.runner `
  --source lufy `
  --model-name meta-llama/Llama-3.2-3B-Instruct `
  --output-dir outputs/lufy_llama32_3b_final_prompt_120qa `
  --max-facts 120 `
  --max-context-turns 24 `
  --max-new-tokens 24 `
  --position-family final_prompt
```

LUFY rows include two suites:

- `lufy_shift`: same fact/question under evidence-only, recent-context, and noisy-context prompts.

## ConvoMem Query-to-fact Run

ConvoMem is used for the natural-data check. The default builder samples real
QA evidence items directly; it does not require identical questions or a fixed
fact set. Each selected item gets:

- `fact_reference`: context stops after the evidence/fact.
- `question_query`: user profile plus the specific natural question, with no
  conversation evidence. This prevents the recall query from seeing the fact it
  is expected to retrieve.

The row also retains a separate `history_prompt` containing evidence plus the
question for answer-block history generation. It is not used as the retrieval
query.

Rows use raw prompts so the captured final token is the evidence/question token,
not a chat-template generation marker. The primary report is
`query_to_fact_report.json`: query states are ranked against fact-reference
states, and a hit means the query retrieves its own fact. Targets include
`pre_attn`, `q`, and `block_output` for residual-vs-q comparison at the same
last prompt position.

For instruction-tuned models, pass `--convomem-chat-template` to apply the
model's chat template to both `fact_reference` and `question_query`. With
`--position-family final_prompt`, the captured anchor is then the final
processed prompt token: the position whose logits predict the first generated
token.

Pass `--convomem-knowledge-prompt` to ask both sides to predict the same
short, single-section retrieval-key format:

```text
FACT: <entity>, <fact>
```

The entity is copied from the user profile. Evidence prompts provide a compact
supported fact; question-only prompts name the fact being requested without
inventing its value. The prompt limits the first line to at most 18 words after
`FACT:`, then requires the model to continue its normal response rather than
stopping. The same instruction is retained for answer-block history prompts,
where the input contains both evidence and the question. Before a full
retrieval run, inspect a small greedy autoregressive smoke test without analysis
metrics:

```powershell
python -m residual_cache.knowledge_prompt_smoke `
  --model-name PATH_TO_MODEL `
  --output-dir outputs/knowledge_prompt_smoke `
  --pair-count 4 `
  --max-new-tokens 64
```

The command saves fact-only, query-only, and evidence-plus-question history
generations. Exact chat prompts, the last prompt token, generated token IDs, and
decoded model outputs are stored in `outputs.jsonl`, plus a readable
`outputs.txt`.

To evaluate the state immediately after the model has actually generated the
shared `FACT:` prefix, use `--position-family generated_prefix`. Collection
greedily generates the prefix, finds the token that completes the colon, and
captures that token's residual state. Entity/fact content after the colon is not
included in the index vector.

## WildChat Long-Conversation Learnable Index Run

`allenai/WildChat-1M` is the preferred HF source for a natural long-dialogue
training run. Its non-toxic release contains 837,989 conversations, with up to
249 user/assistant rounds (498 messages). The preparation stage streams the
single `train` split, makes deterministic 90/5/5 train/validation/test
partitions, keeps conversations with at least 10 rounds, and retains only rows
whose native chat-template encoding reaches 4096 tokens. The resulting records
are contiguous prefixes of real conversations; no synthetic distractors or
random insertion are used.

```powershell
..\venv\Scripts\python.exe -m learnable_index.hpc `
  --config configs/learnable_index_wildchat4096_hpc.json
```

The preparation stage fails explicitly if a split cannot supply its requested
number of exact-length sequences, instead of silently padding or reusing a
conversation.

## Answer Block Cache Run

`answer_block_cache` tests sparse answer-trajectory KV reuse with two independent
KV branches over one shared model:

1. Feed one raw evidence/question pair at a time through the main branch with
   no knowledge instruction and let the model answer naturally.
2. Use monotonically increasing `position_ids` across all history episodes.
3. Clone the current main-history cache into an index-only branch. Give that
   branch the same content with the knowledge instruction appended at the end.
   Let it generate `FACT:` and save the index vector at the colon token. Discard
   the index branch output and cache; neither is appended to main history.
4. Generate the main answer through EOS or `max_new_tokens`. Save at most the
   first `block_size` generated tokens' all-layer K/V, beginning exactly at the
   main model's first output token. Prompt/template suffix tokens are excluded.
   Later main tokens remain in continuous history but not in the saved block.
5. At test time, build the query index through the same index-only instruction
   branch, retrieve top-N main-output blocks, sort them by original position,
   and prepend their sparse K/V to an instruction-free main query generation.
   `--custom-positioned-replay` keeps the current question/generation on its
   own autoregressive trajectory and supplies an explicit 4D causal mask for
   `[retrieved sparse blocks] + [current tokens]` during replay.

```powershell
$env:HF_HOME=(Resolve-Path outputs).Path + "\hf_cache"

..\venv\Scripts\python.exe -m residual_cache.answer_block_cache `
  --model-name google/gemma-4-E4B `
  --output-dir outputs/answer_block_cache_gemma4_24facts `
  --max-facts 24 `
  --block-size 8 `
  --top-n-blocks 8 `
  --max-new-tokens 24 `
  --index-layer 40 `
  --semantic-model BAAI/bge-small-en-v1.5
```

`--semantic-model` is optional. The runner loads it only through Hugging Face
`AutoTokenizer`/`AutoModel`, rejects decoder-only/CausalLM models, and compares
normalized encoder semantic-token embeddings. The default small recommendation
is `BAAI/bge-small-en-v1.5`.

Outputs:

- `history_blocks/*.pt`: per-block all-layer K/V.
- `history_blocks.jsonl`: block positions, tokens, episode metadata, and
  history-payload quality flags.
- `index_vectors.pt`: residual/block-output index vectors for retrieval.
- `test_results.jsonl`: baseline vs sparse-KV replay outputs.
- `generated_outputs.md`: per-episode actual decoded model outputs for direct
  inspection.
- `summary.json`: history-payload validity, hit rate, text-similarity, and
  semantic-similarity summary when `--semantic-model` is provided.

`test_results.jsonl` records both cleaned text and raw decoded text, generated
token ids, generated-token counts, EOS hits, and stop reasons. Generation stops
as soon as an EOS token is generated, otherwise it stops at `--max-new-tokens`.

To reuse existing history blocks and rerun only the retrieval/replay test with a
different `top_n_blocks`, pass `--reuse-history-dir`:

```powershell
..\venv\Scripts\python.exe -m residual_cache.answer_block_cache `
  --model-name google/gemma-4-E4B `
  --output-dir outputs/answer_block_cache_gemma4_24facts_top2_replay_only `
  --reuse-history-dir outputs/answer_block_cache_gemma4_24facts `
  --top-n-blocks 2 `
  --max-new-tokens 128 `
  --index-layer 40 `
  --semantic-model BAAI/bge-small-en-v1.5
```

Add `--custom-positioned-replay` to the replay-only command to test the explicit
autoregressive mask path without regenerating history blocks.

`--filter-fact-line-from-blocks` is intentionally rejected for this pipeline:
main blocks contain natural instruction-free output and never include the
index-only `FACT:` line.

To inspect whether replay attends to the recalled answer bodies, trace the
already saved replay trajectory with eager attention:

```powershell
..\venv\Scripts\python.exe -m residual_cache.attention_analysis `
  --replay-dir outputs/answer_block_run `
  --output-dir outputs/answer_block_attention_smoke `
  --limit 4 `
  --layers all
```

The trace separates query-prompt, generated-so-far, recalled suffix,
gold-body, and distractor-body attention. It teacher-forces the saved replay
tokens and reports eager next-token agreement so backend differences remain
visible.

To isolate absolute position distance with one fixed fact/query pair, regenerate
its history block and recall response across a geometric gap sweep:

```powershell
..\venv\Scripts\python.exe -m residual_cache.distance_sweep `
  --replay-dir outputs/answer_block_run `
  --output-dir outputs/answer_block_distance_case3 `
  --case-index 3
```

The main sweep fixes history at position zero and changes only the query gap.
The joint-shift control moves history and query together at a fixed gap, which
separates relative-distance effects from absolute-position effects.
The runner defaults to eager attention and refuses to generate the gap-zero
condition unless every decoder layer and the continuation logits pass the
continuous-vs-cache equivalence check. Use `--attention-backend default` only
to diagnose backend-specific divergence.

Use `--test-position-override 0` for low-position sanity checks. Without this
override, replay-only tests start from the reused history final position.

```powershell
..\venv\Scripts\python.exe -m residual_cache.runner `
  --source convomem `
  --model-name google/gemma-4-E4B `
  --output-dir outputs/convomem_gemma4_e4b_final_prompt_120facts `
  --max-facts 120 `
  --max-context-turns 48 `
  --max-new-tokens 24 `
  --position-family final_prompt
```

Use `--convomem-root` only if the local ConvoMem cache is not in the default
Hugging Face cache path for this machine.

## Prompt-free User-input Index Run

`prompt_free_index` evaluates retrieval keys that never depend on model output.
It feeds the natural ConvoMem fact/reference input and question/query input,
then restricts every representation to tokens that overlap the original user
content (chat-template role markers and generation suffixes are excluded).

The runner compares three training-free indexes at every selected layer:

- `momentum_dtw`: the sequence of first differences
  `h[t] - h[t-1]`, matched against history with cosine-cost DTW.
- `mean_state_cosine`: mean block-output residual over the same user tokens.
- `mean_q_cosine`: mean self-attention `q_proj` output over the same tokens.

For tractable all-pairs DTW, momentum sequences are linearly resampled to a
fixed number of time points and compressed by a deterministic signed random
projection. This projection is not fitted to the data and uses no labels.
Its point count, width, and seed are saved with the collection. Set
`--projection-dim 0` to disable projection for an exact-width ablation.

```powershell
..\venv\Scripts\python.exe -m residual_cache.prompt_free_index `
  --model-name google/gemma-4-E4B `
  --output-dir outputs/prompt_free_momentum_gemma4_120facts `
  --max-facts 120 `
  --layers all `
  --state-target block_output `
  --trajectory-points 24 `
  --projection-dim 64 `
  --top-k 10
```

Use `--chat-template` when the model requires its normal conversation wrapper;
the wrapper is fed to the model but excluded from pooling and momentum. No
knowledge instruction or generated `FACT:` prefix is used. To analyze selected
layers first, pass a specification such as `--layers 8,16,24-31`.

Collection and analysis are separable, so DTW windows or K can be changed
without loading the model again:

```powershell
..\venv\Scripts\python.exe -m residual_cache.prompt_free_index `
  --reuse-collection outputs/prompt_free_momentum_gemma4_120facts/collect `
  --output-dir outputs/prompt_free_momentum_gemma4_window4 `
  --dtw-window 4 `
  --top-k 10
```

Primary outputs are `analysis/layer_report.jsonl`,
`analysis/neighbors.jsonl`, and `analysis/summary.json`.

## Standalone CAMELoT-Protocol Model Evaluation

`camelot_model_eval` is independent of the residual-index pipeline. It injects
memory directly into every selected Gemma 4 text-attention layer. Each layer
retrieves before attention and writes the current K/V only after attention, so
the first window is exactly the native model and current-window information
cannot leak through external memory.

The default command uses the paper's main causal-language-model settings:
batch size 4, window lengths 256/512/1024 for WikiText-103 or 512/1024/2048
for PG-19 and Pile-ArXiv, 10,000 slots per layer, cosine threshold 0.93,
sequential Transformer-XL-style streams, and token perplexity. The backbone is
intentionally changed from the paper's LLaMA2-7B to the cached local Gemma 4
E4B base model.

Because the paper describes retrieved K/V as a past cache, memory methods use
cache-relative RoPE positions by default: the first window uses positions
`0..L-1`, while later windows use `L..2L-1`. The base method resets each
independent window. `--position-policy continuous` and
`--position-policy window_reset` are available as explicit sensitivity
ablations, and the chosen policy is written to the run manifest.

```powershell
..\venv\Scripts\python.exe -m residual_cache.camelot_model_eval `
  --dataset-preset wikitext103 `
  --hf-home E:\huggingface `
  --output-dir outputs/camelot_protocol_gemma4_wikitext103
```

The compared methods are:

- `base`: no external memory;
- `camelot`: one count-weighted average K/V payload per hard-threshold slot;
- `vmf_records`: vMF posterior slot assignment, with every write retained as
  an original token-level K/V record and exact K reranking inside routed slots.

This round deliberately has no block-level index. The vMF model adapter also
does not enable the standalone oracle's split/merge hierarchy yet; it isolates
the effect of probabilistic writes and original-record payloads first.

For a small end-to-end check before a corpus run:

```powershell
..\venv\Scripts\python.exe -m residual_cache.camelot_model_eval `
  --dataset-preset wikitext103_raw `
  --dataset-path tests/fixtures/camelot_smoke.txt `
  --methods base,camelot,vmf_records `
  --window-sizes 16 `
  --batch-size 1 `
  --max-tokens 64 `
  --slot-capacity 32 `
  --record-capacity 32 `
  --augmented-layers 0 `
  --output-dir outputs/camelot_protocol_gemma4_smoke
```

The run writes aggregate results and protocol metadata to `summary.json` and
per-window loss, perplexity, and timing to `windows.jsonl`. Memory byte counts
are included because equal entry counts do not imply equal storage: the vMF
variant retains both routing statistics and original records.

Progress bars are enabled by default for model loading, corpus tokenization,
stream batchification, the evaluation matrix, every sequential window, and
result writing. Use `--no-progress` for non-interactive jobs or clean log
files; `--progress` explicitly enables them.

For CAMELoT, `--slot-capacity 10000` remains the exact logical per-stream,
per-layer limit, but GPU backing storage is allocated only as novel slots are
created. Read assignments are reused by the following write and familiar
writes targeting the same slot are reduced as one count-weighted update. These
are execution optimizations only: the threshold, averaging rule, and
oldest-slot replacement policy are unchanged. The result manifest records both
the allocated and maximum slot counts.

The vMF implementation likewise allocates routing slots and original K/V
records on demand. Its default `--vmf-write-chunk-size 32` evaluates posterior
assignments for 32 post-attention writes against one pre-commit memory state,
then commits their sufficient statistics and original records together. Chunks
remain sequential and writing still happens only after attention, so current
window information cannot leak into that window's predictions. This removes
the arbitrary ordering among tokens inside a mini-batch and makes the baseline
practical on GPU, but it is a different estimator from fully online posterior
updates. Append `--vmf-write-chunk-size 1` to an evaluation command for the
strict sensitivity baseline when exact token order is required.

The chosen chunk size and its definition are saved in `summary.json`; research
comparisons should report it and preferably include a small chunk-size
sensitivity sweep.
