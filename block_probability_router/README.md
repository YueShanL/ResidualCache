# Block probability router

This package trains an independent, positive two-tower router from the aligned
datasets produced by `learnable_index`.  It does not add a mode switch to the
legacy cosine-logit router.

For retrieval point `t`, the frozen Gemma collector supplies these states from
one causal, block-aligned streaming pass:

- `q_t`: layer-40 summary of the currently visible restricted history;
- `k_b`: layer-40 summary of each completed historical block `b`;
- `p_teacher(b | M_t)`: mean next-block teacher attention mass, normalized only
  across eligible historical memory blocks `M_t`.

The live block that contains `q_t` is not part of `M_t`. Complete 64-token
blocks are captured only when atomically evicted. The native local cache is
allowed to grow from 512 to 575 tokens; reaching 576 unloads one complete block
and returns it to 512. Candidate construction enforces
`block.end_position <= local_context_start`, so neither the model nor the loss
contains a fallback that masks a leaked current block.

The trained model computes

```text
w_b = phi_q(q_t)^T phi_k(k_b) > 0
S_M = sum_{b in M_t} phi_k(k_b)
Z_M = phi_q(q_t)^T S_M = sum_b w_b
p_hat(b | M_t) = w_b / Z_M
```

`phi_q` and `phi_k` are separate MLPs with a strictly positive softplus output.
The loss is cross entropy against the frozen teacher's conditional block
distribution.  There is no demand head and no student softmax.

At inference, the retrieval parameter is a global missing-mass tolerance
`epsilon`. The router returns the smallest descending-probability prefix whose
total predicted mass reaches `1 - epsilon`:

```text
choose the smallest R such that
sum_{b in R} w_b >= (1 - epsilon) * Z_M
```

Thus `epsilon = 0.02` targets 98% retained probability mass. This global rule
is essential for flat attention: many individually small blocks are retained
when their combined tail is large. A MIPS implementation can expand its result
set or lower a range cutoff until the returned weight sum reaches the global
target; a single fixed per-block cutoff is not a valid substitute.

## Commands

Train from any existing aligned collection:

```bash
python -m block_probability_router train \
  --dataset-dir outputs/.../collection/train/dataset \
  --output-dir outputs/.../training \
  --missing-mass-tolerances 0.01,0.02,0.05,0.1 \
  --device cuda
```

Run the complete ConvoMem synthesis, collection, training, and official
validation/test pipeline from a JSON config:

```bash
python scripts/run_block_probability_router_hpc.py \
  --config configs/block_probability_router_convomem4096_hpc.json
```

Evaluate an existing checkpoint without entering the training pipeline:

```bash
python scripts/run_block_probability_router_evaluation_hpc.py \
  --config configs/block_probability_router_evaluation_convomem4096_hpc.json
```

The committed evaluation config is the full official 290-example ConvoMem test
run.  After attention/retrieval diagnostics it performs greedy autoregressive
QA without teacher forcing under these conditions:

- `full_context`: the complete prompt prefix from the synthesized 4096-token
  sequence;
- `evidence_only`: the correct memory conversation plus the final question;
- `local_only`: the native restricted local window;
- `router_epsilon_<epsilon>`: the same local window augmented with the blocks
  selected by each cumulative missing-mass tolerance.

The fixed-point QA evaluation first performs one full-context prefill through
the retrieval point. It then cuts the query summary, every candidate block
summary, candidate block K/V, and the native local K/V suffix from that same
causal trajectory. Router selection is therefore the only intervention at the
fixed retrieval point: the prefix state is not degraded by pretending that no
earlier retrieval occurred. Selected blocks retain their original logical
positions and augment only physical full-attention layers. Sliding-attention
layers keep the native block-aligned 512-to-575-token local window after the
cut during autoregressive generation.

Training and persisted aligned collections retain their independent streaming
protocol. QA reports the collection, checkpoint, and full-context posthoc-cut
protocols separately and marks whether the checkpoint protocol matches the QA
router-state source; legacy and streaming checkpoints remain loadable for
controlled compatibility measurements.
The report includes exact match, token F1, answer containment,
95% bootstrap intervals, paired deltas against all three baselines, evidence
block recall, retained teacher/predicted mass, selected-block fraction, and a
visible layer-token KV ratio. This is an answer-generation test; the gold
answer is never fed back during decoding.

`oracle_replay_smoke` isolates replay from both the learned router and memory.
It synthesizes four 4096-token ConvoMem examples, obtains the full-context
teacher block distribution, selects the smallest block set retaining
`1 - epsilon` historical teacher mass, and autoregressively compares the
resulting KV replay with full-context generation. Replay K/V is sliced directly
from a full-context source cache, eliminating router-state or streaming-ingest
differences. An all-historical replay control must exactly match full-context
generated tokens before compressed replay quality is interpreted. Only JSON
metrics are saved; synthesized examples and KV payloads remain transient.

The independent evaluation config accepts either one number or a sorted list in
`evaluation.epsilon`. `evaluation.max_block` is applied after cumulative-mass
selection: `-1` disables the hard retrieval limit, while a positive integer
caps the number of returned blocks. When the cap prevents the requested
`1 - epsilon` mass from being reached, the output reports the cap application
rate, truncated block count, target success rate, and mass shortfall instead of
silently treating the capped result as successful. The default evaluation
device is CPU because this router is small; the frozen Gemma collection stage
still runs on the configured model device.

With `paths.use_tmp_workspace=true`, synthesized inputs, aligned collections,
per-sample QA predictions, model caches, and event logs stay below
`paths.tmp_workspace_root` and are removed after a successful run. Only the
combined router and aggregate QA `metrics.json` is copied to
`paths.output_root`.

The example deliberately preserves the previous experiment's controlled
variables: Gemma-4 E4B IT, layer-40 query/key summaries, teacher layers
29/35/41, 4096-token randomized ConvoMem synthesis, 64-token blocks, a
64-token maximum next-block label horizon, 4096 train documents, 10% internal
validation, and the same
optimizer/regularization values.  With the temporary workspace enabled, only
the best checkpoint, training metrics/summary, and official validation/test
metrics are copied back; synthesized inputs, collections, caches, and database
state remain under the configurable `/tmp` workspace and are removed.
