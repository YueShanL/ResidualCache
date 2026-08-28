# Block probability router

This package trains an independent, positive two-tower router from the aligned
datasets produced by `learnable_index`.  It does not add a mode switch to the
legacy cosine-logit router.

For retrieval point `t`, the frozen Gemma collector supplies:

- `q_t`: layer-40 summary of the currently visible restricted history;
- `k_b`: layer-40 summary of each completed historical block `b`;
- `p_teacher(b | M_t)`: mean next-block teacher attention mass, normalized only
  across eligible historical memory blocks `M_t`.

The live block that contains `q_t` is not part of `M_t`.  Candidate construction
already enforces `block.end_position <= local_context_start`, so neither the
model nor the loss contains a fallback that masks a leaked current block.

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

At inference, probability-threshold retrieval is an exact range condition:

```text
p_hat(b | M_t) > tau  <=>  w_b > tau * Z_M
```

This leaves candidate search free to use a MIPS/range index while preserving
the denominator needed for calibrated probabilities.

## Commands

Train from any existing aligned collection:

```bash
python -m block_probability_router train \
  --dataset-dir outputs/.../collection/train/dataset \
  --output-dir outputs/.../training \
  --device cuda
```

Run the complete ConvoMem synthesis, collection, training, and official
validation/test pipeline from a JSON config:

```bash
python scripts/run_block_probability_router_hpc.py \
  --config configs/block_probability_router_convomem4096_hpc.json
```

The example deliberately preserves the previous experiment's controlled
variables: Gemma-4 E4B IT, layer-40 query/key summaries, teacher layers
29/35/41, 4096-token randomized ConvoMem synthesis, 64-token blocks, a
64-token maximum next-block label horizon, 4096 train documents, 10% internal
validation, and the same
optimizer/regularization values.  With the temporary workspace enabled, only
the best checkpoint, training metrics/summary, and official validation/test
metrics are copied back; synthesized inputs, collections, caches, and database
state remain under the configurable `/tmp` workspace and are removed.
