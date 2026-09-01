# Output-preserving Gaussian region router

This package is an independent experiment entry. It reuses the established
ConvoMem preparation, block-aligned streaming capture, logical-position, and
Gemma 4 replay infrastructure, but it does not add a mode or fallback to
`learnable_index` or `block_probability_router`.

## System definition

For a restricted-stream query residual `q` and each completed historical block
residual `k_i`, the router learns

```text
mu_q, sigma_q = f_q(q), softplus(g_q(q)) + sigma_min
z_i             = f_k(k_i)
D_i^2           = mean_j(((z_ij - mu_qj) / sigma_qj)^2) / radius^2
g_i             = sigmoid((1 - D_i^2) / temperature)
```

The mean over feature axes makes `radius` an RMS standardized radius rather
than a feature-dimension-dependent raw chi-square radius. Hard retrieval uses
exactly `D_i^2 <= 1`; there is no Top-N, cumulative attention mass, ANN
fallback, or memory-size boundary in this package.

Query and block residuals come from the same streaming inference trajectory.
The current live window remains native Gemma sliding context. Completed block
K/V is replayed only into physical full-attention layers; sliding-attention
layers are never augmented.

## Training objective

The language model is frozen. For every retrieval example the runner computes:

1. `L_full`: future logits from full context;
2. `L_all`: future logits from the same streaming K/V trajectory with every
   historical block physically replayed;
3. `L_gate`: future logits from that trajectory with differentiable block
   gates added to historical attention logits as `log(g_i)`.

The optimization target is

```text
KL_gate   = KL(softmax(L_full) || softmax(L_gate))
KL_floor  = KL(softmax(L_full) || softmax(L_all))
KL_excess = KL_gate - KL_floor

loss = lambda_preserve * max(0, KL_excess - delta)
     + lambda_sparse   * sum_i(g_i)
     + lambda_entropy  * mean_i(H(g_i))
```

`KL_floor` is not a substitute teacher. It records the irreducible difference
between full-context K/V and K/V produced by the real streaming trajectory, so
the router is penalized only for degradation beyond an all-history replay that
it can actually realize. Gradients still originate from the full-context
future logits and flow through frozen attention into the query/key router.
Teacher attention probabilities are not collected or used.

## Evaluation contracts

`evaluate` performs two distinct checks:

- teacher-forced hard replay: blocks satisfying `D_i^2 <= 1` are physically
  packed, then future-logit KL, excess KL, Top-1 agreement, and compression are
  measured;
- greedy autoregressive QA: `full_context`, `evidence_only`, `local_only`,
  `all_history_upper_bound`, and `region_router` run through the same generation
  infrastructure. The region path contains no gold-answer teacher forcing and
  no teacher-attention selection.

Checkpoints embed model, residual, streaming-window, block, future-horizon,
router, and training-protocol contracts. Evaluation rejects a different model
fingerprint or checkpoint kind.

## HPC smoke run

The committed config is deliberately a 64/16/16 ConvoMem smoke test at 4096
tokens. It is not a full training config:

```bash
python -u scripts/run_output_preserving_region_router_hpc.py \
  --config configs/output_preserving_region_router_convomem4096_smoke_hpc.json
```

With temporary mode enabled, synthesized JSONL, model/dataset cache, transient
K/V, and per-sample QA rows remain under the configurable temporary root and
are deleted after verified export. The persistent output contains only the two
checkpoints, training metrics/config/summary, and aggregate validation/test
hard-replay and QA metrics.

Validate the config without loading a model:

```bash
python -u scripts/run_output_preserving_region_router_hpc.py \
  --config configs/output_preserving_region_router_convomem4096_smoke_hpc.json \
  --validate-only
```

After the smoke run validates memory use and the loss trajectory, the full
configuration keeps the same architecture, optimizer, regularization, and
4096-token synthesis policy while expanding to 4096 training, 295 official
validation, and 290 official test examples. It uses an internal 10% split,
up to 10 epochs, and early-stopping patience 2:

```bash
python -u scripts/run_output_preserving_region_router_hpc.py \
  --config configs/output_preserving_region_router_convomem4096_full_hpc.json
```
