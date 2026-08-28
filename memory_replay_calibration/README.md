# Memory replay calibration

This runner isolates memory retention from router Top-N selection. For each
dynamic 4096-token ConvoMem example it executes:

1. full-context teacher-forced reference;
2. rolling ingestion with every historical record retained;
3. full-memory replay while collecting mean historical attention probability;
4. counterfactual cluster-local eviction plans for every configured threshold;
5. full replay of every record retained by each plan.

The same ingestion and usage pass is shared by all thresholds. The report keeps
both `compressed vs full context` metrics (end-to-end target) and `compressed vs
uncompressed replay` metrics (the isolated effect of eviction). A setting is
accepted only if uncompressed replay remains close to full-context answer F1 and
the compressed replay satisfies the configured KL, argmax-agreement, and F1
constraints.

Run the one-sample real-model smoke:

```powershell
..\venv\Scripts\python.exe -u -m memory_replay_calibration `
  --config configs/memory_replay_calibration_convomem4096_smoke.json
```

Use `configs/memory_replay_calibration_convomem4096_window512.json` for the
eight-sample 512-token-window sweep with full-memory feedback during ingestion.
Outputs are `sample_metrics.jsonl`, `metrics.json`, and `config.resolved.json`.

## Current calibrated profile

The eight-sample ConvoMem 4096-token calibration in
`outputs/memory_replay_calibration_convomem4096_window512_full_feedback_v1`
selected a usage threshold of `1e-3`. Its aggregate retained-record ratio was
`0.5841`; answer token F1 was `0.7480` versus `0.7417` for full context, and
mean KL from uncompressed replay was `0.00303`. Treat this as an initial
profile, not a final statistical estimate: deployment confidence still needs a
larger held-out sweep and repeated-recall evaluation for the usage EMA.

The corresponding 256-token-window run is stored in
`outputs/memory_replay_calibration_convomem4096_window256_full_feedback_v1`.
Its uncompressed replay preserved answer F1 (`0.7364` versus `0.7352`) but did
not meet full-context distribution fidelity: mean KL was `0.0402` and mean
argmax agreement was `0.9698`. Under the strict replay-baseline gates, its
status is `uncompressed_replay_not_equivalent` and no eviction threshold is
selected. This failure precedes eviction and therefore must be resolved at the
streaming-state/replay layer before calibrating 256-token retention.
