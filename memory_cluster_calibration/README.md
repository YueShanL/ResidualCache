# Fixed-cache memory cluster calibration

This runner calibrates the existing `GpuLocalClusterMemory` assignment parameters.
It does not modify the memory structure and it does not tune parameters online.

For each ConvoMem sample it performs one 512-to-576 rolling model pass, keeps the
evicted full-attention-layer K/V tensors on GPU, and then replays that exact fixed
cache into a new empty memory for every parameter-grid variant. Variant execution
is sequential, so the model and one fixed cache remain resident while only one
candidate memory exists at a time. Fact IDs are attached only to offline metric
records; neither posterior assignment nor router ranking can read them.

The calibration jointly checks:

- cluster allocation shape: mean/tail cluster size and singleton ratios;
- fact separation: B-cubed precision, recall and F1;
- a deterministic fact-label permutation baseline for the same cluster layout;
- learned-router Top-1/4/8 target-fact recall and precision;
- created-vs-existing assignment rates and locality candidate counts.

`selection.status=possible_structural_limit` is emitted only when the configured
minimum sample count has been reached and no grid variant separates facts
consistently above its per-condition permutation baseline. A variant that
separates facts but still degenerates into singleton or oversized clusters is
reported as `grid_did_not_meet_allocation_constraints` instead.

Run a one-sample execution smoke test:

```bash
python -u scripts/run_memory_cluster_calibration.py \
  --config configs/memory_cluster_calibration_convomem4096_smoke.json
```

Run the first 30-variant static sweep over eight independently synthesized
4096-token samples:

```bash
python -u scripts/run_memory_cluster_calibration.py \
  --config configs/memory_cluster_calibration_convomem4096.json
```

Only `config.resolved.json`, `sample_metrics.jsonl`, and `metrics.json` are
written. The synthesized dataset, fixed K/V captures, and candidate memories are
never persisted.
