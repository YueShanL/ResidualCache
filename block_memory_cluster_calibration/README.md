# Router-key block-memory cluster calibration

This package calibrates the independent `GpuBlockClusterMemory`. It is not a
mode of the token-record calibration runner.

For each dynamic random-position ConvoMem sample, the model performs one
block-aligned rolling pass. The local cache is allowed to vary from 512 to 575
tokens; only a complete 64-token mechanical block can enter historical memory.
The learned router key and one representative full-attention layer's original
K/V payload are retained on GPU for the duration of that sample. Every static
parameter variant starts from an empty block memory and consumes the identical
fixed block sequence. No synthesized dataset or cache tensor is persisted.

K/V does not participate in block assignment. Since each physical memory layer
receives the same router key sequence, its cluster assignment is mathematically
identical. The runner therefore executes one representative payload layer and
records every classification-equivalent layer in the report.

## Offline metrics

- complete-block cluster size, tail size, and singleton ratios;
- primary-fact B-cubed precision/recall/F1 and a deterministic permutation
  baseline;
- multi-fact block ambiguity and dominant-overlap diagnostics;
- learned-query Top-1/4/8 target-fact block recall and precision;
- selected-block ratio, which prevents full-memory selection from appearing
  successful merely because it contains the target;
- new/existing assignment rates and bounded locality candidate counts.

Fact labels are derived from exact token overlaps only after assignment. A
block's largest-overlap fact is used by B-cubed, while target recall counts any
block containing target-fact tokens. Labels are never visible to memory.

Validate the full configuration without loading Gemma:

```bash
python -u scripts/run_block_memory_cluster_calibration.py \
  --config configs/block_memory_cluster_calibration_convomem4096.json \
  --validate-only
```

Run the one-sample integration smoke or the eight-sample, 32-variant sweep:

```bash
python -u scripts/run_block_memory_cluster_calibration.py \
  --config configs/block_memory_cluster_calibration_convomem4096_smoke.json

python -u scripts/run_block_memory_cluster_calibration.py \
  --config configs/block_memory_cluster_calibration_convomem4096.json
```

Outputs are limited to `config.resolved.json`, `sample_metrics.jsonl`, and
`metrics.json`.
