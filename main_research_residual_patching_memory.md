# Main Research: Residual Patching for Memory Recall

## Entry Condition

Run this study only if the pre-research finds reusable hidden-state structure under at least one controlled setting.

If hidden states only cluster by task format and not by target fact, restrict this study to procedural memory. Do not claim factual memory recall.

## Goal

Use stored residual or attention-output deltas from past contexts as an internal memory payload, then patch them into a future forward pass when the current state matches a stored memory state.

The target is not exact full-attention equivalence. The target is useful memory recall through activation reuse.

## Memory Record

Each stored memory entry should contain:

```text
index_state:
  layer
  token_position_type
  normalized residual vector
  optional canonical task key
  optional fact key

payload:
  layer
  residual delta or attention output delta
  optional logZ/weight metadata

metadata:
  source prompt id
  target fact id
  task id
  timestamp/order
  answer correctness
  conflict policy
```

Start with residual deltas:

```text
delta_l = x_after_attention_l - x_before_attention_l
```

Then compare with:

```text
post_attn residual
post_mlp residual
full layer delta = x_{l+1} - x_l
```

## Retrieval Key

Use the best layer/position found in pre-research.

Compare three keys:

- raw residual
- layer-normalized residual
- canonical task/fact key plus residual similarity

Keep retrieval simple:

```text
score = cosine(current_state, memory_state)
```

Add recency only after the no-recency baseline is measured.

## Patching Timing

Test four patching sites:

### 1. Pre-Attention

Patch before self-attention:

```text
x_l = x_l + alpha * delta_memory
```

Use when the memory should influence what the model attends to next.

Risk: can distort Q/K/V generation.

### 2. Post-Attention

Patch after attention output and before MLP:

```text
x_l = x_l + alpha * delta_memory
```

Use when the memory payload is attention-like.

This is the safest first site.

### 3. Post-MLP

Patch after MLP:

```text
x_{l+1} = x_{l+1} + alpha * delta_memory
```

Use when memory behaves like semantic steering.

Risk: may affect final answer without improving retrieval.

### 4. Answer-Only Patch

Patch only at the final question token or first generated token.

Use as the smallest diagnostic. If this fails, high-frequency patching is unlikely to help.

## Weight Computation

Start with fixed alpha:

```text
alpha in {0.05, 0.1, 0.2, 0.4, 0.8}
```

Then test similarity weighting:

```text
alpha = base_alpha * max(0, cosine_similarity)
```

Then add time decay:

```text
alpha = base_alpha * sim * exp(-lambda * age)
```

If storing attention-style summaries, optionally store:

```text
logZ_old
```

This only matters when attempting softmax-compatible merging. It is not required for residual steering.

## Multiple Memory Merge

For top-k memory entries:

```text
patched_delta = sum_i softmax(beta * score_i) * delta_i
```

Start with:

```text
k = 1
```

Only test larger k after top-1 works. Averaging wrong memories will hide failure modes.

## Patching Frequency

Test from least invasive to most invasive:

1. answer-only patch
2. once at question-final token
3. every generated token for first N tokens
4. every token after a recall marker
5. every layer/token where similarity crosses threshold

Prefer sparse patching. Dense patching makes attribution harder.

## Patching Granularity

Compare:

- single layer
- small layer band, for example layers 12-16
- late layers only
- attention heads if accessible
- token span patching instead of single token patching

Do not start with per-head patching unless layer-level patching shows signal.

## Prompt and Task Protocol

Run two versions:

### Unstructured Prompt

Natural user phrasing.

### Canonical Memory Protocol

Use a fixed read/write format:

```text
Memory write:
Task: store_fact
Entity: Alice
Attribute: color
Value: blue

Memory read:
Task: recall_fact
Entity: Alice
Attribute: color
Policy: latest
```

The protocol should increase state reuse. If it does not, residual patching is likely too unstable.

## Evaluation Tasks

Use small controlled tasks first:

- exact fact recall
- latest-fact recall under conflict
- entity-attribute lookup
- paraphrased question recall
- multi-hop recall
- negative control with absent facts

Then test:

- long conversation preference memory
- code variable/function recall
- instruction persistence
- agent procedure reuse

## Metrics

Primary:

- exact match accuracy
- conflict resolution accuracy
- absent-fact abstention rate
- wrong-memory injection rate

Secondary:

- KL divergence from unpatched logits
- target-token logit lift
- answer latency
- memory hit rate
- sensitivity to alpha

Report negative controls. A memory method that always increases confidence is not useful.

## Model Differences

Compare at least:

- small vs medium model
- Qwen-style vs Llama-style architecture
- GQA vs non-GQA if available
- RoPE variants
- instruction-tuned vs base model

Expected differences:

- instruction-tuned models may benefit more from canonical protocol
- larger models may have more stable task states
- smaller models may be easier to steer but more prone to false positives
- different layer bands may carry memory-relevant states

## Main Ablations

Run these before adding complexity:

- no patch
- random memory patch
- wrong-target memory patch
- same-task wrong-fact patch
- same-fact different-task patch
- correct memory patch
- correct memory with shuffled layer
- correct memory with shuffled token position

A valid method should improve only the correct-memory condition.

## Success Criteria

Claim procedural memory if:

- task success improves under same procedure
- wrong-fact factual recall does not increase
- negative controls stay stable

Claim factual memory only if:

- exact recall improves on held-out paraphrases
- conflict handling respects latest/authority policy
- wrong-memory injection remains low
- patching works across noise and prompt variation

Minimal factual-memory bar:

```text
patched exact recall improves >= 15 points over no-patch
wrong-memory injection <= 5%
absent-fact false recall <= 5%
```

## Failure Modes

Watch for:

- style steering mistaken for memory
- stale fact recall under conflict
- patching boosts confidence but not correctness
- memory entries cluster by task but not target
- high alpha causes generic answer drift
- larger k averages incompatible memories

## Research Outcome

Possible outcomes:

- strong: residual patching supports factual recall under canonical protocol
- partial: residual patching supports procedural/task memory only
- weak: residual patching only changes style/confidence
- negative: hidden-state reuse is too unstable for memory recall

Do not call it infinite context unless it can recall exact old facts under conflict and absent-fact controls.

