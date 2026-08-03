# Pre-Research: Hidden-State Reuse Under Context Variation

## Purpose

Test whether a selected pretrained decoder-only model enters reusable internal states when solving the same or related memory task under different context conditions.

This stage does not inject memory. It only records activations and measures whether hidden/residual states are stable enough to justify residual patching later.

## Core Question

Can the model's internal state for a target fact or task be recognized across different prompts, noise, and task phrasings?

If yes, hidden-state reuse may be viable. If no, residual patching is likely to become brittle state steering rather than memory recall.

## Selected Model

Start with one small open model before scaling:

- Primary: Qwen2.5-1.5B-Instruct or Llama-3.2-1B-Instruct
- Secondary: Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct

Use deterministic decoding:

- temperature = 0
- top_p = 1
- max_new_tokens fixed per task
- same chat template for every run

## Activation Targets

Record residual stream states at:

- pre-attention residual
- post-attention residual
- post-MLP residual
- final token before answer
- answer-token positions

Record across all layers first. Later reduce to candidate layers.

For each activation, store:

- layer id
- token position
- prompt id
- task id
- target fact id
- condition id
- hidden vector
- model answer
- correctness label

## Experimental Conditions

### 1. Same Question and Same Task With Context Noise

Goal: test whether irrelevant context changes hidden state.

Example structure:

```text
Noise block A/B/C
Fact: The access code is R7Q-19.
Question: What is the access code?
```

Variants:

- no noise
- short irrelevant text
- long irrelevant text
- adversarial distractors with similar keys
- shuffled noise placement before/after fact

Measure whether states around the question and answer remain close for the same target fact.

### 2. Same Task With Useful Information Present or Absent

Goal: separate task-mode similarity from fact-availability similarity.

Examples:

```text
Present: The access code is R7Q-19. What is the access code?
Absent: No access code is given. What is the access code?
```

Expected finding:

- task-mode states may be similar
- answer/fact states should diverge

If they do not diverge, the representation may be too coarse for memory recall.

### 3. Same Task With Different Contexts

Goal: test whether the model keeps a stable retrieval state while facts change.

Examples:

```text
Fact: Alice's color is blue. Question: What is Alice's color?
Fact: Alice's color is green. Question: What is Alice's color?
```

Compare:

- same task, same entity, different value
- same task, different entity, same value
- same task, different entity, different value

Useful signal:

- task state clusters by task
- fact state separates by entity/value

### 4. Different Task With Same Target

Goal: test whether the same fact can be reached through different reasoning paths.

Examples:

```text
Direct recall: What is Alice's color?
Verification: Is Alice's color blue?
Comparison: Who has the blue color?
Transformation: Return Alice's color in uppercase.
```

Useful signal:

- target-fact states converge near answer formation
- task states differ earlier

This is the key condition for cross-task residual reuse.

### 5. Different Task and Different Target With Context Conflict

Goal: test false-positive risk.

Examples:

```text
Earlier: Alice's color is blue.
Later: Alice's color is green.
Question: What is Alice's latest color?
```

Conflict variants:

- earlier vs later facts
- user instruction vs document fact
- negated facts
- distractor facts with same attribute

Measure whether states encode recency, authority, and target identity.

## Prompt Normalization Study

Run each condition in two modes:

### Natural Mode

Use varied user phrasing.

### Canonical Mode

Rewrite into a fixed protocol:

```text
Task: recall_fact
Entity: Alice
Attribute: color
Policy: latest
Question: Return the value only.
```

Compare whether canonical prompts increase hidden-state clustering for the same target and reduce false matches.

## Similarity Metrics

Use simple metrics first:

- cosine similarity
- centered cosine similarity
- Euclidean distance after layer-wise normalization
- CKA across groups

For retrieval-style evaluation:

- top-1 same target accuracy
- top-k same target recall
- false positive rate under conflict
- cluster purity by task id
- cluster purity by target fact id

Do not start with learned probes unless simple metrics fail but visual inspection suggests structure exists.

## Analysis Views

For each layer and token position, plot:

- same target vs different target similarity
- same task vs different task similarity
- noise length vs similarity decay
- conflict vs non-conflict similarity
- correct-answer vs wrong-answer state separation

Use PCA/UMAP only as visualization, not as proof.

## Success Criteria

Proceed to the main residual patching study only if at least one layer/position family shows:

- same target top-k recall clearly above random
- conflict false positives are bounded
- canonical prompting improves or preserves target clustering
- correct answers have separable states from wrong answers

Minimal threshold for moving forward:

```text
same-target top-10 recall >= 80%
conflict false-positive rate <= 10%
```

These thresholds are placeholders. Tighten them after seeing baseline variance.

## Failure Criteria

Stop or redesign if:

- states cluster only by surface prompt, not target fact
- states cluster only by task mode, not fact identity
- conflict cases retrieve stale or wrong facts
- canonical prompting collapses different targets together
- high similarity does not predict answer correctness

## Output Artifact

Produce:

- activation dataset metadata
- layer-by-layer similarity report
- retrieval confusion matrix
- short recommendation:
  - viable
  - viable only with canonical protocol
  - viable only for procedural/task memory
  - not viable for factual recall

