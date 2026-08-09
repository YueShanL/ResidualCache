# Learned Prompt-Free Block-Attention Index for Residual Cache

## Status

This document is the development specification for a learned, prompt-free
block index over historical KV-cache records. It records the design decisions
made before implementation so later code and experiments use the same
definitions.

The first version is a research prototype. Its goal is to determine whether a
small learned router can predict how a frozen language model will use past
context, while the language model itself is restricted to a local context
window.

## Core Idea

The method trains a pair of small networks to produce query and key index
vectors from residual states:

```text
local residual state at a retrieval point -> query network -> block query
historical block residual summary        -> key network   -> block key
```

The score between the current query and a historical block key predicts the
attention that an unrestricted version of the same frozen model will assign to
that block during a future token horizon.

The language model used by the router is restricted to a local window of
length:

```text
local_context_length = 256
```

Historical information outside that window is available only through the
router's selected KV-cache blocks. The memory payload is the complete
all-layer KV cache for the selected block. The pooled residual is an index
representation only; it is not the memory payload.

## Relationship to the Existing Prompt-Based Index

The learned method and the current prompt-based residual index rely on the same
core hypothesis:

> A residual state computed from currently visible content contains enough
> information to predict what historical content the model will need next.

The current prompt-based method uses an index-only prompt branch and generated
prefix such as `FACT:` to make the frozen model create a retrieval-oriented
state. It is an implicit retrieval head implemented through prompting and the
model's existing instruction-following computation.

The learned method replaces that implicit transformation with a directly
trained router:

```text
prompt-based:
    z_prompt = frozen_model(local_state, retrieval_prompt, generated_prefix)

learned:
    q = query_network(local_residual_summary)
```

The new method is therefore not based on a different forecasting assumption.
Its intended advantages are:

- no retrieval-specific natural-language prompt;
- no index-only prefix generation;
- direct supervision from the frozen model's full-context attention pattern;
- explicit control over prediction horizon, retrieval interval, and retrieval
  budget;
- fewer assumptions about prompt format at deployment time.

The existing prompt-based index remains the primary baseline. Its results may
support the shared forecasting hypothesis, but they do not determine the
accuracy ceiling of the learned router because the state construction and
training targets differ.

## Goals

The project will test whether the learned router can:

1. predict future block-level attention from only a 256-token local state;
2. retrieve historical KV-cache blocks without a retrieval prompt;
3. reduce the behavioral gap between local-context and full-context inference;
4. control the accuracy/cost tradeoff through retrieval interval and `top_n`;
5. match or improve the current prompt-based index under the same memory,
   position, candidate-block, and inference budgets.

## Non-Goals for the First Version

The first version does not attempt to:

- train or modify the frozen language-model weights;
- claim that attention probability is a complete causal explanation;
- reconstruct exact full-context attention at every token, head, and layer;
- replace the KV payload with a residual vector;
- solve unbounded-context storage or approximate-nearest-neighbor scaling;
- use attention-output/value contribution as the primary training target;
- require autoregressively generated training trajectories.

## Terminology

### Memory block

A memory block is a mechanically segmented, fixed-size span of historical
tokens. Segmentation is independent of retrieval timing.

Each block record contains:

```text
block_id
logical token positions
token ids
complete all-layer K/V tensors
residual state samples used to construct the block index
block index key
optional source metadata
```

Block segmentation determines storage granularity. It does not determine when
the router is allowed to make a prediction.

### Retrieval point

A retrieval point is an inference boundary at which the router summarizes the
currently available local residual state and predicts historical blocks for a
future token horizon.

Retrieval points may occur at arbitrary token positions. They are not required
to coincide with memory-block boundaries.

### Prediction horizon

For a retrieval point `t`, the prediction horizon is the future interval whose
full-context attention pattern provides the supervision target:

```text
(t, t + H]
```

The exact first affected token must follow the actual cached-decoding schedule.
A retrieval decision computed after a model forward cannot change logits that
have already been produced by that forward. In code and experiment artifacts,
the target horizon must therefore be defined relative to the first subsequent
forward that can see the retrieved KV records, rather than by an ambiguous
"next token" label.

### Local state

The local state is computed by the frozen model with a strict visible-content
boundary. It may contain:

- at most the most recent 256 local tokens;
- KV blocks selected by an earlier router prediction, when that experimental
  condition enables recurrent retrieval;
- no unrestricted full-context states or outputs.

### Teacher

The teacher is the same frozen language model evaluated with full visible
context. The teacher supplies attention supervision only. Teacher residuals,
teacher KV payloads, and teacher-generated retrieval states are not exposed to
the student router.

### Student

The student is the same frozen model evaluated with the restricted local
window and any KV blocks selected by the router. The query network sees only
student-side residual information available at the retrieval point.

## Isolation and Information Boundary

The teacher and student use:

- the same model weights;
- the same token sequence for a supervised example;
- the same logical token positions;
- the same causal ordering.

They differ only in visible content:

```text
teacher: full historical context
student: local 256-token context + router-selected historical KV blocks
```

The teacher's full-context attention probabilities are labels. They must not
enter the query/key network inputs.

Every student forward must have an explicit context boundary. This prevents a
query residual from already containing unrestricted historical information.
Historical block summaries and block keys used during training must likewise
come from states available under the student-side memory policy used at
inference.

## Position and Cache Semantics

Every token has one logical position. Teacher and student runs use the same
logical positions for the same tokens.

Historical KV blocks retain the positions at which they were originally
created. Selecting a block changes its visibility, not its logical position.
Physical concatenation inside a packed cache must not be interpreted as
renumbering the historical tokens.

The attention mask must enforce:

- ordinary causality inside the local trajectory;
- visibility of selected historical blocks to subsequent eligible queries;
- invisibility of unselected historical blocks;
- no visibility from a token to future tokens;
- no current-window write/read leakage.

The first version follows the position and custom-prefix-mask principles of the
existing answer-block replay pipeline.

## Router Inputs

### Historical block key input

For a historical block `B_i`, collect residual states from the frozen,
student-side model trajectory and form a block summary:

```text
r_i = Pool({h_j : j in B_i})
k_i = KeyNetwork(r_i)
```

The initial pooling baseline is the mean residual state over the block. This
summary is used only to produce the index key. The full KV cache remains the
record returned after selection.

### Current query input

At retrieval point `t`, construct a query-side summary from residual states
available inside the local window:

```text
r_t = QuerySummary(local residual states available at t)
q_t = QueryNetwork(r_t)
```

The first research axis compares:

- a single residual state at the retrieval point;
- statistics over multiple recent residual states;
- statistics over a configurable query-summary window.

This choice does not change the memory-block segmentation.

### Router score

The two networks produce compatible vectors. The base score is:

```text
s(t, i) = similarity(q_t, k_i)
```

The initial similarity is a normalized dot product or cosine-compatible dot
product. Exact hidden sizes and projection widths are implementation
hyperparameters, not research assumptions.

## Teacher Attention Target

### Full-resolution label

For teacher layer `l`, head `h`, future query token `u`, and historical token
`j`, let:

```text
A[l, h, u, j]
```

be the full-context attention probability.

For a historical block `B_i`, aggregate token attention into block attention:

```text
a_i(t, H) = aggregate over
    u in future horizon (t, t + H]
    selected teacher layers l
    selected teacher heads h
    j in historical block B_i
```

The collector should preserve sufficient raw aggregation statistics so layer,
head, horizon, and block reductions can be changed without rerunning the
teacher whenever practical.

### Absolute historical mass and conditional block distribution

The target must distinguish total demand for external history from the
distribution of that demand across blocks.

Define total historical attention mass:

```text
m_hist(t, H) = sum_i a_i(t, H)
```

and, when `m_hist > 0`, the conditional distribution across historical blocks:

```text
p_i(t, H) = a_i(t, H) / m_hist(t, H)
```

These quantities have different meanings:

- `m_hist` answers whether the future horizon needs external history at all;
- `p_i` answers which historical blocks are expected to receive that demand.

A flat `p_i` with very small `m_hist` represents negligible historical use. A
flat `p_i` with large `m_hist` represents broad historical dependence. The
training artifacts must not erase this distinction by normalizing over
historical blocks without retaining the original total mass.

The exact first-version parameterization for preserving retrieval demand is an
implementation decision. It may be represented through calibrated absolute
scores, an explicit demand output, or an equivalent no-retrieval/local outcome.
Whichever representation is chosen must preserve both `m_hist` and `p_i` in
the saved labels and evaluation reports.

Current staged implementation decision: train the query-key ranking network
first and defer learned demand prediction. The dataset continues to retain
`m_hist`, while the optimization target uses `p_i` only. Retrieval uses either
a fixed Top-N budget or an explicitly configured threshold over query-key
softmax probabilities. This isolates ranking quality before a demand model is
designed and evaluated separately.

### Target aggregation remains configurable

The first version uses attention probability, but the following reductions
remain explicit experiment settings:

- teacher layers included;
- heads included;
- sum or mean across future query tokens;
- optional distance weighting inside the future horizon;
- block token sum versus length-normalized block mass;
- treatment of the current local window versus external history.

No aggregation choice should be silently hard-coded into the saved dataset.

## Training Objective

Training does not use hard `top_n` selection.

For every retrieval sample, the router scores all candidate historical blocks
used by that sample. The scores are trained against the teacher's soft
block-attention target.

The complete design may report separate losses for:

```text
conditional block-distribution prediction
total historical-demand prediction or calibration
```

Suitable probability losses include cross-entropy or KL divergence for the
conditional distribution. The exact loss for absolute historical mass depends
on its chosen parameterization.

Hard `top_n` is an inference policy and evaluation condition, not part of the
training graph. In the current query-key-only stage, conditional
block-distribution prediction is the sole training loss; historical-demand
prediction is deferred rather than implicitly approximated.

## Training Sample Contract

Each supervised retrieval sample should contain at least:

```text
sample_id
sequence_id
retrieval_position
first_future_position_affected_by_retrieval
future_horizon_length
local_context_start
local_context_end
candidate historical block ids
query residual state or query residual statistics
historical block residual summaries or references to stored keys
absolute teacher attention mass per block
total teacher historical attention mass
conditional teacher block distribution
teacher layer/head aggregation metadata
logical position metadata
```

The full-context teacher data and restricted student data must be generated
from aligned tokens and positions, while remaining isolated at the model-input
level.

The first version may use teacher-forced long sequences. Autoregressive data
generation is not required to establish whether local residual states predict
future block access.

## Inference Contract

At inference time:

1. The frozen model maintains a local visible-content window of at most 256
   tokens.
2. At a retrieval point, the query-side summary is constructed only from
   states permitted by the current student context boundary.
3. The query network scores historical block keys.
4. The inference policy decides whether retrieval is needed and how many
   blocks to expose.
5. Selected records contribute their complete all-layer KV cache, with original
   logical positions, to subsequent eligible forwards.
6. Unselected historical blocks remain invisible.
7. The local trajectory continues autoregressively until the next retrieval
   point.

The primary inference controls are:

```text
retrieval interval
prediction horizon
top_n
dynamic retrieval condition
query-summary length/statistics
```

The interval and horizon are related but distinct. The horizon defines the
teacher behavior being forecast. The interval determines how long an inference
decision is reused before the router is evaluated again.

## Dynamic Retrieval Interpretation

The learned output represents the frozen teacher's expected future attention
pattern.

A concentrated prediction indicates that a small number of historical blocks
may capture most expected historical attention. A flat prediction may indicate
either broad historical dependence or negligible historical dependence; the
total historical mass distinguishes those cases.

The inference policy may vary `top_n` and retrieval frequency according to the
predicted demand and concentration. The policy is evaluated by its downstream
accuracy/cost tradeoff rather than by assuming one fixed threshold is correct.

## Primary Research Hypotheses

### H1: Future-access predictability

Residual states computed from a 256-token local window contain enough
information to predict the historical blocks that the same frozen model will
attend to during a future horizon.

Prediction accuracy is expected to depend on:

- distance from the retrieval point;
- query-summary length;
- conversation/topic changes;
- whether previous retrieved content is visible;
- teacher layer and head selection.

### H2: Block-level concentration

The teacher's future historical attention is sufficiently concentrated at
block level that a limited `top_n` can cover a useful fraction of it.

For a target distribution `a_i`, oracle attention coverage at budget `N` is:

```text
coverage@N = sum of a_i over the N highest-mass historical blocks
```

This gives the compression ceiling before router prediction error is
considered.

### H3: Prompt-free router efficiency

A directly trained router can match or exceed the prompt-based implicit router
while reducing prompt constraints and index-generation cost.

## Evaluation

### Attention-prediction metrics

Report at minimum:

- conditional block-distribution KL or cross-entropy;
- top-1 and top-N teacher-block recall;
- predicted-block teacher attention coverage;
- oracle coverage@N;
- total historical-mass prediction error;
- prediction entropy and teacher entropy;
- metrics by prediction distance inside the future horizon.

### Downstream model metrics

Attention imitation alone is not the final success criterion. Evaluate the
restricted model after exposing predicted KV blocks:

- next-token negative log-likelihood or perplexity;
- divergence from full-context teacher logits;
- task accuracy, exact match, or answer F1 where labels exist;
- degradation relative to full context;
- improvement relative to the local-256 baseline;
- wrong-memory and conflict behavior on controlled tasks.

### Efficiency metrics

Report:

- router latency;
- retrieval frequency;
- mean and maximum selected block count;
- KV bytes transferred or made visible;
- attention tokens processed;
- end-to-end generation latency or throughput;
- extra cost relative to the prompt-based index branch.

### Required baselines

Use the same frozen model, token sequence, positions, local window, KV block
store, and candidate history for:

```text
full context
local context only (256)
random blocks
recent blocks
oracle teacher-attention top-N blocks
current prompt-based residual index
learned prompt-free block-attention index
```

The learned and prompt-based methods must be compared at matched `top_n` and KV
visibility budgets.

## First-Version Experiment Matrix

The initial study varies only the dimensions needed to validate the core
hypothesis:

### Query representation

```text
single residual at retrieval point
mean of recent residual states
mean plus simple residual statistics over a recent window
```

### Temporal controls

```text
future prediction horizon H
fixed retrieval interval
dynamic retrieval interval
```

### Retrieval budget

```text
fixed top_n
dynamic top_n
```

### Target scope

```text
selected layer
small layer band
all compatible attention layers
```

### Block controls

```text
block size
total historical block count
history distance
```

The matrix should be staged rather than expanded as one full Cartesian sweep.
The first result to establish is attention predictability at a fixed horizon,
interval, block size, and `top_n`.

## Later Target: Attention Output Contribution

The first version predicts attention probability. A later version may use
attention output to remove blocks that receive attention but contribute little
after value filtering.

For block `B_i`, a per-head contribution before output projection is:

```text
c_i = sum_{j in B_i} A_j * V_j
```

Unlike attention probability, `c_i` is a signed vector rather than a natural
probability. Contributions from different blocks may reinforce or cancel, and
their effect may change after head combination and output projection.

Attention-output supervision is therefore a separate follow-up objective. Its
purpose is to test whether the probability target contains blocks that are
attended to but filtered by `V`; it does not invalidate the probability target
as the first prompt-free routing baseline.

## Known Boundaries

The following are accepted properties of the design, not reasons to reject it:

- A retrieval prediction affects only forwards that occur after the decision;
  it cannot modify already-computed logits.
- Future-attention prediction becomes harder as the horizon moves farther from
  the retrieval point.
- A flat teacher distribution may require a larger `top_n` or may represent
  negligible historical demand; total historical mass disambiguates the two.
- Full-context and restricted-context internal trajectories differ because
  their visible content differs. The learned router exists to minimize the
  resulting behavioral gap; exact trajectory equality is not assumed.
- First-version attention probability is a routing target, not a claim of
  causal attribution.
- The teacher's supported full-context length bounds the directly supervised
  horizon available during training.

## Proposed Development Artifacts

Names may change during implementation, but responsibilities should remain
separate:

```text
attention target collector
restricted-context residual/KV collector
aligned retrieval-sample dataset
query/key router model
router trainer
KV block store and retrieval policy
restricted replay/evaluation runner
attention-prediction and downstream analysis reports
```

The existing answer-block cache and Gemma attention adapter provide reference
implementations for KV slicing, cache validation, logical positions, prefix
masks, and layer-online memory visibility. The learned router should remain a
separate pipeline until its data and inference contracts are validated.

## Decision Log

The following decisions are fixed for the first implementation:

- The frozen language model is not trained.
- The student local context length is 256.
- Memory records contain complete KV-cache blocks.
- Residual pooling creates only the block index representation.
- Memory-block segmentation is mechanical and independent of retrieval points.
- Retrieval points may occur at arbitrary token positions.
- The router predicts attention beginning with the first subsequent forward
  that can observe the retrieved KV blocks.
- The query network sees only restricted student-side state.
- The teacher uses full context and supplies attention labels only.
- Training uses soft attention targets and no hard `top_n`.
- Inference uses configurable or dynamic `top_n` and retrieval intervals.
- Teacher and student use the same logical positions for aligned tokens.
- Attention probability is the first-version target.
- Attention-output contribution is a later research extension.
- The current prompt-based residual index is the primary baseline and shares
  the same future-access prediction hypothesis.

The following remain experiment or implementation decisions:

- block size;
- future horizon;
- retrieval interval;
- query residual summary and statistics;
- router projection dimension and architecture;
- teacher layer/head aggregation;
- block-length normalization;
- representation of total historical demand;
- fixed versus dynamic inference policy;
- the attention-output contribution metric used after the first version.
