# 03 Prefill Phase

## Scope

This document defines the **semantic and execution characteristics of the prefill phase** in decoder-only Transformer inference.

It focuses on:
- Sequence-level execution semantics
- Token parallelism properties
- Interaction with KV Cache initialization

**Out of scope**:
- Decode phase behavior
- Training-time forward passes
- Prescriptive optimization or parallelization strategies

The prefill phase is treated as a **finite, stateless-to-stateful transition process**.

---

## Definition

The *prefill phase* is the stage of inference where the model processes the **initial prompt tokens** of each request and initializes the KV Cache.

Given a prompt of length `L`:
- The model consumes all `L` tokens
- KV Cache entries for positions `1..L` are produced
- No new tokens are generated

---

## Semantic Role of Prefill

Prefill serves two fundamental purposes:

1. Establishing the initial hidden states for all layers
2. Materializing the complete KV Cache corresponding to the prompt

After prefill completes, the model is eligible to enter the decode phase.

---

## Token-Level Semantics

### Causal Constraint Within Prefill

Within a single sequence:

> Token `i` may attend to tokens `≤ i`, but not to future tokens.

This constraint is identical to decode and is enforced by the causal attention mask.

---

### Execution Parallelism Over Tokens

Despite causal masking, prefill exhibits a key property:

> All prompt tokens can be processed in a **single forward pass**.

This is possible because:
- All token embeddings are known in advance
- Causal masking removes illegal dependencies

As a result, token-level computation during prefill is **structurally parallelizable**.

---

## Execution Flow of Prefill

For each request, prefill consists of:

1. Embedding lookup for all prompt tokens
2. Forward propagation through all Transformer layers
3. Attention computation with causal masking
4. KV Cache construction for all layers and positions

All operations occur within one logical inference step.

---

## Batch Semantics During Prefill

### Batch Composition

During prefill:
- Requests with different prompt lengths may be batched together
- Padding or packing may be applied

Batching does not alter semantic correctness.

---

### Batch Stability

Unlike decode:
- Batch size is fixed for the duration of a prefill pass
- Sequences do not terminate mid-prefill

This yields predictable execution behavior.

---

## KV Cache Interaction

### Write-Only Initialization

During prefill:
- KV Cache entries are **written**, not read
- One K and V entry is produced per layer per token

The KV Cache transitions from an empty or uninitialized state to a fully materialized state.

---

### Determinism

Given identical:
- Model parameters
- Prompt tokens

Prefill KV Cache contents are deterministic.

---

## Latency and Throughput Characteristics

Prefill is typically:

- Throughput-oriented
- Less sensitive to per-token latency

Total prefill cost scales approximately with:

```
Total prompt tokens × Per-token compute cost
```

Batching and token parallelism strongly influence performance.

---

## Parallelism Envelope (Non-Prescriptive)

From a semantic standpoint, prefill permits parallelism along:

- Token dimension (within a sequence)
- Batch dimension (across sequences)
- Independent computations inside a layer

It does **not** permit:
- Violation of causal masking
- Reuse of uninitialized KV Cache entries

---

## Transition to Decode Phase

Prefill ends when:

- All prompt tokens have been processed
- KV Cache is fully initialized

Only after this point:
- Autoregressive decode may begin
- Token-by-token generation becomes valid

The transition is a **hard semantic boundary**.

---

## Failure-Prone Assumptions

The following assumptions are **invalid** under prefill semantics:

- Treating prefill as an iterative token-by-token process
- Assuming KV Cache reads during prefill
- Mixing decode scheduling constraints into prefill

---

## Summary

The prefill phase is:
- Finite and bounded by prompt length
- Structurally parallel over tokens
- Responsible for KV Cache initialization

Correct separation of prefill and decode semantics is essential for any inference reasoning or strategy construction.