# 04 Sequence Parallelism (SP)

## Scope

This document defines the **semantic foundations, legality conditions, and structural constraints of Sequence Parallelism (SP)** in decoder-only Transformer inference.

It focuses on:
- Partitioning along the sequence (token) dimension
- Interaction with causal attention semantics
- Phase-dependent validity (prefill vs decode)

**Out of scope**:
- Specific algorithms such as Ring Attention or Ulysses
- Hardware- or topology-specific implementations

SP is treated as a **cross-token parallelization primitive with strict semantic preconditions**.

---

## Definition

Sequence Parallelism partitions computation along the **sequence length dimension**.

Given a sequence of length `L`:
- Tokens are divided into disjoint contiguous or logical segments
- Each partition processes a subset of tokens

Key distinction from TP:

> SP partitions *tokens*, whereas TP partitions *tensor dimensions*.

---

## Fundamental Legality Constraint

SP is semantically valid **if and only if** all required token dependencies are satisfied.

Formally:

> For any token `i`, all tokens `j ≤ i` that `i` may attend to must be available (materialized or communicated) before computing token `i`.

Violating this condition breaks causal correctness.

---

## Interaction with Causal Attention

### Causal Mask Implications

Decoder-only attention enforces:

- Token `i` cannot attend to token `j > i`
- Token `i` may attend to all `j ≤ i`

This induces a **strict partial order** over tokens.

SP must preserve this order.

---

### Dependency Shape

The dependency graph along the sequence dimension is:

- Lower-triangular
- Prefix-complete

This structure heavily constrains admissible SP schemes.

---

## Phase-Dependent Semantics

### Prefill Phase

During prefill:

- All tokens are known in advance
- Causal masking removes illegal forward dependencies

As a result:

> Token-level computation can be parallelized across the sequence dimension.

SP is **semantically permissible** in prefill, subject to correct handling of attention dependencies.

---

### Decode Phase

During decode:

- Only one new token exists per step
- Future tokens do not exist

Therefore:

> There is no sequence dimension to parallelize across time.

SP across decode steps is **semantically invalid**.

---

## Communication Requirements

### Dependency Materialization

When SP is applied:

- Tokens earlier in the sequence must expose their K/V states
- Later partitions must receive these states before attention computation

This induces communication proportional to:

- Sequence length
- Attention head dimensions

---

### Synchronization Characteristics

SP introduces:

- Ordered communication along the sequence dimension
- Synchronization points aligned with attention computation

These synchronization points are **phase-dependent**.

---

## Interaction with KV Cache

### Prefill

- KV Cache entries are produced for all tokens
- SP partitions jointly contribute to cache construction

KV Cache remains logically ordered by token index.

---

### Decode

- KV Cache grows one token at a time
- There is no meaningful SP axis

SP cannot alter KV Cache growth semantics.

---

## Batch Semantics

Sequence Parallelism:

- Operates within a single sequence
- Is orthogonal to batching across sequences

Batching does not relax SP legality constraints.

---

## Failure-Prone Assumptions

The following assumptions are **invalid**:

- SP can parallelize autoregressive decode steps
- SP removes causal dependencies
- SP is equivalent to TP

SP does not change the fundamental attention dependency graph.

---

## Role in Parallel Strategy Composition

Sequence Parallelism is typically:

- Applicable primarily to prefill
- Combined with TP for scalability

Any higher-level strategy must explicitly disable SP during decode.

---

## Summary

Sequence Parallelism:

- Partitions computation along the token dimension
- Is tightly constrained by causal attention semantics
- Is phase-sensitive: valid in prefill, invalid in decode

It represents the **most easily misapplied parallel primitive** and must be handled with explicit semantic checks.