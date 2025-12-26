# 04 Tensor Parallelism (TP)

## Scope

This document defines the **semantic foundations and structural constraints of Tensor Parallelism (TP)** in decoder-only Transformer inference.

It focuses on:
- Mathematical legality of tensor partitioning
- Operator-level execution semantics
- Communication and synchronization boundaries

**Out of scope**:
- Pipeline scheduling
- Expert Parallelism
- Any hardware- or vendor-specific implementation detail

TP is treated as a **within-layer parallelization primitive**.

---

## Definition

Tensor Parallelism partitions **individual tensors or tensor operations** across multiple devices while preserving exact numerical equivalence to single-device execution.

Key property:

> TP does not change model structure or execution order; it only decomposes tensor computations.

---

## Fundamental Constraint

Tensor Parallelism is valid **if and only if**:

- The partitioned operation is mathematically decomposable
- Partial results can be recombined deterministically

Any TP scheme violating this constraint is semantically invalid.

---

## Applicable Tensor Dimensions

### Hidden Dimension Partitioning

The most common TP axis is the hidden dimension `H`.

For a linear transformation:

```
Y = X · W
```

Where:
- `X ∈ R^{B×H}`
- `W ∈ R^{H×H}`

The weight matrix may be partitioned along:

- Column dimension (output split)
- Row dimension (input split)

Both preserve correctness under proper aggregation.

---

### Attention Head Dimension

Multi-head attention exposes a natural TP boundary:

- Heads are independent before the output projection
- Each device may own a subset of heads

This form of TP aligns with head-level independence defined in the attention structure.

---

## Operator-Level Semantics

### Linear Layers

For linear layers:

- Partial matrix multiplications are computed independently
- Results are combined via concatenation or summation

No cross-step dependency is introduced.

---

### Attention Computation

Within a single attention layer:

- Q, K, V projections may be tensor-parallel
- Attention score computation is local to each partition
- Output projection requires aggregation

KV Cache entries remain logically contiguous despite physical partitioning.

---

## Communication Semantics

### Collective Operations

TP introduces mandatory collectives such as:

- All-reduce
- All-gather

These collectives:
- Occur at fixed operator boundaries
- Are synchronous within a decode or prefill step

---

### Synchronization Points

Synchronization introduced by TP:

- Does not cross layer boundaries
- Does not cross decode steps

TP therefore preserves the original execution DAG structure.

---

## Interaction with Execution Phases

### Prefill Phase

During prefill:

- TP benefits from large token-level parallelism
- Communication overhead is amortized

TP scales efficiently in this phase.

---

### Decode Phase

During decode:

- TP operates within a single-token forward pass
- Communication latency directly impacts per-step latency

TP effectiveness is bounded by synchronization cost.

---

## Batch Semantics

Tensor Parallelism:

- Is orthogonal to batch size
- Does not alter per-sequence independence

Batching and TP compose multiplicatively.

---

## Failure-Prone Assumptions

The following assumptions are **invalid**:

- TP can remove autoregressive dependencies
- TP reduces the number of decode steps
- TP changes KV Cache semantics

TP only reduces **intra-step compute**, not temporal depth.

---

## Role in Parallel Strategy Composition

Tensor Parallelism is typically:

- The lowest-level parallel primitive
- Combined with SP, EP, or PP

Higher-level strategies must respect TP-imposed synchronization boundaries.

---

## Summary

Tensor Parallelism:

- Decomposes tensor operations without altering semantics
- Operates strictly within a layer and a step
- Introduces fixed, synchronous communication points

It defines the **base unit of safe parallel decomposition** for Transformer inference.