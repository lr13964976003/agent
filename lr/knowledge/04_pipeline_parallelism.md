# 04 Pipeline Parallelism (PP)

## Scope

This document defines the **semantic foundations, constraints, and execution structure of Pipeline Parallelism (PP)** in decoder-only Transformer inference.

It focuses on:
- Layer-wise partitioning across devices
- Interaction with decode and prefill phases
- Communication and synchronization boundaries

**Out of scope**:
- Specific hardware or framework implementations
- Low-level kernel optimization
- TP or SP internal mechanics

PP is treated as a **cross-layer parallelization primitive**.

---

## Definition

Pipeline Parallelism partitions the model **across layers**, assigning subsets of consecutive layers to different devices.

Key property:

> PP introduces potential concurrency along the layer dimension but must respect intra-layer and inter-step data dependencies.

---

## Fundamental Constraints

1. **Layer Dependency**
   - Layer `l` cannot execute until layer `l-1` has produced its outputs for the current step.
2. **Step Dependency**
   - For decode, step `t+1` cannot start until step `t` completes and KV Cache updates are committed.
3. **Phase Sensitivity**
   - Prefill allows more aggressive overlapping due to full prompt availability.

Violating any of these constraints results in **incorrect inference semantics**.

---

## Execution Semantics

### Prefill Phase

During prefill:
- All tokens are available
- Layers can be staggered in a pipelined fashion
- Communication occurs along layer boundaries to transfer intermediate activations

Characteristics:
- Batch and token dimensions can be exploited for parallelism
- Layer-wise pipeline introduces controlled concurrency
- Latency can be hidden by overlapping computation and communication

### Decode Phase

During decode:
- Only one token per sequence exists per step
- Strict autoregressive dependency prevents cross-step pipelining
- PP can only overlap layers within the same step

Consequences:
- Per-step latency is dominated by **the slowest layer in the pipeline**
- Step-to-step overlap is not permissible

---

## Communication Semantics

PP requires transferring **intermediate activations** between layers/devices:
- Occurs after forward pass of a layer
- Synchronous within a step
- Must preserve batch, head, and token ordering

Communication overhead must be considered but **cannot violate KV Cache or causal mask semantics**.

---

## Interaction with Tensor and Sequence Parallelism

- **TP** can be applied within each pipeline stage
- **SP** is phase-dependent and can only be applied in prefill, even within a stage
- PP defines inter-layer synchronization points

Combination constraints:
- TP and SP within stage must not break intra-step correctness
- PP must respect all stage boundaries for correctness

---

## Batch and Token Considerations

- Batch dimension is orthogonal to PP
- Prefill: multiple tokens per step can exploit pipeline concurrency
- Decode: single token per step limits pipeline depth concurrency

---

## Failure-Prone Assumptions

Invalid assumptions include:
- PP can parallelize across decode steps
- PP removes KV Cache temporal dependencies
- PP can ignore causal masking constraints

PP only provides **within-step, cross-layer concurrency**.

---

## Role in Parallel Strategy Composition

Pipeline Parallelism:
- Introduces the first **time-overlapped dimension** in inference
- Interacts with TP/SP to maximize device utilization
- Must always respect phase-specific constraints

It serves as a bridge from **layer semantics** to **multi-device scheduling**.

---

## Summary

PP semantics can be summarized as follows:
- Partitions layers across devices
- Introduces layer-level concurrency
- Strictly bounded by intra-step and inter-step dependencies
- Prefill: more concurrency; Decode: limited by single-token autoregression

PP forms the **backbone for cross-layer multi-device inference reasoning** and is essential for constructing valid parallel DAGs.