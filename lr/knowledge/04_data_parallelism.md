# 04 Data Parallelism in Inference (DP)

## Scope

This document defines the **semantic foundations, constraints, and execution structure of Data Parallelism (DP)** during decoder-only Transformer inference.

It focuses on:
- Parallelization across batch/request dimension
- Interaction with prefill and decode phases
- Interaction with TP, SP, PP, and EP

**Out of scope**:
- Training-time DP specifics
- Low-level kernel implementation

DP is treated as a **batch-level parallelization primitive**.

---

## Definition

Data Parallelism replicates the model across multiple devices, each processing a distinct subset of the batch or request set.

Key properties:
- Each device holds a full copy of the model
- Input batches are split across devices
- Outputs are gathered post-computation
- KV Cache is maintained per device per sequence

---

## Phase Sensitivity

### Prefill Phase

- Batches can be split across devices
- Each device independently computes embeddings, attention, and KV Cache for its subset of sequences
- SP, TP, PP, and EP can still be applied within each device

### Decode Phase

- DP can operate across sequences in the batch
- Each device handles independent sequences, respecting single-token-per-step constraint
- No cross-device dependencies exist except for optional output gathering
- Temporal autoregressive dependencies within each sequence remain unchanged

---

## Interaction with Other Primitives

- **TP**: Can be applied within a device for tensor-level parallelism
- **SP**: Phase-sensitive; prefill only, per-device
- **PP**: Layer-wise pipeline per device; DP does not change layer assignment
- **EP**: Sparse expert routing per device; DP does not alter routing logic

Constraints:
- DP only partitions the batch dimension
- Cannot violate intra-sequence or cross-step dependencies
- Compatible with all other primitives within device boundaries

---

## Communication and Synchronization

- Minimal communication is required during inference: typically only for output gathering
- Synchronization occurs at batch boundaries if needed
- No cross-sequence KV Cache or attention communication is required

---

## Failure-Prone Assumptions

Invalid assumptions include:
- DP can reduce per-sequence decode steps
- DP can parallelize within a single sequence across devices
- DP affects KV Cache or attention semantics

DP only increases **throughput across sequences**, not per-sequence latency.

---

## Role in Parallel Strategy Composition

- DP is **the top-level, batch-oriented parallel primitive**
- It can coexist with TP, SP, PP, and EP
- Used to scale throughput when multiple sequences or requests are processed
- Does not interfere with temporal or layer-level correctness

---

## Summary

Data Parallelism in inference:
- Replicates the full model across devices
- Splits sequences across devices
- Preserves all intra-sequence and step-level dependencies
- Compatible with all other parallel primitives
- Primarily improves throughput across sequences, not latency within a sequence

DP completes the set of fundamental parallel primitives for decoder-only Transformer inference.