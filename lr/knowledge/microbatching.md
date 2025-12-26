# Microbatching

## Scope

This document defines the **execution semantics and scheduling of microbatching** in decoder-only Transformer inference.

It focuses on:
- Splitting input batches into microbatches for device-level parallelism
- Integration with Pipeline Parallelism (PP), Tensor Parallelism (TP), Sequence Parallelism (SP), Expert Parallelism (EP), and Data Parallelism (DP)
- Phase-specific considerations (prefill vs decode)

**Out of scope**:
- Training-time microbatching for gradient accumulation
- Hardware-specific kernel optimizations

---

## Definition

Microbatching splits the overall batch into smaller chunks (microbatches) that are processed sequentially or in a pipelined manner to optimize device utilization and reduce memory footprint.

Key properties:
- Each microbatch contains a subset of sequences from the original batch
- Each microbatch passes through the model independently
- DP replicates microbatches across devices

---

## Prefill Phase

- Multiple tokens per sequence are available
- Microbatches can be pipelined across PP stages
- TP can be applied within a microbatch
- SP can partition tokens within a microbatch
- EP applies per token/expert routing as necessary
- Communication occurs asynchronously between pipeline stages for each microbatch

Characteristics:
- Overlaps computation and communication across microbatches
- Reduces peak memory usage per device
- Enables more efficient utilization of multiple devices

---

## Decode Phase

- Only one token per sequence per step is available
- Microbatches contain multiple sequences but only single token per sequence
- PP can pipeline layers per microbatch token step
- TP and EP can be applied within a microbatch
- SP is not allowed
- Communication remains synchronous at stage boundaries

Consequences:
- Each microbatch is processed sequentially for correctness
- Stage-to-stage latency accumulates per microbatch
- KV Cache updates remain isolated per microbatch and per sequence

---

## Interaction with Other Primitives

- **TP**: Within microbatch, tensors are split along specified dimensions
- **SP**: Token-dimension partitioning only in prefill microbatches
- **PP**: Microbatch can be pipelined across stages
- **EP**: Sparse expert routing per token within microbatch
- **DP**: Microbatch replicated across devices for throughput

Constraints:
- Must preserve intra-sequence dependencies
- Stage boundaries and KV Cache updates must be respected
- Microbatching does not introduce cross-microbatch dependencies

---

## Failure-Prone Assumptions

- Overlapping decode steps across microbatches is invalid
- Ignoring stage boundaries within microbatch processing leads to incorrect outputs
- SP cannot be applied in decode microbatches
- KV Cache must not be shared across microbatches for independent sequences

---

## Summary

Microbatching provides:
- A mechanism to split large batches into manageable units
- Prefill: overlapping computation and communication across microbatches
- Decode: sequential microbatch processing for correctness
- Integration with TP, SP (prefill), PP, EP, and DP

This ensures **memory-efficient and throughput-optimized scheduling** of Transformer inference workloads.

