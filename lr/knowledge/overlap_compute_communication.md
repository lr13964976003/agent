# Overlap of Compute and Communication

## Scope

This document defines the **principles and execution semantics for overlapping computation and communication** in decoder-only Transformer inference.

It focuses on:
- How to overlap TP/SP/PP/EP/DP computations with inter-device communication
- Phase-specific considerations (prefill vs decode)
- Integration with microbatching

**Out of scope**:
- Low-level hardware implementation details
- Training-time gradient communication

---

## Principles

- **Overlap** improves device utilization and reduces effective latency
- Communication should be asynchronous where possible
- Computation that does not depend on pending communication can proceed concurrently

Types of communication:
1. **TP All-Reduce / All-Gather**
2. **PP activation transfer**
3. **SP K/V aggregation (prefill only)**
4. **EP scatter/gather for expert routing**
5. **DP output gathering**

---

## Prefill Phase

- Multiple tokens per sequence are available
- TP, SP, PP, EP, and DP communication can be overlapped with computation for other tokens or layers
- Microbatching allows pipelining communication for one microbatch while computing another
- Activation transfer between PP stages can begin as soon as previous layer outputs are ready
- K/V aggregation for SP can occur concurrently with layer computation within a stage
- EP scatter/gather can be performed asynchronously per token

Characteristics:
- Reduces idle time for devices
- Improves throughput without violating dependencies
- Requires careful scheduling to avoid race conditions

---

## Decode Phase

- Only one token per sequence per step is available
- PP, TP, EP, and DP communication can be overlapped within a stage but **not across decode steps**
- Microbatching can provide some concurrency across sequences
- SP is not allowed
- KV Cache updates must complete before using the token in subsequent layers

Consequences:
- Latency limited by the slowest stage or device per token step
- Overlap helps throughput but cannot violate causal dependencies
- Correctness requires strict stage and step synchronization

---

## Interaction with Other Primitives

| Primitive | Overlap Allowed | Notes |
|-----------|----------------|-------|
| TP        | Yes | All-reduce/all-gather can overlap with independent computations |
| SP        | Prefill only | Token K/V aggregation can be overlapped with other computations within stage |
| PP        | Yes | Activation transfer between stages can overlap with intra-stage computation |
| EP        | Yes | Scatter/gather per token can overlap with other expert computations |
| DP        | Yes | Output gathering can overlap with next microbatch computation |

Constraints:
- Dependencies must be preserved
- No cross-step overlap for decode
- Asynchronous communication must respect tensor shapes and device mapping

---

## Failure-Prone Assumptions

- Overlapping decode steps across devices or microbatches is invalid
- Ignoring KV Cache dependencies when overlapping SP/PP/EP
- Assuming all communication is automatically overlapped without scheduling
- Misalignment of tensor shapes during asynchronous transfer

---

## Summary

Overlapping compute and communication provides:
- Reduced idle time
- Increased throughput
- Maintained correctness for both prefill and decode phases

Key rules:
- Prefill: maximize concurrency across tokens, layers, and microbatches
- Decode: overlap within token step, not across steps
- Must coordinate TP, SP, PP, EP, and DP communication properly

This ensures **efficient and correct execution of multi-device Transformer inference**.