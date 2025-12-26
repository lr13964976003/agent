# Synchronization Points

## Scope

This document defines the **required synchronization points** in decoder-only Transformer inference to maintain correctness across devices and parallel primitives.

It focuses on:
- When and where to synchronize TP, SP, PP, EP, and DP
- Phase-specific constraints (prefill vs decode)
- Interaction with microbatching and overlapping compute/communication

**Out of scope**:
- Low-level hardware implementation
- Training-time gradient synchronization

---

## Principles

- Synchronization points ensure correctness of:
  - KV Cache updates
  - Inter-stage activation transfers
  - All-reduce/all-gather operations
  - Expert routing
  - Output collection for DP
- Too few synchronization points lead to race conditions or incorrect outputs
- Too many synchronization points reduce parallelism and throughput

---

## Prefill Phase

- Multiple tokens per sequence allow asynchronous execution
- Synchronization required at:
  - **TP boundaries**: after linear layer or attention computation
  - **PP boundaries**: after stage completes its layers for a token batch
  - **SP token partitions**: after K/V aggregation
  - **EP expert routing**: after expert outputs are gathered per token
  - **DP microbatch outputs**: before moving to next microbatch

Characteristics:
- Allows overlap of computation and communication
- Preserves intra-token and inter-layer dependencies
- Reduces idle time by careful placement of sync points

---

## Decode Phase

- Only one token per sequence per step is available
- Synchronization required at:
  - **TP boundaries**: after linear or attention layers per token
  - **PP boundaries**: stage-to-stage activation transfer per token
  - **EP boundaries**: after expert outputs are combined for token
  - **DP outputs**: gather logits across devices for current token step
- SP is not allowed
- KV Cache must be synchronized after each token step

Consequences:
- Correctness depends on strict stage-to-stage and step-to-step synchronization
- Overlap is allowed within a token step, not across steps

---

## Interaction with Other Primitives

| Primitive | Synchronization Points | Notes |
|-----------|----------------------|-------|
| TP        | After linear layers and attention computation | All-reduce/all-gather completed before next dependent computation |
| SP        | After K/V aggregation (prefill only) | Ensures consistent attention state across token partitions |
| PP        | After each stage per token/microbatch | Activation transfer must complete before downstream stage starts |
| EP        | After expert outputs per token | Sparse outputs gathered before next layer computation |
| DP        | After microbatch outputs or per-token logits | Ensures consistent batch-level outputs |

Constraints:
- Must preserve all causal, token, and layer dependencies
- No cross-step synchronization in decode except for batch-level DP gathering

---

## Failure-Prone Assumptions

- Ignoring necessary synchronization points leads to incorrect KV Cache or logits
- Over-synchronization reduces throughput unnecessarily
- Overlapping decode steps across devices without proper sync is invalid
- Misaligned tensor shapes during inter-stage or inter-device sync

---

## Summary

Synchronization points are critical to:
- Preserve correctness across TP, SP, PP, EP, and DP
- Ensure proper KV Cache updates, expert routing, and microbatch handling
- Maximize throughput while avoiding race conditions

Phase-specific summary:
- Prefill: allow asynchronous execution with sync at key primitive boundaries
- Decode: strict per-token, per-stage, per-device synchronization

This module ensures **safe and efficient multi-device Transformer inference**.