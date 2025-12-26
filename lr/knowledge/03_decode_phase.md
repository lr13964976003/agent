# 03 Decode Phase

## Scope

This document defines the **semantic and execution characteristics of the decode phase** in decoder-only Transformer inference.

It focuses on:
- Token-level execution semantics
- Temporal and dependency constraints
- System-level behavior during iterative decoding

**Out of scope**:
- Prefill phase behavior
- Training-time decoding
- Any prescriptive optimization or parallelization strategy

The decode phase is treated as a **stateful, iterative process** governed by strict causal dependencies.

---

## Definition

The *decode phase* is the stage of inference where new tokens are generated autoregressively, one token at a time, for each active sequence.

At decode step `t`:
- Each active sequence produces **at most one new token**
- The model conditions on all previously generated tokens

---

## Autoregressive Semantics

### Token-Level Dependency

Decode obeys the following invariant:

> Token `t` cannot be computed until token `t-1` has been fully generated and its KV Cache state has been committed.

This dependency applies:
- Within a single sequence
- Across all Transformer layers

As a result, decode steps form a strictly ordered sequence in time.

---

### Step Granularity

A *decode step* corresponds to:

- One forward pass through all Transformer layers
- For exactly one new token per active sequence

There is no semantic subdivision of a decode step along the time dimension.

---

## Execution Flow of a Decode Step

For each active sequence in the batch, a single decode step consists of:

1. Embedding lookup for the current token
2. Forward propagation through all Transformer layers
3. Attention computation using the full KV Cache
4. Logits computation
5. Token selection (e.g., argmax or sampling)
6. KV Cache update

All steps must complete before the next decode step begins.

---

## Batch Semantics During Decode

### Dynamic Batch Size

During decoding:
- New requests are typically not inserted mid-step
- Batch size decreases as sequences terminate

Consequences:
- Effective batch size is non-increasing
- Compute utilization may degrade over time

---

### Per-Sequence Independence

Within a decode step:
- Sequences are independent except for shared batching
- There is no semantic dependency across sequences

Batching is purely a performance and scheduling mechanism.

---

## KV Cache Interaction

### Read Semantics

At decode step `t`:
- Attention queries read all KV Cache entries from steps `≤ t-1`
- Reads occur for every layer and every head

### Write Semantics

- Exactly one new K and V entry is appended per layer per sequence
- Writes occur after the forward pass for the current token

KV Cache consistency is a prerequisite for progressing to the next step.

---

## Latency Characteristics

Decode is inherently **latency-sensitive** due to:

- Strict step-by-step dependency
- Limited parallelism across time
- Repeated access to growing KV Cache

Total decode latency scales approximately with:

```
Number of generated tokens × Per-step latency
```

---

## Parallelism Envelope (Non-Prescriptive)

From a semantic standpoint, decode permits parallelism only along:

- The batch dimension (multiple sequences)
- Independent computations within a single step

It does **not** permit:
- Parallel execution across decode steps
- Speculative reordering of token generation

---

## Interaction with Scheduling and System Behavior

### Synchronization Points

Each decode step introduces implicit synchronization:
- All layers must finish before token selection
- KV Cache updates must complete before the next step

These synchronization points bound achievable overlap.

---

### Resource Utilization Dynamics

As decoding progresses:
- KV Cache memory footprint increases
- Batch size may shrink
- Compute-to-memory ratio may decrease

This leads to time-varying performance characteristics.

---

## Failure-Prone Assumptions

The following assumptions are **invalid** under decode semantics:

- Treating decode as a sequence-parallel workload
- Assuming fixed batch size throughout inference
- Ignoring KV Cache read amplification

Violating these assumptions leads to incorrect reasoning about performance.

---

## Summary

The decode phase is:
- Autoregressive and strictly ordered
- State-dependent via KV Cache
- Latency-dominated rather than throughput-dominated

Any reasoning about inference parallelism or DAG construction must respect these decode-specific constraints.