# 02 KV Cache Semantics

## Scope

This document defines the **semantic, structural, and lifecycle properties of the Key-Value (KV) Cache** used during inference in decoder-only Transformer models.

It focuses on:
- What the KV Cache represents
- How it evolves across inference phases
- The constraints it imposes on execution order and data placement

**Out of scope**:
- Training-time activations
- Cache compression or eviction strategies
- Model-specific kernel optimizations

The KV Cache is treated here as a **logical abstraction**, independent of any particular hardware or framework implementation.

---

## Definition

The KV Cache stores the **Key (K)** and **Value (V)** tensors produced by the self-attention module for previously processed tokens.

For each Transformer layer and each attention head, the cache retains:

```
K_cache[layer][head][time]
V_cache[layer][head][time]
```

These cached tensors are reused in subsequent decode steps to avoid recomputation.

---

## Logical Structure

### Dimensionality

For a single layer, the KV Cache has the following logical shape:

```
[B, S, Nh, Dh]
```

Where:
- `B`: batch size (active sequences)
- `S`: current sequence length
- `Nh`: number of attention heads
- `Dh`: head dimension

Each layer maintains an independent KV Cache.

---

### Layer Isolation

- KV Cache entries are **not shared across layers**
- Each layer appends its own K and V tensors independently

This isolation enforces strict layer-wise execution order during inference.

---

## Lifecycle Across Inference Phases

### Prefill Phase

During prefill:

- K and V are computed for all input prompt tokens
- KV Cache is initialized for each layer
- Cache length grows to the full prompt length in a single forward pass

Characteristics:
- Write-heavy
- No cache reads required within the same layer
- Highly parallelizable across tokens

---

### Decode Phase

During decode:

- Exactly one new token per active sequence is processed per step
- New K and V tensors are appended to the existing cache
- Attention queries read **all cached K and V** up to the current step

Characteristics:
- Read-dominant access pattern
- Monotonic growth of cache length
- Strong temporal dependency between steps

---

## Temporal Dependency Semantics

The KV Cache enforces the following invariant:

> For a given sequence, cache entries at time `t` must exist before attention for time `t+1` can be computed.

Consequences:
- Decode steps cannot be reordered
- Partial execution across time is not semantically valid

This invariant is independent of any batching or parallelism scheme.

---

## Batch Dynamics and Cache Consistency

### Dynamic Batch Size

As sequences terminate:
- Their KV Cache entries are no longer extended
- Remaining sequences continue to grow

This leads to:
- Ragged cache lengths across sequences
- Divergent memory footprints within a batch

---

### Cache Alignment

To enable batched computation:
- Systems may logically align cache tensors
- Padding or indirection may be introduced

Such alignment is a **system-level artifact** and does not alter the logical semantics defined here.

---

## Memory Footprint Characteristics

The total KV Cache memory footprint scales as:

```
O(L × B × S × Nh × Dh)
```

Where:
- `L`: number of Transformer layers

Key observations:
- Memory grows linearly with sequence length
- Memory persists for the lifetime of each sequence
- KV Cache often dominates inference-time memory usage

---

## Access Patterns

### Read Patterns

- During decode, attention queries read the full history of K and V
- Reads are repeated at every decode step

### Write Patterns

- Exactly one write per sequence per layer per decode step
- Writes are append-only

The combination yields a read-heavy, append-only memory behavior.

---

## Implications for Execution Ordering

The KV Cache semantics impose the following constraints:

- Layer `l` must complete KV writes before layer `l+1` can consume them
- Decode step `t` must complete before step `t+1` begins
- Cache consistency must be preserved across all attention heads

These constraints define a strict partial order on execution events.

---

## Non-Prescriptive Notes

This document intentionally does not address:
- How KV Cache is partitioned across devices
- Whether cache is replicated or sharded
- How communication is implemented

Such decisions must respect the semantic constraints described above.

---

## Summary

The KV Cache is:
- Persistent across inference steps
- Append-only in time
- Read-heavy during decoding
- Layer-isolated and sequence-specific

These properties fundamentally constrain parallelism, scheduling, and DAG construction in LLM inference.