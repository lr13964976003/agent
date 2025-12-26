# 02 Attention Structure

## Scope

This document describes the **structural and computational properties of the self-attention module** in decoder-only Transformer models during inference.

It focuses on:
- Tensor shapes and dimensional relationships
- Data dependencies within attention computation
- Memory access patterns, especially KV Cache

**Out of scope**:
- Training-time backward passes
- Optimization techniques
- Any prescriptive parallelization strategy

The purpose is to provide **factual constraints** that downstream reasoning about parallelism must respect.

---

## High-Level Attention Computation

For a single Transformer layer, self-attention consists of the following logical steps:

1. Input hidden states
2. Linear projections to Query (Q), Key (K), and Value (V)
3. Attention score computation
4. Softmax normalization
5. Weighted sum of values
6. Output projection

Each step has distinct dependency and memory characteristics.

---

## Tensor Dimensions and Notation

Let:

- `B`: batch size (number of active sequences)
- `S`: sequence length (context length)
- `H`: hidden size
- `Nh`: number of attention heads
- `Dh`: head dimension, where `H = Nh × Dh`

Input hidden states have shape:

```
[B, S, H]
```

---

## QKV Projections

### Linear Projection

The input hidden states are projected into Q, K, and V using learned weight matrices:

```
Q = X · Wq
K = X · Wk
V = X · Wv
```

Resulting tensor shapes:

```
Q, K, V: [B, S, Nh, Dh]
```

Key properties:
- Projections are independent across tokens
- Projections are independent across heads
- Weight matrices are shared across all tokens and sequences

---

### Head Partitioning

Each attention head operates on a disjoint slice of the hidden dimension:

- Heads are independent during score computation
- Head outputs are concatenated before the output projection

This structural independence is a core property of multi-head attention.

---

## Attention Score Computation

For each head:

```
Scores = Q · K^T / sqrt(Dh)
```

Resulting shape per head:

```
[B, S_query, S_key]
```

Key dependencies:
- Each query token depends on **all previous key tokens**
- In decoder-only models, attention is causally masked

---

## Causal Masking

Decoder-only self-attention applies a causal mask:

- Token `t` may attend only to tokens `≤ t`
- Prevents information flow from future tokens

Causal masking enforces a strict ordering constraint across time steps.

---

## Softmax and Value Aggregation

The attention scores are normalized:

```
P = softmax(Scores)
```

Then applied to values:

```
Output = P · V
```

Properties:
- Softmax is applied independently per head
- Value aggregation depends on all attended positions

---

## Output Projection

The per-head outputs are concatenated and projected:

```
Y = concat(head_outputs) · Wo
```

Resulting shape:

```
[B, S, H]
```

The output projection re-mixes information across heads.

---

## KV Cache Semantics

### Definition

During inference, Keys and Values are cached to avoid recomputation.

- KV Cache stores K and V for all past tokens
- Cache grows monotonically with sequence length

---

### Prefill Phase Behavior

- K and V are computed for all prompt tokens
- Entire KV Cache is initialized in one pass

Memory footprint scales with:

```
B × S × Nh × Dh
```

---

### Decode Phase Behavior

- At each step, only the new token's K and V are computed
- New K and V are appended to the KV Cache
- Queries attend to the full cached history

This introduces:
- Read-heavy access to KV Cache
- Strict temporal dependency across decode steps

---

## Attention Compute vs Memory Characteristics

| Component | Dominant Cost | Scaling Factor |
|---------|---------------|----------------|
| QKV projection | Compute | B × S × H |
| Score computation | Compute + Memory | B × Nh × S^2 |
| Softmax | Compute | B × Nh × S^2 |
| Value aggregation | Memory | B × Nh × S^2 |
| KV Cache | Memory | B × S × Nh × Dh |

During long-context inference, memory access often dominates compute.

---

## Implications for Dependency Structure

The attention module imposes the following invariant constraints:

- Queries at time `t` depend on all keys and values from times `≤ t`
- KV Cache must be logically consistent across all attention heads
- Head-level computations are independent until output projection

These constraints shape the feasible execution order and data placement.

---

## Non-Prescriptive Notes

This document intentionally avoids:
- Recommending any partitioning scheme
- Discussing specific communication primitives
- Proposing performance optimizations

Its role is to define **what must be true**, not **what should be done**.

---

## Summary

Self-attention combines:
- Head-level independence
- Token-level causal dependency
- Heavy reliance on KV Cache

These structural properties form the foundation upon which all attention-level parallelism reasoning must be built.

