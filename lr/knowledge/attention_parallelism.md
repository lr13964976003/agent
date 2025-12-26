# Attention Module Parallelism Mapping

## Scope

This document defines the **mapping of parallel primitives to the Attention module** in a decoder-only Transformer, for module-level parallelism reasoning.

It focuses on:
- Which parallel primitives are applicable per attention sub-component
- Phase-specific constraints (prefill vs decode)
- Interaction with KV Cache

**Out of scope**:
- Linear layers outside attention
- Hardware-specific implementation
- Training-time parallelism

---

## Attention Module Components

Typical attention module consists of:
1. Q/K/V projections (linear layers)
2. Scaled dot-product attention computation
3. Softmax normalization
4. Output projection

KV Cache is maintained for K/V for autoregressive inference.

---

## Applicable Parallel Primitives

### Tensor Parallelism (TP)

- **Q/K/V linear projections**: TP along hidden dimension is valid
- **Attention score computation**: TP along head or hidden dimension
- **Output projection**: TP along input or output dimension

**Constraints**:
- TP must preserve numerical equivalence
- Synchronization points occur at linear operator boundaries

### Sequence Parallelism (SP)

- **Prefill phase only**: SP can partition token dimension across devices
- **Decode phase**: SP is not allowed, only one token per step
- **Attention**: Token-level SP requires proper K/V sharing for causal mask

### Pipeline Parallelism (PP)

- PP can assign attention layer to a pipeline stage
- Communication between stages occurs via attention output
- Within a stage, TP/SP/EP can be applied per token as allowed by phase

### Expert Parallelism (EP)

- Only relevant if attention is part of an MoE module
- Sparse routing per token to different experts
- Prefill: multiple tokens routed in parallel
- Decode: single token per step, respects causal dependencies

### Data Parallelism (DP)

- Batch-level parallelism across sequences
- Each device has full attention module replica
- Compatible with TP/SP/PP/EP within device boundaries
- Preserves per-sequence KV Cache and causal semantics

---

## KV Cache Interaction

- TP/SP/PP/EP must respect KV Cache structure:
  - K/V matrices remain logically contiguous per sequence and head
  - During decode, only the new token updates KV Cache
- SP requires proper K/V aggregation across token partitions in prefill
- PP requires communication of attention outputs across pipeline stages

---

## Phase-Specific Summary

| Primitive | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| TP        | Allowed | Allowed | Linear layers, attention scores, output projection |
| SP        | Allowed | Not Allowed | Must respect K/V dependencies |
| PP        | Allowed | Layer-only | Stage boundaries define node dependencies |
| EP        | Allowed | Allowed | Sparse experts, per token routing |
| DP        | Allowed | Allowed | Batch-level replication, per-sequence independence |

---

## Failure-Prone Assumptions

- SP across decode steps is invalid
- TP/EP cannot remove causal mask constraints
- PP cannot overlap decode steps
- KV Cache cannot be shared improperly across SP partitions

---

## Summary

For the Attention module:
- TP is the most widely applicable primitive
- SP is prefill-only, token-dimension partitioning
- PP can assign the layer to a stage and communicate outputs
- EP applies only in MoE context
- DP replicates module across batch

This mapping ensures **module-level parallel strategies are valid, phase-consistent, and KV Cache safe**.