# FFN Module Parallelism Mapping

## Scope

This document defines the **mapping of parallel primitives to the Feed-Forward Network (FFN) module** in a decoder-only Transformer, for module-level parallelism reasoning.

It focuses on:
- Applicable parallel primitives per FFN sub-component
- Phase-specific constraints (prefill vs decode)
- Interaction with TP/SP/PP/EP/DP

**Out of scope**:
- Attention layers
- Hardware-specific implementation
- Training-time parallelism

---

## FFN Module Components

Typical FFN module consists of:
1. First linear projection (input to intermediate dimension)
2. Activation function (e.g., GELU)
3. Second linear projection (intermediate back to hidden dimension)

---

## Applicable Parallel Primitives

### Tensor Parallelism (TP)

- **First linear projection**: TP along hidden or intermediate dimension
- **Activation**: TP applied element-wise, inherits partitioning from previous linear layer
- **Second linear projection**: TP along intermediate or output dimension

**Constraints**:
- TP must preserve numerical equivalence
- Synchronization points occur at linear operator boundaries

### Sequence Parallelism (SP)

- **Prefill phase only**: SP can partition token dimension across devices
- **Decode phase**: SP is not allowed, only one token per step
- **FFN**: Token-level SP requires consistent partitioning across both linear layers

### Pipeline Parallelism (PP)

- PP can assign FFN layer to a pipeline stage
- Communication occurs via output activations to the next stage
- Within stage, TP/SP/EP can be applied as allowed by phase

### Expert Parallelism (EP)

- Only relevant if FFN is part of an MoE layer
- Sparse routing per token to different experts
- Prefill: multiple tokens routed in parallel
- Decode: single token per step, respects causal dependencies

### Data Parallelism (DP)

- Batch-level parallelism across sequences
- Each device has full FFN module replica
- Compatible with TP/SP/PP/EP within device boundaries
- Preserves per-sequence KV Cache and phase semantics

---

## Phase-Specific Summary

| Primitive | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| TP        | Allowed | Allowed | Linear layers partitioning, activation inherits TP |
| SP        | Allowed | Not Allowed | Partition tokens only in prefill, consistent across layers |
| PP        | Allowed | Layer-only | Stage boundaries define node dependencies |
| EP        | Allowed | Allowed | Sparse experts, per token routing if FFN is MoE |
| DP        | Allowed | Allowed | Batch-level replication, per-sequence independence |

---

## Failure-Prone Assumptions

- SP across decode steps is invalid
- TP/EP cannot remove causal mask constraints (for FFN in MoE context within attention-connected paths)
- PP cannot overlap decode steps improperly
- DP does not alter per-token or per-layer semantics

---

## Summary

For the FFN module:
- TP is the primary primitive for linear projections and activations
- SP is prefill-only, token-dimension partitioning
- PP assigns the layer to a stage and communicates outputs between stages
- EP applies only if FFN is part of MoE
- DP replicates module across batch

This mapping ensures **FFN module-level parallel strategies are valid, phase-consistent, and compatible with all primitives**.

