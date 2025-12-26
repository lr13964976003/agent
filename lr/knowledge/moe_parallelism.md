# MoE Module Parallelism Mapping

## Scope

This document defines the **mapping of parallel primitives to the Mixture-of-Experts (MoE) module** in a decoder-only Transformer, for module-level parallelism reasoning.

It focuses on:
- Applicable parallel primitives per MoE sub-component
- Phase-specific constraints (prefill vs decode)
- Interaction with TP/SP/PP/EP/DP

**Out of scope**:
- Attention layers
- Standard FFN layers outside MoE
- Hardware-specific implementation
- Training-time specifics (e.g., load balancing, expert dropout)

---

## MoE Module Components

Typical MoE module consists of:
1. Gating function (determines expert selection per token)
2. Multiple expert FFN submodules
3. Aggregation of expert outputs

KV Cache interaction is relevant only if MoE is used inside attention-connected pathways.

---

## Applicable Parallel Primitives

### Tensor Parallelism (TP)

- Applied **within each expert** independently
- Linear projections and activations in expert FFNs can be tensor-parallelized
- Synchronization occurs at linear operator boundaries

### Sequence Parallelism (SP)

- Prefill only: SP can partition tokens across devices
- Must ensure correct K/V routing for attention if MoE experts are part of attention pathways
- Decode phase: SP is not allowed

### Pipeline Parallelism (PP)

- MoE layer can be assigned to a pipeline stage
- Inter-stage communication occurs via aggregated expert outputs
- TP/SP/EP can be applied within the pipeline stage

### Expert Parallelism (EP)

- Core primitive for MoE
- Sparse routing per token to selected experts
- Prefill: multiple tokens routed in parallel
- Decode: single token per step, respecting causal dependencies
- Only selected experts process the token

### Data Parallelism (DP)

- Batch-level replication across devices
- Each device has full MoE module replica
- Compatible with TP/SP/PP/EP within device boundaries
- Preserves per-sequence KV Cache and causal semantics

---

## Phase-Specific Summary

| Primitive | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| TP        | Allowed | Allowed | Within expert linear layers and activations |
| SP        | Allowed | Not Allowed | Token-dimension partitioning in prefill, K/V routing must be correct |
| PP        | Allowed | Layer-only | Stage boundaries define node dependencies, aggregate outputs between stages |
| EP        | Allowed | Allowed | Sparse expert routing per token, deterministic per input |
| DP        | Allowed | Allowed | Batch-level replication, per-sequence independence |

---

## KV Cache Interaction

- KV Cache updates only relevant if MoE is inside attention-connected pathway
- Must preserve per-sequence and per-step integrity
- SP requires aggregation of K/V states across token partitions in prefill
- PP requires output communication across pipeline stages

---

## Failure-Prone Assumptions

- SP across decode steps is invalid
- EP routing cannot bypass phase or causal constraints
- TP cannot break deterministic expert computation
- DP does not change per-token or per-layer semantics

---

## Summary

For MoE modules:
- EP is the primary primitive for expert selection
- TP can be applied within expert FFNs
- SP is prefill-only for token partitioning
- PP assigns MoE layer to pipeline stage and aggregates outputs
- DP replicates module across batch

This mapping ensures **MoE module-level parallel strategies are valid, phase-consistent, and compatible with all primitives**.