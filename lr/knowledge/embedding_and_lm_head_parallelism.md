# Embedding and LM Head Module Parallelism Mapping

## Scope

This document defines the **mapping of parallel primitives to the Embedding and LM Head modules** in a decoder-only Transformer, for module-level parallelism reasoning.

It focuses on:
- Token embedding layers (input embedding, positional embedding)
- LM Head (output projection) layer
- Phase-specific constraints (prefill vs decode)
- Interaction with TP/SP/PP/EP/DP

**Out of scope**:
- Attention or FFN layers
- Hardware-specific implementation
- Training-time parallelism

---

## Module Components

1. **Input Embedding Layer**
   - Converts token IDs to dense embeddings
   - Includes positional embeddings
2. **LM Head**
   - Projects hidden states back to vocabulary logits

KV Cache is not directly updated in embedding or LM Head layers, but embeddings feed into subsequent layers and LM Head reads hidden states from last decoder layer.

---

## Applicable Parallel Primitives

### Tensor Parallelism (TP)

- **Input Embedding**: TP can split embedding matrix along embedding dimension
- **LM Head**: TP can split projection matrix along hidden or vocab dimension
- Synchronization occurs at linear operator boundaries

### Sequence Parallelism (SP)

- **Prefill phase only**: SP can partition token sequences across devices
- **Decode phase**: SP is not allowed, single token per step
- Embeddings and LM Head computations must respect token partitioning

### Pipeline Parallelism (PP)

- Embedding layer typically at stage 0
- LM Head layer typically at last stage
- Communication of embeddings to first layer and logits to output stage
- TP/SP/EP can still be applied within the stage

### Expert Parallelism (EP)

- Generally not applicable unless embedding or LM Head are part of MoE, which is rare
- Sparse routing is mostly irrelevant here

### Data Parallelism (DP)

- Batch-level replication across sequences
- Each device holds full embedding and LM Head modules
- Compatible with TP/SP/PP within device boundaries
- Preserves per-sequence processing semantics

---

## Phase-Specific Summary

| Primitive | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| TP        | Allowed | Allowed | Embedding and LM Head linear operators partitionable |
| SP        | Allowed | Not Allowed | Token-dimension partitioning only in prefill |
| PP        | Allowed | Layer-only | Embedding at stage 0, LM Head at final stage |
| EP        | Rarely Applicable | Rarely Applicable | Only if MoE applied to embedding/LM Head |
| DP        | Allowed | Allowed | Batch-level replication across sequences |

---

## Failure-Prone Assumptions

- SP across decode steps is invalid
- TP cannot break embedding or LM Head numerical correctness
- PP cannot overlap decode steps improperly
- DP does not affect per-token embedding or LM Head output semantics

---

## Summary

For Embedding and LM Head modules:
- TP is applicable to embedding/projection matrices
- SP is prefill-only for token sequences
- PP assigns embedding to first stage, LM Head to last stage
- EP is generally not applicable
- DP replicates modules across batch

This mapping ensures **embedding and LM Head module-level parallel strategies are valid, phase-consistent, and compatible with all applicable primitives**.