# Memory Constraints

## Scope

This document defines the **memory-related constraints** for decoder-only Transformer inference on multi-device systems.

It focuses on:
- GPU/TPU memory limits per device
- Memory footprint of model layers, KV Cache, and intermediate activations
- Integration with TP, SP, PP, EP, DP, microbatching, and overlapping compute/communication

**Out of scope**:
- Training-time memory allocation strategies
- Low-level hardware-specific optimization (e.g., memory allocator internals)

---

## Memory Components

1. **Model parameters**
   - Includes embeddings, attention layers, FFN layers, MoE experts, LM head
   - TP partitions tensors to reduce per-device footprint
2. **KV Cache**
   - Stores K/V matrices for attention layers per token
   - Prefill: multiple tokens increase memory linearly with sequence length
   - Decode: only one token per step, smaller memory footprint
3. **Intermediate activations**
   - Required for layer computation, pipeline stages, and microbatching
   - SP partitions activations across tokens in prefill
   - PP requires communication buffers per stage
4. **Communication buffers**
   - TP: All-Reduce/All-Gather buffers
   - EP: scatter/gather buffers per token
   - DP: output aggregation buffers

---

## Prefill Phase Memory Considerations

- Multiple tokens per sequence lead to large KV Cache
- Microbatching reduces peak memory by splitting batch into smaller chunks
- TP reduces memory per device for weight matrices
- SP reduces activation memory across token partitions
- PP adds stage-wise buffer requirements
- EP introduces sparse expert output buffers

Constraints:
- Sum of model parameters, KV Cache, activations, and communication buffers must not exceed device memory
- Microbatch size and SP partitioning can be adjusted to fit within memory budget

---

## Decode Phase Memory Considerations

- Single token per sequence reduces KV Cache growth
- PP and TP memory requirements remain similar per token
- SP is not applied
- Microbatching across sequences can still be used to reduce peak memory
- EP sparse routing reduces memory usage by only activating selected experts per token

Constraints:
- KV Cache per sequence must fit within device memory
- Communication buffers per stage must not exceed available memory
- Memory for overlapping compute and communication must be considered

---

## Interaction with Other Primitives

| Primitive | Memory Impact | Notes |
|-----------|---------------|-------|
| TP        | Reduces per-device parameter memory | Partition linear/embedding/FFN tensors |
| SP        | Reduces activation memory (prefill only) | Partition tokens across devices |
| PP        | Requires per-stage activation buffers | Buffers must fit within stage memory limits |
| EP        | Sparse buffers per token | Only selected experts allocated per token |
| DP        | Replicates model across devices | Total memory per device includes full model copy |
| Microbatching | Reduces peak memory | Split batch into smaller microbatches |

---

## Failure-Prone Assumptions

- Ignoring KV Cache growth in prefill phase leads to OOM errors
- Misestimating PP stage buffer requirements
- Overlapping compute and communication exceeding memory budget
- Assuming DP replication does not increase per-device memory usage

---

## Summary

Memory constraints define **how model parameters, KV Cache, intermediate activations, and communication buffers fit into device memory**:
- Prefill: memory scales with sequence length and microbatching mitigates peak usage
- Decode: memory per token is smaller, EP sparsity and microbatching help
- TP, SP, PP, EP, DP, and microbatching interact to shape memory footprint

This ensures **Transformer inference DAGs can be executed safely within device memory limits**.