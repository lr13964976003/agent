# Helix Parallelism: Key Points Extraction (Phase 1)

## Core Problem Addressed
- **Ultra-long context inference**: Handling multi-million-token KV histories (contexts) while maintaining millisecond-level Token-to-Token Latency (TTL) for real-time applications
- **Two fundamental bottlenecks**: 
  1. KV cache reads during self-attention (memory bandwidth bound)
  2. FFN weight reads during autoregressive decoding (memory bandwidth bound)

## Key Innovations
1. **Hybrid execution strategy**: Decouples attention and FFN parallelism
2. **KV parallelism (KVP)**: Shards KV cache across sequence dimension during attention
3. **Temporal pipeline**: Same GPUs reused for both attention and FFN phases with different sharding strategies
4. **Helix HOP-B**: Batch-wise communication-computation overlap to minimize exposed communication cost

## Technical Architecture
- **Attention phase**: Uses N = KVP × TPA GPUs (TPA ≤ K) for KV partitioning
- **FFN phase**: Reuses same N GPUs for either:
  - Dense models: TP across all N GPUs
  - MoE models: TP × Expert Parallelism grid
- **Communication**: Lightweight All-to-All exchange after attention, overlapped with computation

## Performance Gains
- **DeepSeek-R1**: Up to 1.5× TTL reduction, 32× larger batches under same latency budget
- **Llama-405B**: 1.13× improvement in interactivity, 4× higher throughput and batch capacity
- **HOP-B impact**: Recovers up to 12% degraded performance from exposed communication latency

## Architecture Support
- **Dense models**: Works with GQA/MLA attention
- **MoE models**: Compatible with Expert Parallelism
- **Hardware**: Optimized for Blackwell (GB200 NVL72) with large NVLink domains

## Key Constraints Addressed
- **TP limitation**: Traditional tensor parallelism fails when TP > K (KV heads) due to KV duplication
- **Memory balancing**: Staged KV concatenation ensures uniform distribution across KVP ranks
- **Exact attention**: Preserves exact attention behavior without approximation

## Scalability Features
- **Sequence length**: Handles million+ token contexts efficiently
- **Batch size**: Enables 32× larger concurrent user batches
- **GPU count**: Effective across 1-64 GPU configurations
- **Precision**: Support for FP4 low-precision inference