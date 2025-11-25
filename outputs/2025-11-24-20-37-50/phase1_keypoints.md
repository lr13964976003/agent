# Phase 1: Key Points Extraction

## Core Contribution
- **Novel two-level partitioning method** for multi-head attention in large transformer models
- Combines **head-level partitioning** with **intra-head dimension-level partitioning** for improved scalability

## Key Innovations
1. **Dual-level slicing**: Divides MHA into `m × n` partitions where:
   - `n`: number of head partitions (head groups)
   - `m`: number of dimension slices per head
2. Enables deployment on up to `m × n` devices, exceeding traditional head-wise splitting limits
3. Reduces communication overhead through localized computation

## Technical Specifications
- **Input tensor**: `X ∈ ℝ^(B×L×D)` where B=batch size, L=sequence length, D=embedding dimension
- **Head configuration**: D = h × d (h heads, d dimension per head)
- **Partition sizes**:
  - Heads per group: `h_g = h/n`
  - Slice dimension: `d_s = d/m`
- **Weight matrices**: Each W_Q, W_K, W_V partitioned into `ℝ^(d_s·h_g × d_s·h_g)` blocks

## Advantages
- **Scalability**: Supports `m×n` devices vs traditional limit of h heads
- **Load Balancing**: Even workload distribution across both head count and feature dimension
- **Memory Efficiency**: Each device stores fraction of MHA parameters and activations
- **Communication Efficiency**: Localized partitions reduce cross-device synchronization

## Experimental Validation
- **Setup**: 16 NVIDIA H100 GPUs, FP16 precision
- **Model**: 4-layer Dense Transformer
- **Fixed parameters**: Batch=128, SeqLen=10000, Heads=32, HeadDim=128, MLP Hidden=16384
- **Results**: 31.7% throughput improvement (1.2M→1.58M tokens/sec), 37.1% overhead reduction
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)