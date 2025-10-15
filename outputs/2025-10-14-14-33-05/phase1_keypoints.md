# Phase 1: Key Points Extraction

## Core Problem
- Transformer models with Multi-Head Attention (MHA) face scalability challenges when deploying across many devices
- Traditional head-wise partitioning is limited by the fixed number of attention heads (h)
- Cannot efficiently utilize hardware when available devices > number of heads

## Proposed Solution
- **Two-level partitioning method** for MHA layers
- Combines:
  1. **Head-level partitioning**: Split h heads into n groups (h/n heads per group)
  2. **Intra-head dimension partitioning**: Split each head's feature dimension d into m segments (d/m per segment)
- Results in m×n total partitions that can be mapped to m×n devices

## Key Innovations
- First method to combine head-wise and dimension-wise partitioning for MHA
- Enables deployment on m×n devices, exceeding traditional head-count limits
- Reduces communication overhead through localized computation
- Better load balancing across devices

## Technical Details
- Each partition handles (head_group, dimension_slice) pairs
- Weight matrices W_Q, W_K, W_V are partitioned into blocks W^(i,j) where:
  - i: head group index [1,n]
  - j: dimension slice index [1,m]
  - Each block: ℝ^(d_s·h_g × d_s·h_g) where d_s = d/m, h_g = h/n
- Hierarchical aggregation: concatenate dimension slices first, then head groups

## Performance Claims
- 31.7% throughput improvement over baseline (1.2M → 1.58M tokens/sec)
- 37.1% reduction in time per output token (0.35ms → 0.22ms)
- Tested on 16 NVIDIA H100 GPUs with 2-layer Dense Transformer
- Uses FP16 precision, batch size 1024, sequence length 10000

## Model Specifications Used
- 2-layer Dense Transformer
- 16 attention heads (h=16)
- 512 dimension per head (d=512)
- 32768 hidden size for MLP
- Total embedding dimension D = h×d = 16×512 = 8192