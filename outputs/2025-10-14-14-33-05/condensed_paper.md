# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Deployment

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures with multi-head attention (MHA) face scalability challenges when model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this approach is limited by the fixed number of heads and leads to suboptimal utilization when available devices exceed the number of heads.

We introduce a two-level partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions mapped onto *m × n* devices.

## Method

### Multi-Head Attention Recap

Given input tensor X ∈ ℝ^(B×L×D) where:
- B = batch size = 1024
- L = sequence length = 10000
- D = embedding dimension = 8192

MHA splits D into h heads, each with dimension d = D/h:
- h = 16 heads
- d = 512 dimension per head

### Two-Level Partitioning Scheme

**Parameters:**
- n = 4 (head partitions)
- m = 4 (dimension partitions per head)
- h_g = h/n = 16/4 = 4 heads per group
- d_s = d/m = 512/4 = 128 dimension per slice

**Partitioning:**
Each projection matrix W_Q, W_K, W_V ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] = [1,2,3,4] (head group index)
- j ∈ [1,m] = [1,2,3,4] (dimension slice index)
- Each block: ℝ^(d_s·h_g × d_s·h_g) = ℝ^(512×512)

### Computation Flow

1. **Input Projection**: Each device (i,j) computes:
   - Q^(i,j) = X W_Q^(i,j) ∈ ℝ^(1024×10000×512)
   - K^(i,j) = X W_K^(i,j) ∈ ℝ^(1024×10000×512)
   - V^(i,j) = X W_V^(i,j) ∈ ℝ^(1024×10000×512)

2. **Attention Computation**: Each device computes:
   - Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)
   - Output: ℝ^(1024×10000×512)

3. **Aggregation**:
   - Concatenate dimension slices within each head group
   - Concatenate head groups to reconstruct full output
   - Final output: ℝ^(1024×10000×8192)

### Communication Pattern
- Total devices: 16 (4×4 grid)
- Device (i,j) handles partition (i,j)
- Hierarchical communication reduces overhead

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Model**: 2-layer Dense Transformer
- **Parameters**: 16 heads, 512 dim/head, 32768 MLP hidden size

### Results

| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

**Improvements:**
- Throughput: +31.7% (1.2M → 1.58M tokens/sec)
- Communication overhead: -37.1% (0.35ms → 0.22ms)

## Conclusion

Our two-level partitioning method enables deployment across m×n devices, achieving substantial improvements in inference throughput (31.7%) while reducing communication overhead (37.1%). This approach offers a promising direction for efficient distributed inference of large transformer architectures.