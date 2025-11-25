# Helix: Two-Level Attention Partitioning for Large Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures, particularly those employing multi-head attention (MHA), have become the cornerstone of state-of-the-art models in natural language processing and beyond. As model sizes continue to grow exponentially, efficiently distributing their computations across multiple hardware units becomes critical. Traditional MHA parallelization typically involves splitting the attention heads across devices; however, this approach alone can lead to suboptimal utilization and communication bottlenecks when the number of available devices exceeds the number of heads.

In this work, we introduce a novel partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions that can be mapped onto *m × n* devices. This fine-grained partitioning scheme enables more flexible scaling, better memory distribution, and reduced inter-device communication by localizing computations more effectively.

## Method

### Two-Level Partitioning Scheme

Our method partitions the MHA layer along two dimensions:

1. **Head Dimension Partitioning** - divides total `h` heads into `n` groups, each containing `h/n` heads
2. **Intra-Head Dimension Partitioning** - splits each head's feature dimension `d` into `m` segments of size `d/m`

This creates `m × n` partitions total, each corresponding to a unique `(head group, dimension slice)` pair.

### Mathematical Formulation

**Input**: `X ∈ ℝ^(B×L×D)` where B=batch size, L=sequence length, D=embedding dimension

**Partition Parameters**:
- Total heads: `h`
- Head dimension: `d = D/h`
- Head partitions: `n`
- Dimension slices: `m`
- Heads per group: `h_g = h/n`
- Slice dimension: `d_s = d/m`

**Weight Partitioning**:
Each projection matrix `W ∈ ℝ^(D×D)` is partitioned into blocks `W^(i,j)` where:
- `i ∈ [1,n]` indexes head group
- `j ∈ [1,m]` indexes dimension slice
- `W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)`

**Computation per Partition**:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)
Attention^(i,j) = softmax((Q^(i,j)(K^(i,j))^T)/√d_s) V^(i,j)
```

**Output Reconstruction**:
```
Output = Concat_{i=1}^n (Concat_{j=1}^m Attention^(i,j))
```

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 4-layer Dense Transformer
- **Precision**: FP16 (mixed precision)

**Fixed Parameters**:
- Batch size: 128
- Sequence length: 10000
- Heads: 32
- Head dimension: 128
- MLP hidden size: 16384

### Baseline Configuration
- Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- Utilizes all 16 GPUs in traditional configuration

### Results

| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

**Improvements**:
- **Throughput**: +31.7% (1.2M → 1.58M tokens/sec)
- **Overhead**: -37.1% (0.35 → 0.22 ms per token)

## Conclusion

Our two-level partitioning method enables deployment of MHA computations across `m × n` devices, significantly improving scalability beyond traditional head-wise splitting. Experiments demonstrate substantial improvements in inference throughput (31.7%) while reducing communication overhead by 37.1% compared to strong baselines using tensor and pipeline parallelism.

## Key Implementation Details

### Advantages
- **Scalability**: Deploy on `m×n` devices vs traditional limit of h heads
- **Load Balancing**: Even workload distribution
- **Memory Efficiency**: Each device stores fraction of parameters
- **Communication Efficiency**: Reduced cross-device synchronization

### Deployment Requirements
- Choice of `m` and `n` depends on hardware topology and network bandwidth
- Compatible with existing model parallel frameworks
- Supports both training and inference modes
- Optimal for large batch sizes (128+) and FP16 precision