# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures, particularly those employing multi-head attention (MHA), have become the cornerstone of state-of-the-art models in natural language processing and beyond. As model sizes continue to grow exponentially, efficiently distributing their computations across multiple hardware units becomes critical. Traditional MHA parallelization typically involves splitting the attention heads across devices; however, this approach alone can lead to suboptimal utilization and communication bottlenecks when the number of available devices exceeds the number of heads.

In this work, we introduce a novel partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions that can be mapped onto *m × n* devices. This fine-grained partitioning scheme enables more flexible scaling, better memory distribution, and reduced inter-device communication by localizing computations more effectively.

## Method

### Two-Level Partitioning Scheme

Our method partitions the MHA layer along two dimensions:

1. **Head Dimension Partitioning** - The total *h* heads are divided into *n* groups, each containing *h/n* heads
2. **Intra-Head Dimension Partitioning** - Each head's feature dimension *d* is further sliced into *m* segments, each of size *d/m*

This results in *m × n* partitions, where each partition corresponds to a distinct *(head group, dimension slice)* pair.

### Detailed Partitioning

Given:
- *h* = 16 (number of heads)
- *d* = 512 (dimension per head)
- *D* = *h* × *d* = 8192 (total embedding dimension)
- *n* = 4 (head groups)
- *m* = 4 (dimension slices per head)

Each partition handles:
- *h_g* = *h/n* = 4 heads per group
- *d_s* = *d/m* = 128 dimensions per segment

### Computation Flow

Each device handling partition *(i,j)* computes:
- *Q^(i,j) = X W_Q^(i,j)*
- *K^(i,j) = X W_K^(i,j)*
- *V^(i,j) = X W_V^(i,j)*

Where each *W^(i,j) ∈ ℝ^(512×512)* handles a specific head group and dimension slice.

### Aggregation

1. **Dimension concatenation**: Concatenate *m* dimension slices within each head group
2. **Head concatenation**: Concatenate *n* head groups to reconstruct full MHA output

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 2-layer Dense Transformer
- **Precision**: FP16 mixed precision
- **Parameters**:
  - Batch size: 1024
  - Sequence length: 10000
  - Heads: 16
  - Head dimension: 512
  - MLP hidden size: 32768

### Configurations
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: Two-level partitioning (m×n=16)

### Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

### Performance Analysis
- **Throughput improvement**: 31.7% increase (1.2M → 1.58M tokens/sec)
- **Communication overhead reduction**: 37.1% decrease (0.35ms → 0.22ms TPOT)
- **Full hardware utilization**: All 16 GPUs effectively utilized

## Conclusion

Our two-level partitioning method enables efficient deployment of MHA computations across *m × n* devices, achieving substantial improvements in inference throughput while reducing communication overhead. The approach provides a promising pathway for scaling transformer models to larger distributed infrastructures.

## Technical Implementation Details

### Memory Requirements
- Each device stores 1/(m×n) = 1/16 of total parameters
- Parameter distribution: 8192×8192 matrix split into 16×512×512 blocks

### Communication Patterns
- Hierarchical communication reduces synchronization overhead
- Localized dimension partitions minimize cross-device bandwidth
- No additional communication needed for final concatenation with proper placement

### Deployment Parameters
- **m**: 4 (must divide head dimension 512 evenly)
- **n**: 4 (must divide head count 16 evenly)
- **Devices**: Exactly m×n = 16 devices required
- **Precision**: FP16 essential for throughput optimization
- **Batch size**: 1024 ensures GPU saturation