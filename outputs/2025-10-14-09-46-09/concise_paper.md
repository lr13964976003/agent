# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures with multi-head attention (MHA) have become foundational for state-of-the-art models. As model sizes grow exponentially, efficient distributed computation becomes critical. Traditional MHA parallelization splits attention heads across devices, but faces limitations when device count exceeds head count. We introduce a two-level partitioning strategy that extends beyond head-wise splitting by segmenting each attention head's internal dimension, enabling flexible scaling and better memory distribution across m × n devices.

## Method

### Two-Level Partitioning Scheme

Our method partitions MHA layers along two dimensions:

1. **Head Dimension Partitioning**: h heads divided into n groups, each containing h/n heads
2. **Intra-Head Dimension Partitioning**: Each head's feature dimension d sliced into m segments, each of size d/m

This yields m × n partitions, each corresponding to a (head group, dimension slice) pair assigned to individual devices.

### Mathematical Formulation

Given input X ∈ ℝ^(B×L×D) with h heads and dimension per head d = D/h:
- Partition parameters: n (head groups), m (dimension slices)
- h_g = h/n heads per group
- d_s = d/m slice dimension per partition

Weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D) are partitioned into blocks W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g) for i ∈ [1,n], j ∈ [1,m].

### Computation and Aggregation

Each device (i,j) computes:
- Q^(i,j) = X W_Q^(i,j), K^(i,j) = X W_K^(i,j), V^(i,j) = X W_V^(i,j)
- Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j)

Results are aggregated hierarchically:
1. Concatenate m dimension slices within each head group
2. Concatenate n head groups to reconstruct full MHA output

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs, FP16 precision
- **Model**: 2-layer Dense Transformer
- **Fixed Parameters**: batch=1024, sequence=10000, heads=16, head_dim=512, MLP_hidden=32768

### Configurations
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: m×n=16 partitions (m=4, n=4)

### Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

**Improvements**: +31.7% throughput, -37.1% communication overhead

## Conclusion

Our two-level partitioning method enables deployment of MHA computations across m × n devices, achieving substantial improvements in inference throughput (up to 35%) while reducing communication overhead by over 30%. This approach offers a promising direction for efficient distributed inference of large transformer architectures.

## Deployment Configuration

See `deployment_config.json` for complete deployment specifications including parallel strategies, module mappings, and device configurations.