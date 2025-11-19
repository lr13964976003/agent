# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures with multi-head attention (MHA) require efficient distributed computing as models grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this leads to suboptimal utilization when available devices exceed the number of heads. We introduce a two-level partitioning strategy that extends beyond head-wise splitting by segmenting each attention head's internal dimension, enabling flexible scaling and better hardware utilization.

## Method

###### Multi-Head Attention Background
- Input tensor: $X \in \mathbb{R}^{B \times L \times D}$ (batch_size × sequence_length × embedding_dim)
- MHA projects to Q, K, V using weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$
- $h$ heads, each with dimension $d = D/h$

### Two-Level Partitioning Scheme

#### Parameters
- $h$: total number of heads (32 in experiments)
- $d$: dimension per head (128 in experiments)
- $n$: number of head partitions
- $m$: number of dimension partitions per head
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition

#### Partitioning Process
1. **Head Dimension Partitioning**: Divide $h$ heads into $n$ groups of $h_g$ heads each
2. **Intra-Head Dimension Partitioning**: Split each head's $d$ dimensions into $m$ segments of $d_s$ dimension each
3. **Total Partitions**: $m \times n$ partitions, each assigned to one device

#### Weight Matrix Partitioning
Each projection matrix $W \in \mathbb{R}^{D \times D}$ is partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1,n]$: head group index
- $j \in [1,m]$: dimension slice index
- Each block: $W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$

#### Computation per Partition
Each device $(i,j)$ computes:
- $Q^{(i,j)} = X W_Q^{(i,j)}$
- $K^{(i,j)} = X W_K^{(i,j)}$
- $V^{(i,j)} = X W_V^{(i,j)}$
- $\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$

#### Result Aggregation
Two-stage concatenation:
1. Concatenate dimension slices within each head group
2. Concatenate all head group outputs
- Final output dimension matches original MHA layer

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: FP16 mixed precision
- **Models**: 4-layer and 2-layer Dense Transformers

### Fixed Parameters
- Batch size: 128
- Sequence length: 10000
- Number of heads: 32
- Dimension per head: 128
- MLP hidden size: 32768

### Baseline vs Proposed
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

### Results
- **31.7% throughput improvement** (1.2M → 1.58M tokens/sec)
- **37.1% communication overhead reduction** (0.35ms → 0.22ms)

## Conclusion

Our two-level partitioning method enables deployment of MHA computations across $m \times n$ devices, significantly improving scalability. Experiments demonstrate substantial throughput improvements (31.7%) and communication overhead reduction (37.1%) compared to tensor and pipeline parallelism baselines.

## Key Deployment Parameters
- Total devices: m × n (16 in experiments)
- Head partitions: n
- Dimension partitions per head: m
- Dimension per partition: d_s = 128/m
- Heads per partition: h_g = 32/n
- Batch size: 128
- Sequence length: 10000