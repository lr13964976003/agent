# Phase 1: Keypoints Extraction

## Abstract Keypoints
- Novel attention partitioning method for large-scale transformer models
- Divides MHA mechanism into n head groups and m dimension segments per head
- Results in m × n partitions for m × n devices
- Improves scalability and hardware utilization
- Reduces communication overhead and enhances load balancing

## Introduction Keypoints
- Transformer architectures with MHA are cornerstone of state-of-the-art models
- Model sizes growing exponentially, requiring efficient distributed computation
- Traditional MHA parallelization splits attention heads across devices
- Limited when available devices exceed number of heads
- Proposed method extends beyond head-wise splitting by segmenting each head's internal dimension

## Method Keypoints

### Core Innovation
- Two-level partitioning scheme for MHA layers
- Level 1: Head dimension partitioning - h heads divided into n groups
- Level 2: Intra-head dimension partitioning - each head's feature dimension d sliced into m segments
- Total: m × n partitions mapped to m × n devices

### Technical Details
- Input tensor X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding
- h heads, each with dimension d = D/h
- h_g = h/n heads per group
- d_s = d/m slice dimension per partition
- Weight matrices W_Q, W_K, W_V partitioned into blocks W^(i,j)
- Each device computes attention for subset of heads and dimension slice
- Results aggregated through hierarchical concatenation

### Advantages
- Scalability: Supports m × n devices beyond head-wise splitting limits
- Load Balancing: Even division of both head count and feature dimension
- Reduced Memory: Each device stores fraction of parameters and activations
- Communication Efficiency: Localized partitions reduce synchronization bandwidth

## Experiments Keypoints

### Setup
- 16 NVIDIA H100 GPUs
- Mixed precision (FP16)
- 2-layer Dense Transformer model
- Fixed parameters: batch=1024, sequence=10000, heads=16, head_dim=512, MLP_hidden=32768

### Baseline
- Tensor Parallelism (TP) degree 8 + Pipeline Parallelism (PP) degree 2
- Total: 16 GPUs utilized

### Results
- Throughput improvement: 31.7% (1.2M → 1.58M tokens/sec)
- Overhead reduction: 37.1% (TPOT: 0.35ms → 0.22ms)
- Proposed method (m×n=16) outperforms baseline (TP=8, PP=2)

### Key Metrics
- TPS: Tokens processed per second
- TPOT: Time per output token (ms)

## Conclusion Keypoints
- Two-level partitioning enables deployment across m × n devices
- Significant improvements in inference throughput (up to 35%)
- Communication overhead reduced by over 30%
- Better workload balancing and hardware resource utilization
- Promising direction for efficient distributed inference of large transformers