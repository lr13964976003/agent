# Phase 1: Keypoints Extraction - Helix Two-Level Attention Partitioning

## Key Problem
Traditional attention head splitting faces scalability limitations when the number of devices exceeds the number of heads, leading to suboptimal hardware utilization and communication bottlenecks.

## Core Innovation
Proposed **two-level partitioning method** for Multi-Head Attention (MHA) that combines:
1. **Head-level partitioning**: Splitting h heads into n groups (each with h/n heads)
2. **Intra-head dimension partitioning**: Splitting each head's feature dimension d into m segments (each of size d/m)

## Technical Breakthrough
- Results in **m × n partitions** that can be independently assigned to m × n devices
- Enables deployment beyond traditional head-wise splitting limits
- Achieves improved scalability and hardware utilization
- Reduces communication overhead through localized computations

## Key Dimensions (Fixed in Experiments)
- Batch size: 128
- Sequence length: 10000
- Number of heads: 32
- Dimension per head: 128
- Hidden size of MLP: 32768
- Total embedding dimension: D = h × d = 32 × 128 = 4096

## Performance Claims
- **31.7% throughput improvement** over baseline (1.2M → 1.58M tokens/sec)
- **37.1% overhead reduction** (0.35ms → 0.22ms TPOT)
- Tested on 16 NVIDIA H100 GPUs with 4-layer Dense Transformer

## Critical Insight
The method fully exploits hardware by creating m×n=16 partitions, achieving better load balancing and communication efficiency compared to traditional TP=8 + PP=2 baseline.