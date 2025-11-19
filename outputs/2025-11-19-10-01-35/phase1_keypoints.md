# Phase 1: Key Points Extraction

## Key Problem Addressed
- Transformer model sizes growing exponentially require efficient distributed deployment
- Traditional MHA parallelization only splits attention heads across devices (head-wise splitting)
- Limitations: when available devices > number of heads, results in suboptimal utilization and communication bottlenecks

## Novel Solution Proposed
- **Two-level attention partitioning method** for large-scale transformer models
- **Dual-level slicing**: 
  1. Split attention heads into *n* groups
  2. Further partition each head's internal dimension into *m* segments
- Results in *m × n* total partitions that can be independently assigned to *m × n* devices

## Technical Innovation
- **Fine-grained partitioning**: Beyond conventional head-wise splitting to intra-head dimension-wise partitioning
- **Scalability**: Enables deployment on more devices than the number of attention heads
- **Load balancing**: Even distribution across both head count and feature dimensions
- **Communication efficiency**: Reduced cross-device synchronization through localized partitioning

## Key Parameters
- *h*: number of attention heads (fixed at 32 in experiments)
- *d*: dimension per head (fixed at 128 in experiments)
- *n*: number of head partitions
- *m*: number of dimension partitions per head
- Total devices: *m × n* (16 in experiments)

## Key Results
- **31.7% throughput improvement** over baseline (1.2M → 1.58M tokens/sec)
- **37.1% reduction in communication overhead** (TPOT: 0.35ms → 0.22ms)
- Successfully demonstrated on 16 NVIDIA H100 GPUs with dense transformer models

## Model Configuration Used
- 4-layer Dense Transformer (2-layer also mentioned)
- Batch size: 128
- Sequence length: 10000
- Number of heads: 32
- Dimension per head: 128
- MLP hidden size: 32768
- Precision: FP16 mixed precision

## Baseline Comparison
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) = 16 devices
- **Proposed**: m×n=16 partitions mapped directly to 16 devices
- Dense transformer model shows consistent improvement across metrics