# Phase 1: Key Points Extraction - Helix Paper

## Key Points Summary

### Core Innovation
- **Two-level attention partitioning method** for large-scale transformer models
- **Dual-level slicing**: Combines head-level and intra-head dimension-level partitioning
- Results in **m × n partitions** that can be assigned to **m × n devices**

### Technical Breakthrough
- **Beyond conventional head-wise splitting**: Further partitions each attention head's internal dimension
- **Improved scalability**: Enables deployment on more devices than traditional methods
- **Reduced communication overhead**: Localized computations within partitions
- **Enhanced load balancing**: Even distribution across devices

### Method Details
- **Head Dimension Partitioning**: Divide h heads into n groups, each containing h/n heads
- **Intra-Head Dimension Partitioning**: Slice each head's feature dimension d into m segments of size d/m
- **Total partitions**: m × n partitions for flexible scaling

### Performance Results
- **31.7% improvement in throughput** (1.2M → 1.58M tokens/sec)
- **37.1% reduction in communication overhead** (0.35ms → 0.22ms TPOT)
- **16 GPU utilization**: Full utilization of 16 NVIDIA H100 GPUs
- **Model tested**: 2-layer Dense Transformer

### Configuration Parameters
- **Batch size**: 1024
- **Sequence length**: 10000
- **Number of heads**: 16
- **Dimension per head**: 512
- **Hidden size of MLP**: 32768
- **Precision**: FP16 mixed precision

### Baseline Comparison
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: m×n=16 partitions (m=4, n=4 based on 16 GPUs)
- **Hardware**: 16 NVIDIA H100 GPUs