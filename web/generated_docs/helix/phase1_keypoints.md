# Helix: Two-Level Attention Partitioning - Key Points

## Key Innovation
- **Dual-level slicing** for Multi-Head Attention (MHA) layers in transformer models
- Partitions attention heads into **n groups** AND splits each head's feature dimension into **m segments**
- Results in **m × n total partitions** that can be mapped to m × n devices

## Problem Addressed
- Traditional MHA parallelization only splits attention heads across devices
- Limited by fixed number of heads (h)
- Suboptimal utilization when device count > h
- Communication bottlenecks in large clusters

## Technical Solution
1. **Head Dimension Partitioning**: h heads → n groups with h_g = h/n heads per group
2. **Intra-Head Dimension Partitioning**: Each head's d dimensions → m segments with d_s = d/m per segment
3. **Dual Partitioning**: Creates m×n partitions of W_Q, W_K, W_V matrices
4. **Hierarchical Aggregation**: Concatenate dimension slices within each head group, then concatenate head groups

## Key Benefits
- **Scalability**: Deploy on m×n devices (exceeds head-wise splitting limits)
- **Load Balancing**: Even workload distribution across heads and dimensions
- **Memory Efficiency**: Each device stores only fraction of parameters/activations
- **Communication Reduction**: Localized intra-head partitions reduce synchronization

## Experimental Validation
- **Setup**: 16 NVIDIA H100 GPUs, FP16 precision
- **Model**: 2-layer Dense Transformer
- **Parameters**: 1024 batch size, 10000 sequence length, 16 heads, 512 head dim, 32768 MLP hidden size
- **Results**:
  - Baseline (TP=8, PP=2): 1.2M tokens/sec, 0.35ms TPOT
  - Proposed (m×n=16): 1.58M tokens/sec, 0.22ms TPOT
  - **Improvements**: 31.7% throughput increase, 37.1% overhead reduction