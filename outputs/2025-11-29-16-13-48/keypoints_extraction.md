# Phase 1: Keypoints Extraction

## Core Problem Addressed
The paper addresses the challenge of deploying large neural network models on hardware with limited on-chip memory (SRAM/L2 cache), where accessing external DRAM creates latency and bandwidth bottlenecks.

## Key Innovation
**Layer-wise Partitioning Strategy**: A novel deployment approach that partitions model layers across multiple accelerator cards, ensuring each partition fits entirely within the SRAM or L2 cache of a single device.

## Main Contributions

1. **Memory-Conscious Partitioning**: Explicit consideration of on-chip memory constraints during layer allocation
2. **Systematic Memory Estimation**: Analytical procedure to estimate memory footprint including weights, activations, and temporary buffers
3. **Contiguous Layer Assignment**: Preserves execution order by assigning layers contiguously
4. **Cache-Fit Guarantee**: Ensures each partition fits within target hardware cache capacity

## Technical Approach

### Memory Footprint Calculation
```
size(l_j) = weight_size(l_j) + activation_size(l_j) + buffer_size(l_j)
```

### Partitioning Algorithms
- **Greedy Layer Aggregation**: Simple iterative approach adding layers until cache limit
- **Dynamic Programming**: Optional balanced partitioning to minimize partition count

## Performance Gains
- **20% increase** in Tokens Per Second (TPS)
- **17% reduction** in Time Per Output Token (TPOT)
- Achieved on 4-layer dense model with 16 GPUs

## Critical Implementation Details
- Handles edge cases where single layer exceeds cache capacity
- Supports batch size tuning to reduce activation memory
- Minimizes inter-card communication by keeping intermediate outputs local
- Uses BF16 precision with 30B model weight size

## Hardware Configuration
- Platform: 16 NVIDIA H100 GPUs
- Comparison baseline: TP=8, PP=2 configuration
- Test setup: batch size 128, sequence length 10000