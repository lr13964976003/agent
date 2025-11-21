# Phase 1: Key Points Extraction

## Core Problem
- Large neural network models exceed on-chip memory (SRAM/L2 cache) capacity
- Off-chip memory access introduces latency and bandwidth bottlenecks
- Need deployment strategy that maximizes on-chip memory utilization

## Proposed Solution
- **Layer-wise partitioning**: Split n layers into k disjoint groups
- **Cache fitting constraint**: Each partition must fit entirely within SRAM/L2 cache capacity C
- **Contiguous allocation**: Layers assigned in original order to preserve execution flow

## Key Innovation
- Explicit consideration of on-chip memory constraints during model deployment
- Systematic method to estimate memory footprint per layer
- Dynamic allocation based on actual hardware cache capacities

## Memory Footprint Components
1. **Weights**: Parameter tensors (datatype dependent - BF16 = 2 bytes)
2. **Activations**: Intermediate outputs (batch size × sequence length × hidden dimensions)
3. **Temporary Buffers**: Operator workspace memory

## Partitioning Algorithm
- **Greedy approach**: Simple sequential accumulation until cache limit reached
- **Dynamic programming**: Optimized partitioning for balanced load (optional)

## Performance Results
- **20% increase in TPS** (Tokens Per Second)
- **17% reduction in TPOT** (Time Per Output Token)
- Compared to baseline TP=8, PP=2 setup on 16 GPUs

## Hardware Configuration
- 16 NVIDIA H100 GPUs
- BF16 precision
- Models tested: 16-layer dense network with 30B parameters
- Batch size: 128
- Sequence length: 10000
- Head count: 32
- Head dimension: 128
- MLP hidden size: 16384