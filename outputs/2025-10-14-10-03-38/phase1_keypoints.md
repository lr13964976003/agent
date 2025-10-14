# Phase 1: Keypoints Extraction

## Core Problem
- Deploy large-scale neural networks (n layers) across multiple accelerator cards
- Each partition must fit entirely within SRAM or L2 cache capacity C
- Minimize off-chip memory accesses and reduce latency

## Key Contributions
1. **Layer-wise Distribution Strategy**: Novel approach to partition model layers across multiple processing units
2. **Memory-Constrained Partitioning**: Explicit consideration of on-chip memory limits during deployment
3. **Systematic Memory Estimation**: Analytical method to calculate memory footprint including weights, activations, and buffers
4. **Scalable Deployment**: Method works for inference and can be extended to training

## Memory Footprint Components
- **Weights**: Parameter tensors (datatype size × parameter count)
- **Activations**: Intermediate outputs (output dimensions × batch size)
- **Temporary Buffers**: Operator workspace memory

## Partitioning Algorithms
- **Greedy Layer Aggregation**: Simple sequential grouping until cache limit reached
- **Dynamic Programming**: Optimized for balanced partitions and minimized partition count

## Performance Metrics
- **20% improvement** in Tokens Per Second (TPS) over baseline
- **17% reduction** in Time Per Output Token (TPOT)
- Baseline: TP=8, PP=2 on 16 GPUs
- Proposed: Layer-wise partitioning on 16 GPUs

## Hardware Configuration
- 16 NVIDIA H100 GPUs
- Dense model: 16-layer fully connected network
- FP16 precision, batch size 1024, sequence length 10000
- Model dimensions: 16 heads, 512 head dimension, 32768 MLP hidden size