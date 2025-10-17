# Phase 1: Keypoints Extraction

## Core Problem
The paper addresses the challenge of efficiently deploying large neural network models on hardware with limited on-chip memory (SRAM/L2 cache), where external memory access creates latency bottlenecks.

## Key Innovation
- **Layer-wise Distribution Strategy**: Proposes partitioning model layers across multiple accelerator cards
- **Cache-aware Partitioning**: Ensures each partition fits entirely within SRAM/L2 cache of individual devices
- **Memory Efficiency Optimization**: Minimizes off-chip memory accesses by maximizing on-chip memory utilization

## Technical Approach
1. **Problem Formulation**: Partition n layers into k groups where each group fits cache capacity C
2. **Memory Estimation**: Calculate layer footprint including weights, activations, and temporary buffers
3. **Partitioning Algorithms**: 
   - Greedy layer aggregation (simple and efficient)
   - Dynamic programming for balanced partitions (optional)
4. **Deployment Strategy**: Load entire partitions into SRAM/L2 cache, execute sequentially, transfer outputs between cards only when crossing partition boundaries

## Performance Gains
- **Dense 16-layer model**: 20% increase in TPS (15,360 vs 12,800 tokens/second)
- **Latency reduction**: 17% reduction in TPOT (0.065ms vs 0.078ms per token)
- **Hardware efficiency**: Better utilization of 16 H100 GPUs compared to TP=8, PP=2 baseline

## Key Constraints
- Each partition must fit within single device's SRAM/L2 cache capacity
- Layers must be assigned contiguously in original order
- Batch size of 1024, sequence length of 10000, FP16 precision used in experiments

## Model Architecture Details
- Dense model: 16-layer fully connected network
- Fixed parameters: 16 heads, 512 dimensions per head, MLP hidden size 32768
- Precision: FP16 for all computations