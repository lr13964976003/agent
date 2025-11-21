# Phase 1: Key Points Extraction

## Core Problem
The paper addresses the challenge of efficiently deploying large-scale neural networks on hardware with limited on-chip memory (SRAM/L2 cache), where accessing external memory introduces significant latency and bandwidth bottlenecks.

## Key Innovation
A novel layer-wise deployment strategy that:
- Partitions model layers across multiple accelerator cards
- Ensures each partition fits entirely within SRAM/L2 cache of a single device
- Minimizes off-chip memory accesses while maintaining model execution order
- Maximizes utilization of fast on-chip memory

## Technical Approach
1. **Problem Formulation**: Given n layers, partition into k groups where each group's memory footprint ≤ cache capacity C
2. **Memory Estimation**: Calculate layer footprint as sum of weights + activations + temporary buffers
3. **Partitioning Algorithms**: 
   - Greedy layer aggregation (simple, efficient)
   - Dynamic programming for balanced partitions (optional optimization)
4. **Deployment Strategy**: Each partition loaded entirely into cache, executed sequentially, with inter-card transfers only between partitions

## Experimental Validation
- Hardware: 16 NVIDIA H100 GPUs
- Models: Dense 4-layer network (30B parameters, BF16)
- Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- Results: 20% increase in TPS (15,360 vs 12,800), 17% reduction in TPOT (0.065ms vs 0.078ms)

## Advantages
- Reduced memory access latency
- Improved throughput via parallel execution
- Scalability across different model sizes and hardware configurations
- Explicit consideration of on-chip memory constraints

## Key Technical Parameters
- Model specs: 30B parameters, BF16 precision, batch size 128, sequence length 10000
- Architecture: 32 heads, 128 dimensions per head, 16384 MLP hidden size
- Memory components: weights (30B), activations, temporary buffers