# Phase 1: Keypoints Extraction

## Core Problem
The paper addresses the challenge of deploying large neural network models on hardware with limited on-chip memory (SRAM/L2 cache), where off-chip memory accesses create latency and bandwidth bottlenecks.

## Key Innovation
A novel layer-wise deployment strategy that partitions model layers across multiple accelerator cards, ensuring each partition fits entirely within the SRAM or L2 cache of a single device.

## Main Contributions

### 1. Problem Formulation
- Given: Model with n layers L = {l₁, l₂, ..., lₙ}
- Goal: Partition into k disjoint groups P = {P₁, P₂, ..., Pₖ}
- Constraint: Each group Pᵢ memory footprint S(Pᵢ) ≤ cache capacity C
- Requirement: Preserve execution order (contiguous layer assignment)

### 2. Memory Footprint Estimation
Layer memory size = weight_size + activation_size + buffer_size
- weight_size: Parameters × datatype size (e.g., FP16 = 2 bytes)
- activation_size: Output feature map dimensions × batch size
- buffer_size: Operator workspace requirements

### 3. Partitioning Algorithms
- **Greedy Layer Aggregation**: Sequential grouping until cache limit reached
- **Dynamic Programming**: Optimize for balanced partitions (optional)

### 4. Deployment Strategy
- Load weights and pre-allocate memory within SRAM/L2 cache
- Execute layers sequentially on assigned card
- Minimize inter-card communication (only transfer between partitions)

## Experimental Setup
- Hardware: 16 NVIDIA H100 GPUs
- Models: Dense 16-layer network (30B parameters, BF16 precision)
- Configuration: Batch size 128, sequence length 10000
- Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

## Key Results
- **20% increase in Tokens Per Second (TPS)**: 12,800 → 15,360
- **17% reduction in Time Per Output Token (TPOT)**: 0.078ms → 0.065ms

## Critical Technical Details
- Model dimensions: 32 heads, 128 dim/head, 16384 MLP hidden size
- Memory constraint: Each partition must fit in SRAM/L2 cache
- Performance metric focus: Throughput and latency optimization
- Scalability: Method adapts to varying model sizes and hardware configurations