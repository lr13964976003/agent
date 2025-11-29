# Phase 1: Keypoints Extraction

## Core Problem
Large neural networks face memory bottlenecks when deployed on hardware with limited on-chip memory (SRAM/L2 cache), causing frequent expensive off-chip memory accesses.

## Proposed Solution
Layer-wise partitioning and distribution method that splits n layers across multiple accelerator cards, ensuring each partition fits entirely within SRAM/L2 cache capacity C.

## Key Constraints
- Memory footprint S(P_i) ≤ C for each partition P_i
- Full execution order preserved (contiguous layer assignment)
- Minimize number of partitions k while maximizing hardware utilization

## Memory Footprint Components
1. Weights: Parameter tensors (datatype size × parameters)
2. Activations: Intermediate outputs (feature map dimensions × batch size)
3. Temporary Buffers: Workspace memory for operators

## Partitioning Algorithms
1. Greedy Layer Aggregation: Sequential accumulation until capacity reached
2. Dynamic Programming: Optimize for balanced partitions

## Deployment Strategy
- Load weights and pre-allocate memory within SRAM/L2 cache
- Execute layers sequentially on assigned card
- Transfer intermediate outputs only between partitions

## Experimental Results
- Hardware: 16 NVIDIA H100 GPUs
- Model: 16-layer dense network, 30B parameters, BF16 precision
- Configuration: Batch size 128, sequence length 10000, 32 heads × 128 dim, MLP hidden 16384
- Baseline: TP=8, PP=2 (16 GPUs total)
- Performance: 20% TPS increase (12,800 → 15,360), 17% TPOT reduction (0.078ms → 0.065ms)