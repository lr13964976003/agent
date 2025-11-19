# Phase 1: Key Points Extraction

## Abstract (Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points Summary

### Core Problem
- Traditional MoE implementations colocate multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits true expert parallelism
- Need for better scaling in large GPU clusters

### Proposed Solution
- **Large-scale cross-node expert parallelism** strategy
- Deploy **at most one expert per GPU** (one-expert-per-GPU principle)
- **Large EP regime**: EP ≥ 16 experts per parallel group
- Prioritize compute concurrency over communication reduction

### Key Innovations
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic token distribution with asynchronous routing
3. **Communication Overlap**: Interleave computation and communication for minimal latency

### Technical Details
- **Model Specs**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: BF16
- **Input**: 128 sequences × 10000 tokens per sequence
- **Dimensions**: 4096 token dimension, 32 heads × 128 dim per head, 32768 MLP hidden size

### Performance Results
- **Baseline**: TP=8, PP=2 with 16 H100 GPUs (8 experts per GPU)
  - TPS: 120,000 tokens/s
  - TPOT: 8.3ms
- **Proposed**: Cross-node expert parallelism with 16 H100 GPUs (1 expert per GPU)
  - TPS: 450,000 tokens/s
  - TPOT: 2.2ms
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

### Advantages
1. Maximized Expert Parallelism (one expert per GPU)
2. Balanced Load Across Nodes (topology-aware placement)
3. Scalable Communication Overlap (asynchronous routing)
4. Compatibility with Large Models (integrates with TP and DP)

### Deployment Context
- **GPUs**: 16 H100 GPUs
- **Parallelism**: EP=16 (large EP regime)
- **Memory**: Each GPU hosts exactly one expert per layer
- **Communication**: Cross-node token transfers with async routing
- **Scheduling**: Pipeline scheduling with immediate token routing between layers