# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Condensed)

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction & Problem Statement

Traditional MoE parallelization assigns multiple experts per GPU to reduce cross-node communication, creating computational bottlenecks that limit expert-level parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

## Proposed Method

### Core Innovation: Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Target**: Expert Parallelism (EP) ≥ 16
- **Benefit**: Maximizes compute concurrency by eliminating intra-GPU expert contention

### Expert Placement Strategy
1. **Assignment Rule**: 
   - If E ≤ G: Each expert to distinct GPU
   - If E > G: Replicated experts while maximizing concurrency
2. **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
3. **Load Balancing**: Dynamic gating adjustment to prevent expert overload

### Communication Optimization
- **Token Routing**: Asynchronous batching by destination expert
- **Overlap**: CUDA streams/NCCL for compute-communication overlap
- **Pipeline**: Fine-grained scheduling across MoE layers

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer MoE
- **Experts**: 16 per layer (MLP type)
- **Dimensions**: 
  - Token: 8,192
  - MLP hidden: 32,768
  - MHA: 16 heads × 512 = 8,192
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens

### Hardware
- **GPUs**: 16 H100
- **Network**: High-bandwidth interconnect (NVLink/InfiniBand)

## Results

| Configuration | TPS | TPOT (ms) | Improvement |
|---------------|-----|-----------|-------------|
| Baseline (TP=8, PP=2) | 120,000 | 8.3 | 1.0× |
| Proposed (EP=16) | 450,000 | 2.2 | 3.75× throughput, 3.8× latency |

### Key Findings
- **Throughput**: 3.75× higher than baseline
- **Latency**: 3.8× lower than baseline
- **GPU Utilization**: Maximal compute efficiency (one expert per GPU)
- **Scaling**: Near-linear for EP ≥ 16

## Technical Implementation

### Memory Requirements
- **Per Expert**: 512 MB (8,192 × 32,768 × 2 bytes FP16)
- **Total System**: 8 GB (16 experts × 512 MB)

### Communication Pattern
- **Baseline**: Tensor parallelism all-reduce + pipeline send/receive
- **Proposed**: Expert-level token routing with asynchronous batching

## Conclusion

The proposed large-scale cross-node expert parallelism achieves significant performance improvements by shifting from communication reduction to compute maximization. The method demonstrates 3.75× throughput and 3.8× latency improvements over traditional approaches, providing a scalable blueprint for high-performance MoE deployments.

## Deployment Configuration

See `deployment_configuration.json` for complete device mapping and parallel strategy specifications for both baseline and proposed configurations.