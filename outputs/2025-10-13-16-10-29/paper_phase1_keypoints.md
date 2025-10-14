# Phase 1: Key Points Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Core Problem
- Traditional MoE implementations colocate multiple experts on a single GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

### Proposed Solution
- **Large-scale cross-node expert parallelism** with at most one expert per GPU
- **Large EP regime**: EP ≥ 16 (16 or more experts per parallel group)
- Prioritizes distributing experts across nodes to maximize compute concurrency
- Shifts optimization focus from reducing communication to maximizing compute concurrency

### Technical Innovation
1. **Expert Placement Strategy**: One expert per GPU deployment
2. **Cross-node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity
3. **Routing and Load Balancing**: Token batching, asynchronous routing, dynamic gating adjustment
4. **Communication Overlap**: Interleaving computation and communication using CUDA streams/NCCL
5. **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

### Performance Claims
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling for EP ≥ 16
- Full GPU utilization with minimal contention

### Experimental Validation
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10,000 tokens
- **Hardware**: 16 H100 GPUs
- **Comparison**: TP=8, PP=2 baseline vs proposed cross-node EP

### Scalability Features
- Compatible with tensor parallelism (TP) and data parallelism (DP)
- Handles models exceeding single-GPU memory
- Effective in HPC environments with high-bandwidth interconnects
- Topology-aware routing for network optimization