# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## 2. Methods

### 2.1 Expert Placement Strategy

**Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert per layer. For E experts and G GPUs:
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated to maximize independent concurrency

**Cross-Node Distribution**: Topology-aware placement considering node bandwidth, latency, GPU memory, and routing patterns to minimize network hotspots.

### 2.2 Routing and Load Balancing

**Token Sharding**: 
- Group tokens by destination expert to reduce network messages
- Asynchronous routing overlapping computation and communication
- Dynamic load balancing through gating probability adjustment

**Communication Overlap**:
- CUDA streams/NCCL for asynchronous token transfer
- Pipeline scheduling for multi-layer MoE networks
- Immediate routing between layers to reduce idle time

### 2.3 Scalability Framework

**Large EP Regime (EP ≥ 16)**: Optimized for configurations with 16+ experts per parallel group, where network bandwidth becomes the primary constraint rather than compute contention.

**Integration**: Compatible with tensor parallelism (within experts) and data parallelism (across replicas).

## 3. Experiments

### 3.1 Setup

**Model**: 4-layer MoE, 16 experts/layer, MLP experts
**Configuration**: FP16, 1024 sequences/batch, 10000 tokens/sequence, 8192 token dimension, 16×512 MHA, 32768 MLP hidden size
**Hardware**: 16 H100 GPUs, inference-only
**Metrics**: TPS (tokens/second), TPOT (ms/token)

### 3.2 Deployments

**Baseline (TP=8, PP=2)**:
- 16 GPUs total
- 8 experts per GPU per layer
- Tensor parallelism across 8 GPUs per stage
- 2 pipeline stages

**Proposed (EP=16)**:
- 16 GPUs total
- 1 expert per GPU per layer
- Pure expert parallelism
- Asynchronous cross-node routing

### 3.3 Results

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline | 16 | 8 experts/GPU + TP | 120,000 | 8.3ms |
| **Proposed** | 16 | 1 expert/GPU | **450,000** | **2.2ms** |

**Improvements**: 3.75× throughput gain, 3.8× latency reduction

## 4. Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.