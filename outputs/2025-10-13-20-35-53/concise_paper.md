# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. However, scaling MoE models across GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert parallelism.

We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing Expert Parallelism (EP) to 16 or beyond. This maximizes concurrent computation by leveraging modern HPC networking capabilities.

## 2. Methodology

### 2.1 Expert Placement Strategy

**Single-Expert-Per-GPU Principle**: Deploy at most one expert per GPU. For E experts and G GPUs, assign each expert to a distinct GPU when E ≤ G. When E > G, replicate experts to maximize independent concurrency while balancing memory.

**Cross-Node Distribution**: Use topology-aware placement considering node-to-node bandwidth/latency, GPU memory capacity, and expected routing patterns to minimize cross-link traffic.

### 2.2 Routing and Load Balancing

**Gating Mechanism**: Standard MoE top-K gating determines expert activation per token.

**Token Sharding**: 
- Group tokens by destination expert to reduce network messages
- Asynchronously send token batches to overlap with computation
- Dynamically adjust gating probabilities to balance per-expert load

### 2.3 Communication Overlap

**Compute-Communication Interleaving**: Use CUDA streams/NCCL for asynchronous transfers while processing current batches.

**Pipeline Scheduling**: Route outputs immediately to next layer experts, starting processing on partial batches to reduce idle time.

### 2.4 Large EP Regime (EP ≥ 16)

Optimized for configurations with 16+ experts per parallel group, where network bandwidth becomes the limiting factor rather than compute contention.

## 3. Experiments

### 3.1 Setup
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10,000 tokens = 10.24M tokens
- **Dimensions**: 8192 token dim, 32768 MLP hidden, 16×512 MHA heads
- **Hardware**: 16 H100 GPUs, inference-only

### 3.2 Deployment Configurations

**Baseline (TP=8, PP=2)**:
- 16 GPUs total
- Each GPU: 1/8 tensor-parallel shard + 8 experts per layer
- Sequential pipeline processing with shared GPU resources

**Proposed Cross-Node Expert Parallelism**:
- 16 GPUs total
- Each GPU: exactly 1 expert per layer
- All 16 experts per layer compute in parallel
- Asynchronous token routing with minimal idle time

### 3.3 Results
| Method | TPS | TPOT |
|--------|-----|------|
| Baseline (TP=8, PP=2) | 120,000 | 8.3ms |
| Proposed (EP=16) | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## 4. Conclusion

Our large-scale cross-node expert parallelism method achieves significant performance improvements by dedicating one expert per GPU and leveraging asynchronous communication. This approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.