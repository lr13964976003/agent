# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Concise Version)

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
MoE architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit true expert parallelism.

Our approach prioritizes distributing experts across nodes with at most one expert per GPU, pushing EP to 16 or beyond. This shifts optimization from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert per layer
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory capacity
- **Placement Logic**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G; if E > G, replicate to maximize concurrency

### Routing and Load Balancing
- **Gating Mechanism**: Top-K routing (K=2) with dynamic adjustment
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches while overlapping expert computation
- **Load Monitoring**: Dynamic gating probability adjustment to prevent expert overloading

### Communication Overlap and Scheduling
- **Compute-Communication Interleaving**: Process batch N while transferring batch N+1
- **Fine-Grained Pipeline**: Process partial batches immediately without waiting for full completion
- **CUDA Streams**: Asynchronous communication using NCCL/MPI

### Architecture Specifications
- **Model**: 16-layer MoE with 16 experts per layer
- **Expert Type**: MLP-based experts
- **Dimensions**: Token dimension 4096, MLP hidden size 16384
- **Attention**: 32 heads × 128 dimensions per head
- **Precision**: BF16
- **Batch**: 128 sequences × 10000 tokens each

## Experiments

### Setup
- **Environment**: Adequate H100 GPUs in inference-only setting
- **Baseline**: TP=8, PP=2 configuration with shared experts per GPU
- **Proposed**: EP=16 with one expert per GPU per layer

### Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | TP shard per GPU, shared experts | 120,000 | 8.3ms |
| Proposed Cross-Node EP | 16 | 1 expert per GPU per layer | 450,000 | 2.2ms |

### Key Performance Gains
- **3.75× throughput improvement** (450,000 vs 120,000 tokens/second)
- **3.77× latency reduction** (2.2ms vs 8.3ms per token)
- **Linear scaling** demonstrated with EP ≥ 16
- **Full GPU utilization** achieved through dedicated expert allocation

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By shifting from communication reduction to compute maximization, we achieved 3.75× higher throughput and 3.77× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

---

*This concise version retains the original abstract, key methodology, and experimental results while removing detailed background and implementation specifics.*