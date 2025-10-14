# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

Our method prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This unlocks higher degrees of concurrent computation, allowing each expert to run in near isolation while leveraging modern HPC networking capabilities.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Large EP Regime**: EP ≥ 16 for maximum parallelism

### Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Overlap token transfer with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### Communication Overlap and Scheduling
- **Compute-Communication Interleaving**: Use CUDA streams/NCCL for asynchronous transfers
- **Pipeline Scheduling**: Immediate routing between layers with partial batch processing
- **Topology-Aware Routing**: Minimize network congestion through careful placement

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens
- **Token Dimension**: 8192
- **Hardware**: 16 H100 GPUs

### Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 8 experts/GPU + TP shard | 120,000 | 8.3ms |
| Proposed (EP=16) | 16 | 1 expert/GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by dedicating one GPU per expert. With EP ≥ 16, we achieve near-linear scaling and demonstrate significant performance improvements over traditional approaches, providing a scalable blueprint for high-performance MoE inference.