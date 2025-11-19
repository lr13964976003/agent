# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. Traditional MoE implementations colocate multiple experts on single GPUs to reduce communication, creating computational bottlenecks that limit scaling. We present a cross-node expert parallelism method deploying at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to maximize concurrent computation while leveraging modern HPC networking capabilities.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity
- **Large EP Regime**: EP ≥ 16 experts per parallel group

### Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert
- **Asynchronous Routing**: Send token batches async to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load, adjust gating probabilities

### Communication Overlap and Scheduling
- **Compute-Communication Interleaving**: Process batch N while transferring batch N+1
- **Pipeline Scheduling**: Immediate token routing between layers
- **CUDA Streams**: Async operations using NCCL/MPI

### Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Partition experts within GPU when needed
- **Data Parallelism (DP)**: Replicate MoE network for synchronized updates

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer (MLP experts)
- **Precision**: BF16
- **Input**: 128 sequences × 10,000 tokens
- **Dimensions**: 4096 token dim, 32×128 MHA heads, 32768 MLP hidden
- **Hardware**: 16 H100 GPUs (inference-only)

### Configurations

#### Baseline (TP=8, PP=2)
- **GPUs**: 16
- **Expert Allocation**: 8 experts per layer per GPU
- **Tensor Sharding**: 1/8 shard per GPU
- **Pipeline**: 2 stages, 8 GPUs each

#### Proposed (Large EP)
- **GPUs**: 16
- **Expert Allocation**: 1 expert per layer per GPU
- **Parallelism**: EP=16, TP=1, PP=1
- **Routing**: Cross-node dynamic routing

### Results
| Method | GPUs | TPS | TPOT |
|--------|------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 120,000 | 8.3ms |
| Proposed (EP=16) | 16 | 450,000 | 2.2ms |

**Improvement**: 3.75× higher throughput, 3.8× lower latency

## Conclusion
Our large-scale cross-node expert parallelism maximizes expert-level parallelism (one expert per GPU) while mitigating communication overhead through async token routing and topology-aware placement. Demonstrating 3.75× throughput improvement and 3.8× latency reduction on 16 H100 GPUs, this approach provides a scalable blueprint for high-performance MoE inference in environments with abundant GPU resources.