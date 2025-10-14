# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method prioritizing expert distribution across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methods

### Overview
Our approach maximizes expert-level parallelism through three key components:

1. **Expert Placement Strategy**: Assigning experts across GPUs and nodes
2. **Routing and Load Balancing**: Ensuring balanced input distribution
3. **Communication Overlap and Scheduling**: Minimizing cross-node transfer impact

### Expert Placement Strategy

#### Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Implementation**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G; if E > G, replicate experts while maximizing concurrency
- **Benefit**: Each expert processes tokens without contention from other experts on the same device

#### Cross-Node Distribution
- **Topology-aware placement** considering node-to-node bandwidth/latency, GPU memory capacity, and expected routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

### Routing and Load Balancing

#### Gating Mechanism
Standard MoE gating network determines top-K gating scores per token, activating highest-scoring experts.

#### Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### Communication Overlap and Scheduling

#### Overlapping Compute and Communication
- Interleave expert computation and communication using CUDA streams or NCCL/MPI
- While one batch processes, transfer next batch simultaneously

#### Pipeline Scheduling
- Token outputs immediately routed to next layer's experts
- Subsequent layer experts start processing partial batches without waiting for full completion

### Scalability Considerations

#### Large EP Regime (EP ≥ 16)
- Network bandwidth becomes primary limiting factor
- One-expert-per-GPU ensures full GPU utilization while communication costs amortized
- Benefits most pronounced in high-bandwidth HPC environments

#### Integration with Other Parallelism
- Tensor parallelism (TP) within each expert if exceeding single-GPU memory
- Data parallelism (DP) across MoE network replicas

## Experiments

### Setup
- **Model**: 4-layer MoE with 16 experts per layer (MLP experts)
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens × 8,192 dimensions
- **MHA**: 16 heads × 512 dimensions per head
- **MLP Hidden Size**: 32,768
- **GPUs**: 16 H100 GPUs
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

### Configurations

#### Baseline (TP=8, PP=2)
- 16 GPUs total
- Each GPU holds 1/8 tensor-parallel shard for all layers
- 2 pipeline stages (8 GPUs each)
- 8 experts per layer per GPU (colocated)
- Sequential token processing through pipeline stages

#### Proposed Cross-Node Expert Parallelism
- 16 GPUs total
- Each GPU hosts exactly one expert per layer
- Dynamic token routing to corresponding expert GPU
- Asynchronous token batch transfer
- All 16 experts per layer compute in parallel

### Results
| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 8 experts + TP shard | 120,000 | 8.3ms |
| Proposed | 16 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

### Analysis
- **Baseline limitations**: Intra-GPU contention from colocated experts, pipeline stalls
- **Proposed advantages**: Maximized expert parallelism, dedicated GPU resources per expert, asynchronous operations
- **Scalability**: Near-linear scaling demonstrated for EP ≥ 16 in HPC environments

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU with EP ≥ 16. Achieving 3.75× higher throughput and 3.8× lower latency compared to traditional approaches, this method provides a scalable blueprint for high-performance MoE inference in GPU-rich environments. Future work includes extending to training scenarios and optimizing for even larger expert counts.