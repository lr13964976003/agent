# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference or training cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Methods

### Expert Placement Strategy

Our approach deploys at most one expert per GPU, ensuring minimal contention and maximum parallelism. For a MoE layer with E experts and a cluster of G GPUs:
- If E ≤ G, each expert is assigned to a distinct GPU
- If E > G, experts are replicated across GPUs to maximize concurrency

In our experimental configuration with 64 experts per layer and 16 GPUs, each GPU hosts 4 experts distributed across different layers, ensuring balanced memory usage and compute distribution.

### Routing and Load Balancing

Tokens are routed using a standard gating mechanism with top-K expert selection. Our approach includes:
- **Token Batching**: Groups tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Sends token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitors per-expert load and dynamically adjusts gating probabilities

### Communication Overlap and Scheduling

Cross-node token transfers are mitigated through:
- **Compute-Communication Overlap**: While one batch processes, the next transfers simultaneously
- **Pipeline Scheduling**: Token outputs immediately route to next layer's experts
- **Asynchronous Operations**: CUDA streams ensure non-blocking data transfer

## Experiments

### Experimental Setup
- **Model**: 16-layer MoE with 64 experts per layer
- **Precision**: FP8
- **Batch**: 128 sequences of 128 tokens each
- **Hardware**: H100 GPUs with high-performance interconnects

### Deployment Configurations

**Baseline (TP=8, PP=2)**:
- 16 GPUs with tensor parallelism (8 shards per GPU)
- Pipeline parallelism with sequential processing
- Experts colocated and sharing compute resources

**Proposed Cross-Node Expert Parallelism**:
- 16 GPUs with distributed expert placement
- Each GPU hosts 4 experts across different layers
- Asynchronous token routing with minimal idle time

### Results

| Method | TPS (Tokens/s) | TPOT (ms) |
|--------|----------------|-----------|
| Baseline | 120,000 | 8.3 |
| Proposed | 450,000 | 2.2 |

The proposed method achieves 3.75× higher throughput and 3.8× lower latency by eliminating intra-GPU expert contention and maximizing parallel expert computation.

## Conclusion

We proposed a large-scale cross-node expert parallelism method that maximizes expert-level parallelism by distributing experts across GPUs with minimal colocation. Our approach achieves significant performance improvements by shifting the bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous routing and compute-communication overlap. The method demonstrates 3.75× throughput improvement and 3.8× latency reduction, providing a scalable blueprint for high-performance MoE inference in large GPU clusters.

## Technical Specifications for Deployment

- **Model Architecture**: 16 layers, 64 experts per layer
- **GPU Requirement**: Minimum 16 H100 GPUs
- **Expert Distribution**: 4 experts per GPU across different layers
- **Parallel Strategy**: Large EP ≥ 16 with topology-aware placement
- **Communication**: Asynchronous token routing with batching
- **Performance**: 450,000 TPS throughput, 2.2ms latency per token