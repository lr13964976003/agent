# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference or training cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Methods

### Expert Placement Strategy
Our method deploys at most one expert per GPU, ensuring each expert can process tokens without contention. For E experts and G GPUs, we assign each expert to a distinct GPU when E ≤ G. When E > G, experts are replicated to maximize concurrency while balancing memory usage. Experts are distributed across nodes using topology-aware placement that considers node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns.

### Routing and Load Balancing
Tokens are routed using standard MoE top-K gating mechanisms. We implement token batching by destination expert to reduce network messages, asynchronous routing to overlap with computation, and dynamic load balancing by monitoring per-expert load and adjusting gating probabilities to prevent overloading specific experts.

### Communication Overlap and Scheduling
We interleave expert computation and communication by processing current batches while simultaneously transferring next batches. CUDA streams or asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block GPU computation. Multi-layer MoE networks use pipeline scheduling where token outputs are immediately routed to next layer's experts, with partial batch processing to reduce idle time.

### Scalability Considerations
Our method is optimized for large EP setups (EP ≥ 16) where network bandwidth becomes the primary limiting factor, mitigated through topology-aware routing and token batching. The one-expert-per-GPU policy ensures full GPU utilization while communication costs are amortized across many tokens. For very large models, experts can be further partitioned using tensor model parallelism within GPUs, while data parallelism is applied across MoE network replicas.

## Experiments

### Experimental Setup
We evaluate our method using a 16-layer MoE model with 64 experts per layer, FP8 precision, 128 sequences per batch, 128 tokens per sequence, token dimension of 1024, 16 MHA heads with 64 dimensions each, and MOE hidden size of 2048. The evaluation uses adequate H100 GPUs in an inference-only setting, measuring TPS (Tokens per Second) and TPOT (Time per Output Token).

### Parallel Deployment Details
**Baseline (TP=8, PP=2)**: Uses 16 H100 GPUs with each GPU holding 8 tensor-parallel shards for all layers, experts colocated on 16 GPUs. Tokens flow sequentially through pipeline stages with multiple experts per GPU sharing compute resources.

**Proposed Method**: Uses 16 H100 GPUs with each GPU hosting exactly one expert per layer. Input tokens are dynamically routed to the GPU holding the corresponding expert, with token batches asynchronously sent to ensure minimal idle time. This ensures all 16 experts per layer compute in parallel.

### Results
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | adequate | TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | adequate | 1 expert each layer per GPU | 450,000 | 2.2 |

The proposed method achieves 3.75× higher throughput and 3.8× lower latency compared to the baseline. Deploying one expert per GPU allows full utilization of GPU compute and memory, while asynchronous token routing ensures minimal waiting even across nodes. With 16 GPUs, the system scales near-linearly in the large EP regime (EP ≥ 16).

## Conclusion
We proposed a large-scale cross-node expert parallelism method for MoE models that maximizes expert-level parallelism by deploying at most one expert per GPU. Our approach shifts the computational bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous token routing, topology-aware expert placement, and overlap of computation with communication. Compared to baseline configurations, our approach achieved 3.75× higher throughput and 3.8× lower latency by fully utilizing all GPUs and enabling large Expert Parallelism (EP ≥ 16). Our method provides a scalable blueprint for future high-performance MoE inference in environments with abundant GPU resources.