# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Concise Version

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce cross-node communication, creating computational bottlenecks. Our method distributes experts across nodes with at most one expert per GPU, prioritizing compute concurrency over communication minimization. This design leverages modern HPC networking to achieve near-linear scaling in large MoE deployments.

## Background

### Mixture-of-Experts Architecture
MoE models replace FFN layers with multiple experts, activating only a subset per input token. A gating mechanism determines expert selection, enabling sparse computation and improved efficiency.

### Large Expert Parallelism (Large EP)
Our approach targets EP ≥ 16, distributing experts across as many devices as possible (ideally one per GPU). This minimizes resource contention while using careful scheduling to manage communication costs.

### Multi-Head Latent Attention (MLA)
MLA reduces memory overhead by storing Key/Value in low-dimensional latent representations instead of matching hidden layer size. This significantly reduces KV cache size while maintaining attention quality.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert per layer
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Expert Assignment**: For E experts and G GPUs, ensure distinct GPU assignment when E ≤ G, replicate experts when E > G

### Routing and Load Balancing
- **Gating**: Top-K scores determine expert activation
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Overlap token transfer with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Use asynchronous transfers while processing tokens
- **Pipeline Scheduling**: Route outputs immediately to next layer, process partial batches
- **Implementation**: CUDA streams, NCCL, or MPI for non-blocking transfers

### Scalability Considerations
- **Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth limits performance
- **Integration**: Compatible with tensor parallelism (TP) within GPUs and data parallelism (DP) across replicas

## Experiments

### Model Configuration
- **Architecture**: 61-layer MoE transformer
- **Layer Distribution**: First 3 dense layers, 58 MoE layers
- **Dimensions**: Token dimension 7168, MLA 128 heads × 56 dimensions, MLP hidden size 18432
- **Precision**: FP8
- **Batch/Sequence**: Variable optimized per deployment

### Hardware Environment
- **GPUs**: H100 resources with no limits
- **Compute**: 400 TFlops per GPU, 60% MFU utilization
- **Memory**: 64GB VRAM per GPU, 1.8TBps bandwidth, 80% utilization

### Deployment Details
- **Expert Distribution**: One expert per GPU per layer
- **Routing**: Dynamic token routing to expert GPUs
- **Communication**: Asynchronous cross-node transfers
- **Parallelism**: Maximum expert-level parallelism with concurrent execution

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. Through topology-aware placement, asynchronous routing, and compute-communication overlap, we achieve near-linear scaling in high-performance environments. This provides a scalable blueprint for future MoE deployments in resource-rich clusters.