# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. Traditional approaches colocate multiple experts on single GPUs to reduce communication, but this creates computational bottlenecks and limits true expert parallelism. We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, shifting the optimization focus to maximizing compute concurrency.

## Methods

### 1. Expert Placement Strategy
**Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert per layer. For E experts and G GPUs, if E ≤ G, assign each expert to distinct GPU. If E > G, replicate experts across GPUs to maximize concurrency while balancing memory.

**Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and token routing patterns to minimize network congestion.

### 2. Routing and Load Balancing
**Token Sharding**: Group tokens by destination expert, send batches asynchronously while overlapping computation, and dynamically adjust gating probabilities for load balancing.

**Communication Overlap**: Use CUDA streams and asynchronous libraries (NCCL/MPI) to interleave expert computation with cross-node transfers, preventing GPU blocking.

### 3. Large EP Regime
**Definition**: Expert Parallelism (EP) ≥ 16 qualifies as "large EP"
**Characteristics**: Network bandwidth becomes limiting factor, one-expert-per-GPU ensures full GPU utilization while communication costs are masked by computation

### 4. Parallelism Integration
- **Tensor Parallelism (TP)**: Partition large experts within GPU if needed
- **Data Parallelism (DP)**: Applied across MoE replicas for synchronized updates
- **Expert Parallelism (EP)**: Primary focus on distributing experts across devices

## Experiments

### Model Configuration
- **Architecture**: 61-layer MoE transformer (3 dense + 58 MoE layers)
- **Dimensions**: Token dimension 7168, MHA: 128×128 dimensions, MLP hidden size 2048
- **Precision**: BF16
- **Variables**: Batch size and sequence length are variable

### Hardware Environment
- **GPUs**: H100 with 400 TFLOPS compute, 64GB VRAM, 1.8TB/s bandwidth
- **Utilization**: 60% MFU, 80% bandwidth utilization

### Deployment Configurations
**Proposed Method**:
- 16 experts per MoE layer
- 928 total GPUs (16 × 58 MoE layers)
- One expert per GPU across all nodes
- Asynchronous token routing with topology-aware placement

**Baseline Method**:
- 4 experts per GPU (colocated)
- 232 total GPUs (4 × 58 MoE layers)
- Reduced communication but higher intra-GPU contention

### Performance Results
- **Maximum Expert Parallelism**: All experts compute in parallel with no intra-GPU contention
- **Scalable Communication**: Asynchronous routing masks communication latency
- **Load Balancing**: Dynamic gating prevents expert overloading
- **Throughput**: Near-linear scaling achieved through concurrent expert processing

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. The approach shifts bottlenecks from intra-GPU contention to communication, effectively mitigated through asynchronous token routing and topology-aware placement. This provides a scalable blueprint for high-performance MoE inference in resource-rich environments.