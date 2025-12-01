# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting true expert parallelism. Our cross-node expert parallelism method distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## 2. Methods

### 2.1 Expert Placement Strategy

**Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU to maximize expert-level parallelism. For E experts and G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G. If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage.

**Cross-Node Distribution**: Use topology-aware placement considering node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns. Minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle.

### 2.2 Routing and Load Balancing

**Gating Mechanism**: Use standard top-K gating scores to determine which experts are activated for each input token.

**Token Sharding Across Nodes**: 
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation  
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 2.3 Communication Overlap and Scheduling

**Overlapping Compute and Communication**: Interleave expert computation and communication using CUDA streams or asynchronous libraries (NCCL/MPI) to prevent data transfer blocking GPU computation.

**Pipeline Scheduling**: In multi-layer MoE networks, ensure token outputs from previous layers are immediately routed to next layer's experts, with subsequent layers starting processing as soon as partial batches arrive.

### 2.4 Large EP Regime (EP ≥ 16)

Network bandwidth becomes the primary limiting factor in large EP setups. Mitigate through topology-aware routing and token batching. The one-expert-per-GPU policy ensures all GPUs are fully utilized for compute while communication costs are amortized across many tokens.

## 3. Experiments

### 3.1 Experimental Setup

**Model Configuration**:
- Architecture: 16-layer MoE with 16 experts per layer
- Expert Type: Multi-Layer Perceptron (MLP)
- Precision: BF16
- Batch Size: 128 sequences
- Sequence Length: 10,000 tokens
- Token Dimension: 4096
- MHA: 32 heads, 128 dimensions per head
- MLP Hidden Size: 16384

**Environment**: Inference-only setting using adequate H100 GPUs

### 3.2 Deployment Configurations

**Baseline (TP=8, PP=2)**:
- Tensor Parallelism = 8, Pipeline Parallelism = 2
- Each GPU holds tensor-parallel shard for all layers
- Multiple experts colocated on same GPU
- Tokens flow sequentially through pipeline stages

**Proposed Cross-Node Expert Parallelism**:
- Expert Parallelism = 16 (large EP regime)
- Each GPU hosts exactly one expert per layer
- Input tokens dynamically routed to GPU holding corresponding expert
- Token batches asynchronously sent to minimize idle time

### 3.3 Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | adequate | TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | adequate | 1 expert per layer per GPU | 450,000 | 2.2 |

**Performance Improvements**:
- Throughput: 3.75× higher (450,000 vs 120,000 TPS)
- Latency: 3.8× lower (2.2ms vs 8.3ms TPOT)

### 3.4 Key Findings

1. **Expert Isolation**: One expert per GPU eliminates intra-GPU contention
2. **Parallel Efficiency**: All 16 experts compute simultaneously  
3. **Communication Overlap**: Asynchronous routing prevents waiting
4. **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

## 4. Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By shifting the computational bottleneck from intra-GPU contention to communication (effectively mitigated through asynchronous token routing and topology-aware placement), we achieve 3.75× higher throughput and 3.8× lower latency compared to baseline configurations.

The approach provides a scalable blueprint for high-performance MoE inference in environments with abundant GPU resources, demonstrating that distributing experts across GPUs and overlapping communication with computation can dramatically improve performance for large-scale MoE deployments.

## 5. Technical Specifications for Deployment

**Critical Parameters**:
- Expert count must equal or exceed GPU count for optimal performance
- Network infrastructure must support high-bandwidth cross-node communication
- Batch size: 128 sequences, Sequence length: 10,000 tokens
- Model dimensions: Token dimension = 4096, MLP hidden size = 16384

**Parallel Strategy Requirements**:
- Expert Parallelism ≥ 16 for large EP regime
- Topology-aware expert placement
- Asynchronous token routing with batching
- CUDA streams for compute-communication overlap

**Hardware Requirements**:
- H100-class GPUs with sufficient memory per expert
- High-performance interconnects (NVLink, InfiniBand)
- Topology-aware scheduling for optimal expert placement