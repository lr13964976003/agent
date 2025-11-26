# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Concise Version

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, conventional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks and limiting expert parallelism. Our method distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to large numbers (≥16) to unlock higher concurrent computation degrees.

## Methods

### 1. Expert Placement Strategy

**Single-Expert-Per-GPU Deployment:**
- For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
- If E > G: replicate experts while maximizing independent expert concurrency
- Each expert processes tokens without contention on same device

**Cross-Node Distribution:**
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimize maximum tokens sent across any single link

### 2. Routing and Load Balancing

**Gating Mechanism:**
- Top-K gating scores determine expert activation per token

**Token Sharding:**
- Token batching: group tokens by destination expert
- Asynchronous routing: send batches asynchronously overlapping computation
- Load balancing: monitor per-expert load and dynamically adjust gating probabilities

### 3. Communication Overlap and Scheduling

**Overlapping Compute and Communication:**
- Interleave expert computation and communication
- Use CUDA streams or NCCL/MPI for async communication
- Ensure data transfer doesn't block GPU computation

**Pipeline Scheduling:**
- Token outputs immediately routed to next layer
- Subsequent layers start processing partial batches
- Fine-grained pipeline increases throughput

### 4. Scalability Considerations

**Large EP Regime:**
- Network bandwidth becomes primary limiting factor
- Topology-aware routing and token batching mitigation
- One-expert-per-GPU ensures full GPU utilization

**Integration with Other Parallelism:**
- Tensor Model Parallelism (TP) within expert if needed
- Data Parallelism (DP) across MoE network replicas
- Synchronized weight updates with high expert-level parallelism

## Experiments

### Model Configuration
- **Architecture**: 61-layer MoE (first 3 dense layers, 58 MoE layers)
- **Token Dimension**: 7168
- **Multi-Head Attention**: 128 heads × 128 dimensions
- **MLP Hidden Size**: 2048
- **Precision**: BF16

### Hardware Environment
- **GPUs**: Adequate H100 resources
- **Single-card Power**: 400TFlops at 60% MFU
- **VRAM Bandwidth**: 1.8TBps at 80% utilization
- **Memory**: 64GB per GPU

### Deployment Details
- **Strategy**: One expert per GPU per layer
- **Parallelism**: EP ≥ 16 (large EP)
- **Routing**: Dynamic async routing to GPU holding expert
- **Execution**: All experts per layer compute in parallel
- **Communication**: Token batches asynchronously sent for overlap

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. The approach shifts the bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous token routing, topology-aware placement, and communication-computation overlap. This provides a scalable blueprint for high-performance MoE inference in HPC environments.