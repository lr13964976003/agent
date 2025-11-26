# Phase 1: Keypoints Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Innovation Points

### 1. Large-Scale Cross-Node Expert Parallelism
- Deploy at most one expert per GPU (single-expert-per-GPU deployment)
- Expert Parallelism (EP) degree of 16 or more ("large EP")
- Distribute experts across nodes to maximize computational parallelism
- Shift bottleneck from intra-GPU contention to network communication

### 2. Expert Placement Strategy
- Topology-aware placement considering node-to-node bandwidth and latency
- GPU memory capacity per node consideration
- Expected token routing patterns optimization
- Minimize maximum number of tokens sent across any single link

### 3. Routing and Load Balancing
- Token batching: Group tokens by destination expert
- Asynchronous routing: Send token batches asynchronously
- Dynamic load balancing: Monitor per-expert load and adjust gating probabilities
- Prevent overloading specific experts

### 4. Communication Overlap and Scheduling
- Interleave expert computation and communication
- Use CUDA streams or asynchronous communication libraries (NCCL/MPI)
- Pipeline scheduling for multi-layer MoE networks
- Fine-grained pipeline to increase throughput

## Model Specifications
- 61-layer Mixture-of-Experts (MoE) model
- First three layers are dense, followed by MoE layers
- Token dimension: 7168
- MHA: 128 heads, each head dimension 128
- MLP hidden size: 2048
- Precision: BF16

## Hardware Environment
- H100 GPU resources (adequate, no limits)
- Single-card computing power: 400TFlops
- MFU utilization: 60%
- VRAM Bandwidth: 1.8TBps
- Bandwidth utilization: 80%
- Single-card video memory capacity: 64GB

## Deployment Strategy
- Each GPU hosts exactly one expert per layer
- Input tokens dynamically routed to GPU holding corresponding expert
- Token batches asynchronously sent to minimize idle time
- All experts per layer compute in parallel

## Advantages
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load Across Nodes: Topology-aware placement and dynamic gating
3. Scalable Communication Overlap: Asynchronous token routing
4. Compatibility with Large Models: Integrates with TP and DP for models exceeding single-GPU memory