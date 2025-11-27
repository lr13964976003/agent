# Phase 1: Keypoints Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Contributions

### 1. Large-Scale Cross-Node Expert Parallelism
- Deploys at most one expert per GPU
- Maximizes expert-level parallelism (EP ≥ 16)
- Shifts bottleneck from contention to communication
- Exploits distributed resources fully

### 2. Expert Placement Strategy
- Single-expert-per-GPU deployment
- Cross-node distribution with topology awareness
- Considers bandwidth, latency, memory capacity
- Minimizes hotspotting on single nodes

### 3. Routing and Load Balancing
- Top-K gating mechanism
- Token batching by destination expert
- Asynchronous routing
- Dynamic load balancing

### 4. Communication Optimization
- Overlapping compute and communication
- Pipeline scheduling across layers
- Asynchronous token routing
- CUDA streams/NCCL/MPI utilization

## Model Specifications
- 61-layer MoE model
- First 3 layers are dense, followed by MoE
- Token dimension: 7168
- MLA: 128 heads, 56 dimensions per head
- MLP hidden size: 18432
- FP8 precision

## Hardware Configuration
- H100 GPUs (adequate resources)
- Single-card computing power: 400TFlops
- MFU utilization: 60%
- VRAM Bandwidth: 1.8TBps
- Bandwidth utilization: 80%
- Single-card video memory: 64GB

## Core Innovation
The method transforms MoE scaling by prioritizing expert distribution across nodes rather than minimizing communication, enabling near-linear scaling in large deployments through careful scheduling and routing strategies.