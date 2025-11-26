# Phase 1: Keypoints Extraction

## Abstract Keypoints
- Large-scale cross-node expert parallelism strategy for MoE models
- Deploy at most one expert per GPU (vs conventional multiple experts per GPU)
- Expert Parallelism (EP) >= 16 ("large EP")
- Maximizes computational parallelism and reduces expert-level contention
- Effective in HPC and large GPU cluster environments

## Introduction Keypoints
- MoE architectures scale LLMs while maintaining computational efficiency
- Traditional MoE parallelization assigns multiple experts per GPU (reduces communication but creates bottlenecks)
- Proposed method: distribute experts across nodes, one expert per GPU maximum
- Shifts optimization from communication reduction to compute concurrency maximization
- Leverages modern HPC networking (NVLink, InfiniBand, NVSwitch)

## Background Keypoints
- MoE: FFN layers replaced by multiple experts, gating mechanism activates subset
- Parallelism strategies: DP, TP, PP, EP
- Standard EP: moderate degree, multiple experts per GPU
- Large EP regime: distribute experts across many devices, one per GPU
- Network bandwidth becomes primary limiting factor in large EP

## Methods Keypoints
### Core Components:
1. Expert Placement Strategy - Assign experts across GPUs/nodes
2. Routing and Load Balancing - Balanced input distribution
3. Communication Overlap and Scheduling - Minimize cross-node transfer impact

### Expert Placement Strategy:
- Single-Expert-Per-GPU: Each GPU hosts exactly one expert per layer
- If E <= G: each expert on distinct GPU
- If E > G: replicate experts to maximize concurrency
- Topology-aware placement considering bandwidth, latency, memory, routing patterns

### Routing and Load Balancing:
- Top-K gating scores determine expert activation
- Token batching: group tokens by destination expert
- Asynchronous routing: send batches asynchronously
- Dynamic load balancing: adjust gating probabilities

### Communication Overlap and Scheduling:
- Overlap compute and communication using CUDA streams/NCCL/MPI
- Pipeline scheduling: immediate routing between layers
- Fine-grained pipeline increases throughput

### Scalability Considerations:
- Large EP: 16+ experts per parallel group
- Network bandwidth limiting factor
- TP within GPU if expert too large
- DP across MoE network replicas

## Experiments Keypoints
### Setup:
- Inference-only evaluation
- Model: 61-layer MoE, first 3 layers dense, then MoE
- Precision: BF16
- Token dimension: 7168
- MHA: 128 heads, 128 dim per head
- MLP hidden size: 2048

### Hardware:
- H100 GPUs (ample resources)
- Single-card: 400TFlops compute, 60% MFU
- VRAM: 64GB capacity, 1.8TBps bandwidth, 80% utilization

### Parallel Deployment:
- Proposed: adequate GPUs (one per expert per layer)
- Each GPU hosts exactly one expert per layer
- Dynamic token routing with asynchronous batch transfer

## Conclusion Keypoints
- Large-scale cross-node expert parallelism maximizes expert-level parallelism
- Shifts bottleneck from contention to communication
- Provides scalable blueprint for HPC MoE inference
- Future: training scenarios, dynamic routing, larger models