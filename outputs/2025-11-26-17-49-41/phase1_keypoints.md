# Phase 1: Key Points Extraction

## Core Problem Statement
Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert-level parallelism.

## Proposed Solution
Cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert, pushing Expert Parallelism (EP) to large numbers (EP ≥ 16).

## Key Technical Components

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU Deployment**: Mathematical formulation E ≤ G vs E > G
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory capacity, and routing patterns

### 2. Routing and Load Balancing
- **Token Sharding**: Group tokens by destination expert, asynchronous routing
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### 3. Communication Overlap and Scheduling
- **Compute-Communication Overlap**: CUDA streams, NCCL, MPI implementation
- **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

### 4. Scalability Considerations
- **Large EP Regime**: EP ≥ 16 with network bandwidth as limiting factor
- **Integration with TP/DP**: Tensor parallelism within GPUs, data parallelism across replicas

## Experimental Configuration
- **Model**: 61-layer MoE (3 dense + 58 MoE layers), BF16 precision
- **Dimensions**: Token dimension 7168, 128 MHA heads (128 dims/head), MLP hidden size 2048
- **Hardware**: H100 GPUs (64GB VRAM, 400TFlops, 1.8TBps bandwidth)
- **Targets**: 60% MFU utilization, 80% bandwidth utilization
- **Deployment**: One GPU per expert per layer (1:1 ratio)

## Performance Claims
All experts per layer compute in parallel, maximizing throughput and minimizing token latency through near-linear scaling in large MoE deployments.