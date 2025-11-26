# Phase 2: Methodology Extraction

## Overview
The approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

## Core Methodology Components

### 1. Expert Placement Strategy

#### Single-Expert-Per-GPU Deployment
- For a MoE layer with E experts and a cluster of G GPUs, ensure each expert is assigned to a distinct GPU if E <= G
- If E > G, replicate experts across GPUs to maximize concurrency of independent experts while balancing memory usage
- Each expert processes tokens without contention from other experts on the same device
- Fully utilizes GPU compute units

#### Cross-Node Distribution
- Topology-aware placement strategy considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimize maximum number of tokens sent across any single link
- Maintain one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### Gating Mechanism
- Routing governed by gating network as in standard MoE architectures
- Top-K gating scores determine which experts are activated for each input token

#### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce number of network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts

### 3. Communication Overlap and Scheduling

#### Overlapping Compute and Communication
- Interleave expert computation and communication
- While one batch of tokens is processed on a GPU, next batch is simultaneously transferred from other nodes
- Leverage CUDA streams or asynchronous communication libraries (NCCL or MPI)
- Ensure data transfer does not block GPU computation

#### Pipeline Scheduling
For multi-layer MoE networks:
- Token outputs from previous MoE layer immediately routed to next layer
- Experts in subsequent layers start processing as soon as partial batch arrives, rather than waiting for full batch
- Fine-grained pipeline increases throughput and reduces idle time for each expert

### 4. Scalability Considerations

#### Large EP Regime
- Optimized for large EP setups (16 or more experts per parallel group)
- Network bandwidth becomes primary limiting factor
- Mitigation through topology-aware routing and token batching
- One-expert-per-GPU policy ensures all GPUs fully utilized for compute

#### Memory and Model Parallelism Integration
For very large models that cannot fit on single GPU:
- Each expert can be further partitioned using tensor model parallelism (TP) within its GPU if necessary
- Data parallelism (DP) applied across replicas of MoE network
- Synchronized weight updates while maintaining high expert-level parallelism

## Technical Implementation Details

### Model Architecture
- 61-layer MoE model with first 3 layers dense, remaining 58 layers as MoE
- Each expert is an MLP with hidden size 2048
- Token dimension: 7168
- Multi-head attention: 128 heads, each head dimension 128
- Precision: BF16

### Hardware Configuration
- H100 GPU cluster with adequate resources
- Single-card computing power: 400TFlops
- MFU utilization: 60%
- VRAM Bandwidth: 1.8TBps
- Bandwidth utilization: 80%
- Single-card video memory capacity: 64GB

### Parallel Deployment
- Each GPU hosts exactly one expert per layer
- Total experts per layer distributed across cluster
- Dynamic token routing based on gating mechanism
- Asynchronous token batch transfer between nodes
- All experts per layer compute in parallel

## Key Technical Parameters
- Expert Parallelism (EP) degree: ≥16 (large EP)
- Maximum experts per GPU: 1
- Token batching: Group by destination expert
- Communication overlap: CUDA streams/NCCL/MPI
- Load balancing: Dynamic gating probability adjustment
- Pipeline scheduling: Partial batch processing enabled