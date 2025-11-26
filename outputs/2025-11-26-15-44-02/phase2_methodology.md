# Phase 2: Methodology Extraction

## Method Overview

Our approach maximizes expert-level parallelism in Mixture-of-Experts (MoE) models through large-scale cross-node deployment, fundamentally shifting the optimization focus from communication reduction to compute concurrency maximization.

## Core Methodological Components

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Principle
- **Policy**: Each GPU hosts at most one expert per layer
- **Constraint**: For E experts and G GPUs, ensure E ≤ G for full distribution
- **Replication**: When E > G, replicate experts across GPUs to maximize independent expert concurrency
- **Memory Balance**: Replicate in manner that balances memory usage across devices

#### 1.2 Cross-Node Distribution Algorithm
- **Topology Awareness**: Account for:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU
- **Result**: Experts distributed to prevent node hotspotting

### 2. Routing and Load Balancing System

#### 2.1 Gating Mechanism
- **Input Processing**: Top-K gating scores determine expert activation per token
- **Dynamic Adjustment**: Monitor per-expert load, adjust gating probabilities to prevent overloading

#### 2.2 Token Sharding Strategy
- **Token Batching**: Group tokens by destination expert to minimize network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Continuous monitoring and dynamic probability adjustment

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Interleaving
- **Concurrent Execution**: While GPU processes current batch, next batch transfers simultaneously
- **Async Libraries**: CUDA streams, NCCL, or MPI for non-blocking data transfer
- **Stream Management**: Separate streams for compute and communication

#### 3.2 Pipeline Scheduling for Multi-Layer Networks
- **Immediate Routing**: Token outputs from layer n immediately route to layer n+1
- **Partial Batch Processing**: Experts start processing as soon as partial batches arrive
- **Fine-grained Pipeline**: Reduces idle time and increases throughput

### 4. Scalability Framework Design

#### 4.1 Large EP Regime (EP ≥ 16)
- **Primary Limiting Factor**: Network bandwidth
- **Mitigation Strategy**: Topology-aware routing + token batching
- **Compute Policy**: One-expert-per-GPU ensures full GPU utilization
- **Communication Masking**: Overlapped by calculation process

#### 4.2 Memory and Parallelism Integration
- **Tensor Parallelism**: Partition each expert using TP within GPU when model exceeds memory
- **Data Parallelism**: Apply DP across MoE replicas for synchronized weight updates
- **Hierarchical Parallelism**: DP → EP → TP layering for maximum flexibility

## Technical Implementation Details

### Model Architecture Integration
- **Layer Structure**: 61 total layers (3 dense + 58 MoE)
- **Expert Type**: MLP-based experts with hidden size 2048
- **Token Processing**: 7168-dimensional tokens
- **Attention Configuration**: 128 heads × 128 dimensions per head

### Hardware Interface
- **Compute Capability**: 400TFlops per H100 GPU at 60% MFU
- **Memory Interface**: 1.8TBps VRAM bandwidth at 80% utilization
- **Memory Capacity**: 64GB per GPU
- **Communication**: Optimized for H100-class NVSwitch and InfiniBand fabrics