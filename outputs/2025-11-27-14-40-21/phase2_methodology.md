# Phase 2: Methodology Extraction

## Method Overview
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

## Three Key Components

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: At most one expert per GPU
- **When E ≤ G**: Each expert assigned to distinct GPU
- **When E > G**: Experts replicated across GPUs to maximize concurrency while balancing memory
- **Benefit**: Minimal contention, full GPU utilization

#### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE**: Top-K gating scores determine expert activation
- **Implementation**: Dynamic routing based on gating network

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert
- **Asynchronous Routing**: Send token batches asynchronously
- **Load Balancing**: Monitor per-expert load and adjust gating probabilities dynamically

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Strategy**: Interleave expert computation and communication
- **Process**: 
  - While batch processes on GPU, next batch transfers simultaneously
  - CUDA streams/NCCL/MPI for non-blocking transfers

#### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs immediately routed to next layer
  - Subsequent layers start processing partial batches
  - Fine-grained pipeline for increased throughput

## Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Primary limiting factor**: Network bandwidth
- **Mitigation**:
  - Topology-aware routing
  - Token batching
  - One-expert-per-GPU policy ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Large model handling**:
  - Tensor model parallelism (TP) within GPU if needed
  - Data parallelism (DP) across MoE network replicas
  - Synchronized weight updates maintaining expert-level parallelism

## Technical Implementation Details

### Multi-Head Latent Attention (MLA)
- **Purpose**: Reduce memory overhead in attention mechanism
- **Mechanism**: 
  - Latent projection on Q/K/V
  - KV stored in low-dimensional latent representations
  - Shared latent space among heads
  - Significant reduction in KV cache size

### Memory Optimization
- **Standard Attention**: Q/K/V all have hidden_dim (7168)
- **MLA Approach**:
  - Compress X → K_latent (dimension much smaller than hidden_dim)
  - Each head projects from latent space to individual K_head
  - Heaviest computation moved outside heads and shared

## Parameter Specifications
- **Model Architecture**: 61-layer transformer with MoE
- **Dense Layers**: First 3 layers
- **MoE Layers**: Remaining 58 layers
- **Token Embedding Dimension**: 7168
- **MLA Configuration**: 128 heads, 56-dimensional per head
- **Expert MLP Hidden Size**: 18432
- **Precision**: FP8
- **Batch Size**: Variable (optimized per deployment)
- **Sequence Length**: Variable (context-dependent)