# Phase 2: Methodology Extraction

## Method Overview

The methodology consists of three key components working together to achieve large-scale expert parallelism for MoE models:

1. Expert Placement Strategy
2. Routing and Load Balancing
3. Communication Overlap and Scheduling

## 1. Expert Placement Strategy

### Core Principle: Single-Expert-Per-GPU
- **Implementation**: Each GPU hosts at most one expert per layer
- **Mathematical Constraint**: For E experts and G GPUs, ensure each expert maps to a distinct GPU when E ≤ G
- **Replication Strategy**: When E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage

### Cross-Node Distribution Algorithm
- **Topology Awareness**: Considers node-to-node bandwidth, latency, GPU memory capacity
- **Objective Function**: Minimize maximum tokens sent across any single link
- **Constraint**: Maintain one-expert-per-GPU principle
- **Implementation**: Static placement based on cluster topology and expected routing patterns

## 2. Routing and Load Balancing

### Gating Mechanism
- **Standard MoE Routing**: Top-K gating scores determine expert activation per token
- **K Value**: Typically K=2 for top-2 routing (standard in MoE)
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities

### Token Sharding Process
```
1. Token Batching:
   - Group tokens by destination expert
   - Reduce network message count
   - Batch size optimization for network efficiency

2. Asynchronous Routing:
   - Send token batches independently
   - Overlap with expert computation
   - Non-blocking communication primitives

3. Load Balancing:
   - Real-time monitoring of expert utilization
   - Dynamic probability adjustment in gating network
   - Prevent straggler experts
```

## 3. Communication Overlap and Scheduling

### Compute-Communication Overlap Architecture
- **CUDA Streams**: Separate streams for computation and communication
- **NCCL/MPI**: Asynchronous communication libraries
- **Buffer Management**: Double/triple buffering for seamless overlap

### Pipeline Scheduling Details
```
Layer-wise Pipeline:
┌─────────────────┬─────────────────┬─────────────────┐
│   Layer 1       │   Layer 2       │   Layer 3       │
├─────────────────┼─────────────────┼─────────────────┤
│ Expert 1-16     │ Expert 1-16     │ Expert 1-16     │
│ GPU 1-16        │ GPU 1-16        │ GPU 1-16        │
│ (Parallel)      │ (Parallel)      │ (Parallel)      │
└─────────────────┴─────────────────┴─────────────────┘

Token Flow:
1. Input tokens arrive at Layer 1
2. Tokens routed to 16 experts (GPU 1-16)
3. While Layer 1 computes, Layer 2 receives tokens
4. Continuous pipeline across all 16 layers
```

### Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within each expert if single GPU memory insufficient
- **Data Parallelism (DP)**: Across model replicas for synchronized weight updates
- **Memory Budget**: Expert parameters + activations must fit within single GPU memory

## 4. Large EP Regime Optimization (EP ≥ 16)

### Network Bandwidth Considerations
- **Primary Bottleneck**: Network bandwidth when EP ≥ 16
- **Mitigation Strategy**: 
  - Topology-aware routing
  - Token batching optimization
  - Communication overlap

### Compute Saturation
- **Goal**: Maximize GPU utilization through expert independence
- **Trade-off**: Communication overhead vs compute concurrency
- **Break-even Point**: Determined by network bandwidth and compute ratio

## 5. Method Summary

### Advantages Delivered
1. **Maximized Expert Parallelism**: Zero intra-GPU expert contention
2. **Balanced Load**: Topology-aware placement prevents hotspots
3. **Scalable Communication**: Near-linear scaling for EP ≥ 16
4. **Memory Compatible**: Integrates with TP/DP for large models

### Technical Implementation Flow
```
Pre-processing:
1. Cluster topology analysis
2. Expert placement optimization
3. Routing table generation

Runtime:
1. Token gating and routing
2. Asynchronous token transfer
3. Expert computation
4. Result aggregation
5. Pipeline progression
```