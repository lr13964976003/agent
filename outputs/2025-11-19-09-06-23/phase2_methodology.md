# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Allocation Logic**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs while maintaining concurrency
- **Benefit**: Minimal contention, full utilization of GPU compute units

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation
- K experts selected per token based on gating scores

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches async to overlap with computation
- **Load Balancing**: Monitor per-expert load, dynamically adjust gating probabilities

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While batch N is processed on GPU, batch N+1 is transferred
  - Uses CUDA streams or async communication (NCCL/MPI)
  - Data transfer never blocks GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer Coordination**:
  - Token outputs immediately routed to next layer's experts
  - Subsequent layers start processing partial batches immediately
  - Fine-grained pipeline increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth**: Primary limiting factor
- **Mitigation**: Topology-aware routing + token batching
- **Compute Saturation**: All GPUs fully utilized via one-expert-per-GPU policy

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Partition experts within GPU if needed
- **Data Parallelism (DP)**: Replicate MoE network for synchronized updates
- **Memory Management**: Handle models exceeding single-GPU memory

## 5. Mathematical Formulations

### 5.1 Expert Allocation
- E: Number of experts per layer
- G: Number of available GPUs
- Placement constraint: |experts per GPU| ≤ 1

### 5.2 Token Distribution
- Top-K selection: argmax_k(gating_scores)
- Routing function: route(token, expert_id) → destination_GPU
- Load balancing: adjust P(expert|token) based on utilization

## 6. Implementation Details

### 6.1 Hardware Requirements
- **GPUs**: H100-class hardware
- **Interconnects**: NVLink, InfiniBand, NVSwitch
- **Network**: High bandwidth, low latency cross-node communication

### 6.2 Software Stack
- **Communication Libraries**: NCCL, MPI
- **Scheduling**: CUDA streams for async operations
- **Memory Management**: Careful GPU memory allocation per expert

### 6.3 Monitoring and Optimization
- **Metrics**: Per-expert utilization, network traffic patterns
- **Dynamic Adjustment**: Real-time load balancing based on monitoring
- **Topology Optimization**: GPU placement optimization algorithms