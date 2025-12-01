# Phase 2: Methodology Extraction

## Method Overview
The approach maximizes expert-level parallelism in large-scale MoE models by deploying at most one expert per GPU and distributing experts across nodes. This shifts the bottleneck from inter-expert contention to network communication, mitigated through scheduling, routing, and overlapping communication/computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert per layer
- **Implementation**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G
- **When E > G**: Replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU expert contention, fully utilizes GPU compute units

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link
- **Constraint**: Maintain one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation
- Each token routed to K experts based on gating scores

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities
- **Prevention**: Avoids overloading specific experts and network congestion

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: Process current batch while transferring next batch
- **Implementation**: CUDA streams or asynchronous communication (NCCL/MPI)
- **Benefit**: Data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer Processing**: Token outputs immediately routed to next layer's experts
- **Partial Batch Processing**: Experts start processing as soon as partial batch arrives
- **Throughput**: Increases throughput, reduces idle time per expert

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Network Bottleneck**: Bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Compute Utilization**: One-expert-per-GPU ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Partition experts within GPU if necessary
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Synchronization**: Maintained while preserving high expert-level parallelism

## 5. Implementation Parameters

### 5.1 Model Configuration
- Layers: 16 MoE layers
- Experts per layer: 64
- Precision: FP8
- Batch size: 128 sequences
- Sequence length: 128 tokens
- Token dimension: 1024
- MHA: 16 heads × 64 dimensions
- MOE hidden size: 2048

### 5.2 Deployment Specifications
- **Proposed Method**: 16 GPUs, 1 expert per GPU per layer
- **Baseline**: 16 GPUs, TP=8, PP=2 with expert colocation
- **Hardware**: H100 GPUs with adequate availability
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

## 6. Advantages Summary
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load Across Nodes**: Topology-aware placement prevents bottlenecks
3. **Scalable Communication Overlap**: Asynchronous routing for EP ≥ 16
4. **Large Model Compatibility**: Integrates with TP and DP for memory constraints