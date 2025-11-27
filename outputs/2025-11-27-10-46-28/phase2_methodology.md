# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Rule**: For E experts and G GPUs, assign each expert to distinct GPU if E <= G
- **Extension**: If E > G, replicate experts to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node  
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation per token
- Dynamic adjustment of gating probabilities to prevent expert overloading

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust routing

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving**: Process current batch while transferring next batch from other nodes
- **Implementation**: CUDA streams or asynchronous libraries (NCCL/MPI)
- **Goal**: Prevent data transfer from blocking GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer optimization**:
  - Route token outputs immediately to next layer
  - Start processing partial batches rather than waiting for full batch
- **Result**: Increased throughput, reduced expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Primary limiter**: Network bandwidth (mitigated via topology-aware routing)
- **Compute utilization**: All GPUs fully utilized for computation
- **Communication masking**: Overlapped by calculation process

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Partition experts within GPU if needed for large models
- **Data Parallelism (DP)**: Applied across MoE replicas for synchronized weight updates
- **Compatibility**: Maintains high expert-level parallelism while supporting very large models

## 5. Multi-Head Latent Attention (MLA)

### 5.1 Memory Optimization Design
- **Purpose**: Reduce KV cache memory overhead for long sequences
- **Mechanism**: Store Key/Value in low-dimensional latent representations
- **Implementation**: 
  - Compress X → K_latent (dimension << hidden_dim)
  - Each head projects from latent space to K_head
  - Heaviest computation shared among heads

### 5.2 Traditional vs MLA Comparison
- **Traditional**: Q/K/V all have hidden_dim (7168), independent K/V per head
- **MLA**: Shared latent compression, per-head projection from latent space
- **Benefit**: Significant reduction in KV cache size while maintaining attention quality