# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Model Type**: 61-layer Mixture-of-Experts (MoE)
- **Layer Structure**: First three layers are dense, followed by MoE layers
- **Expert Architecture**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: BF16
- **Batch Size**: Variable batch size
- **Sequence Length**: Variable sequence length
- **Token Dimension**: 7168
- **Multi-Head Attention (MHA)**:
  - Number of heads: 128
  - Dimension of each head: 128
- **MLP Hidden Size**: 2048

### Hardware Environment
- **GPUs**: Ample H100 GPU resources, no limits
- **Single-card Computing Power**: 400TFlops
- **MFU (Machine FLOPS Utilization)**: 60%
- **VRAM Bandwidth**: 1.8TBps
- **Bandwidth Utilization**: 80%
- **Single-card Video Memory Capacity**: 64GB

### Experimental Setting
- **Evaluation Mode**: Inference-only setting
- **Parallel Strategy**: Large-scale cross-node expert parallelism
- **Deployment Environment**: High-performance computing (HPC) cluster with H100 GPUs

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism

#### GPU Allocation Strategy
- **GPUs Used**: Adequate GPUs (one GPU per expert per layer)
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Expert Distribution**: Experts distributed across nodes to maximize computational parallelism

#### Routing Mechanism
- **Dynamic Routing**: Input tokens are dynamically routed to the GPU holding the corresponding expert
- **Asynchronous Transfer**: Token batches are asynchronously sent to ensure minimal idle time
- **Load Balancing**: Continuous monitoring and adjustment of token distribution

#### Parallel Execution
- **Expert-Level Parallelism**: All experts per layer compute in parallel
- **Cross-Node Communication**: Tokens transferred between nodes as needed
- **Communication Overlap**: Computation and communication overlapped to maximize throughput
- **Token Latency Minimization**: Optimized routing to minimize token processing latency

## Key Experimental Parameters

### Large EP Configuration
- **Expert Parallelism (EP) Degree**: ≥16 (qualifying as "large EP")
- **Maximum Experts per GPU**: 1 (strict enforcement)
- **Total Experts per Layer**: Distributed across available GPUs
- **Node Distribution**: Topology-aware placement

### Performance Metrics
- **Throughput Maximization**: Primary objective through parallel expert computation
- **Token Latency Minimization**: Secondary objective through optimized routing
- **Resource Utilization**: Full GPU compute utilization with 60% MFU
- **Communication Efficiency**: 80% bandwidth utilization

### Scalability Features Tested
- **Large-Scale Deployment**: Adequate H100 GPU resources with no artificial limits
- **Cross-Node Scaling**: Distribution across multiple nodes
- **Dynamic Load Balancing**: Real-time adjustment of expert loads
- **Asynchronous Processing**: Non-blocking communication and computation

## Implementation Details

### Token Processing Flow
1. **Input Token Reception**: Tokens received at input layer
2. **Gating Decision**: Top-K experts selected per token
3. **Token Routing**: Tokens routed to appropriate GPU hosting selected expert
4. **Expert Computation**: Each expert processes assigned tokens in parallel
5. **Output Aggregation**: Results collected and forwarded to next layer
6. **Pipeline Continuation**: Process repeats for subsequent layers

### Communication Optimization
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously overlapping with computation
- **Load Monitoring**: Continuous per-expert load monitoring
- **Dynamic Adjustment**: Real-time gating probability adjustment for load balancing

### Memory Management
- **Single Expert per GPU**: Each GPU hosts only one expert per layer
- **Memory Balancing**: Expert replication strategy when E > G (experts > GPUs)
- **Tensor Parallelism Integration**: Optional within-expert TP for very large models
- **Data Parallelism**: Across MoE network replicas for synchronized updates

## Experimental Validation

### Deployment Verification
- **Parallel Execution Confirmed**: All experts per layer execute concurrently
- **Communication Overlap Verified**: Computation and communication successfully interleaved
- **Load Balance Achieved**: Dynamic routing prevents expert overloading
- **Scalability Demonstrated**: Method scales with available GPU resources

### Performance Characteristics
- **Compute Saturation**: Full GPU utilization achieved through one-expert-per-GPU policy
- **Network Optimization**: Communication costs managed through careful scheduling
- **Throughput Maximization**: Parallel expert execution maximizes overall throughput
- **Latency Minimization**: Asynchronous routing reduces token processing latency

## Conclusion from Experiments

The experimental validation demonstrates that the proposed large-scale cross-node expert parallelism method successfully:
1. Maximizes expert-level parallelism through single-expert-per-GPU deployment
2. Achieves balanced load distribution across nodes through topology-aware placement
3. Effectively overlaps communication with computation using asynchronous routing
4. Scales efficiently with available GPU resources in HPC environments
5. Maintains high throughput and low latency for inference workloads

The method provides a scalable blueprint for high-performance MoE inference, particularly effective in environments with abundant GPU resources such as H100 clusters.