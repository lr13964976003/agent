# Phase 2: Methodology Extraction

## Expert Placement Strategy

### Single-Expert-Per-GPU Principle
- **Core constraint**: Each GPU hosts at most one expert from any given layer
- **Mathematical formulation**: For E experts per layer and G GPUs, if E ≤ G, each expert gets a unique GPU
- **When E > G**: Experts are replicated across GPUs to maximize concurrency while balancing memory usage
- **Correction for 64 experts, 16 GPUs**: Each GPU hosts 4 experts (64 ÷ 16 = 4), distributed across different layers

### Cross-Node Distribution
- **Topology-aware placement**: Considers node-to-node bandwidth, latency, GPU memory capacity, and expected routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining expert distribution
- **Implementation**: Experts are distributed such that each GPU processes different experts from different layers

## Routing and Load Balancing

### Gating Mechanism
- **Top-K selection**: Standard MoE gating network determines expert activation for each token
- **Dynamic adjustment**: Gating probabilities are monitored and adjusted to prevent expert overloading

### Token Sharding Across Nodes
1. **Token Batching**: Groups tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Sends token batches asynchronously while overlapping expert computation
3. **Load Balancing**: Monitors per-expert load and dynamically adjusts to prevent stragglers

## Communication Overlap and Scheduling

### Overlapping Compute and Communication
- **Interleaving strategy**: While one token batch processes, the next batch transfers simultaneously
- **Implementation**: CUDA streams or asynchronous libraries (NCCL/MPI) ensure non-blocking data transfer

### Pipeline Scheduling
- **Layer-to-layer routing**: Token outputs immediately route to next layer's experts
- **Partial batch processing**: Subsequent layers start processing as soon as partial batches arrive
- **Benefit**: Increases throughput and reduces expert idle time

## Scalability Considerations

### Large EP Regime (EP ≥ 16)
- **Network bandwidth focus**: Communication becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Compute utilization**: One-expert-per-GPU policy ensures full GPU utilization

### Memory and Model Parallelism Integration
- **Tensor parallelism**: Each expert can be partitioned using TP within its GPU if needed
- **Data parallelism**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamless integration with existing parallelism strategies

## Mathematical Constraints

### Expert Distribution
- **Total experts per layer**: 64
- **Available GPUs**: 16
- **Experts per GPU**: 4 (distributed across layers)
- **Layer coverage**: Each GPU handles experts from different layers to maintain parallelism

### Memory Requirements
- **Per-expert memory**: Must fit within single GPU memory constraints
- **Token buffer**: Additional memory for cross-node token transfers
- **Communication buffers**: Asynchronous transfer requirements