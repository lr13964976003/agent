# Phase 2: Methodology Extraction

## Expert Placement Strategy

### Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert assigned to distinct GPU if E <= G
- **Expert Replication**: If E > G, replicate experts to maximize independent expert concurrency while balancing memory
- **Resource Utilization**: Each expert processes tokens without contention from other experts on same device

### Cross-Node Distribution
- **Topology-Aware Placement**: Considers node-to-node bandwidth, latency, GPU memory capacity, expected token routing patterns
- **Optimization Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle
- **Hotspot Prevention**: Distribute experts across nodes to avoid overloading single nodes

## Routing and Load Balancing

### Gating Mechanism
- **Top-K Selection**: For each input token, top-K gating scores determine activated experts
- **Standard MoE Architecture**: Maintains compatibility with existing gating networks

### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network message count
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## Communication Overlap and Scheduling

### Compute-Communication Overlap
- **Interleaved Execution**: Process current batch while transferring next batch from other nodes
- **Technical Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Non-blocking Transfer**: Data transfer does not block GPU computation

### Pipeline Scheduling for Multi-layer MoE
- **Immediate Routing**: Token outputs from previous MoE layer immediately routed to next layer
- **Partial Batch Processing**: Experts in subsequent layers start processing as soon as partial batch arrives
- **Throughput Optimization**: Fine-grained pipeline increases throughput and reduces expert idle time

## Scalability Considerations

### Large EP Regime (EP >= 16)
- **Primary Limiting Factor**: Network bandwidth
- **Mitigation Strategies**: Topology-aware routing, token batching
- **Compute Utilization**: One-expert-per-GPU ensures full GPU utilization while communication costs are masked

### Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Each expert can be further partitioned using TP within its GPU if necessary
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Maintains high expert-level parallelism while handling very large models

## Hardware-Software Co-design

### H100 GPU Specifications
- Computing Power: 400TFlops per GPU
- Memory Bandwidth: 1.8TBps with 80% utilization
- Memory Capacity: 64GB per GPU
- Utilization Target: 60% MFU (Model FLOPs Utilization)

### Network Infrastructure
- **Interconnect Technologies**: NVLink, InfiniBand, H100-class NVSwitch fabrics
- **Communication Strategy**: Shift focus from reducing communication to maximizing compute concurrency
- **Bandwidth Optimization**: Leverage advanced networking to sustain high bandwidth and low latency

## Mathematical Model

### Token Dimension and Processing
- Token Dimension: 7168
- Sequence Length: Variable
- Batch Size: Variable
- MHA Configuration: 128 heads × 128 dimensions per head
- MLP Hidden Size: 2048

### Parallelism Degrees
- Expert Parallelism (EP): >= 16 (large EP regime)
- Tensor Parallelism (TP): Applied within experts when needed
- Data Parallelism (DP): Applied across model replicas
- Pipeline Parallelism (PP): Implicit in multi-layer scheduling

## Implementation Constraints

### Memory Management
- **Single-Expert Memory**: Each expert fits within single GPU memory (64GB limit)
- **Token Storage**: Variable batch sizes accommodated within memory constraints
- **Activation Memory**: Intermediate activations managed through careful batching

### Load Balancing Metrics
- **Expert Utilization**: Monitor compute time per expert
- **Network Traffic**: Track tokens transferred across links
- **Routing Efficiency**: Measure gating decision quality and load distribution