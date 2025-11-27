# Phase 2: Detailed Methodology Extraction

## Expert Placement Strategy

### Single-Expert-Per-GPU Deployment
For a MoE layer with E experts and a cluster of G GPUs:
- **Case E ≤ G**: Each expert is assigned to a distinct GPU
- **Case E > G**: Experts are replicated across GPUs to maximize concurrency of independent experts while balancing memory usage

Mathematical formulation ensures no GPU hosts more than one expert per layer, eliminating intra-GPU expert contention.

### Cross-Node Distribution
Topology-aware placement algorithm considers:
- **Node-to-node bandwidth and latency**: Minimize maximum tokens sent across any single link
- **GPU memory capacity per node**: Balance memory usage across nodes
- **Expected token routing patterns**: Optimize based on anticipated expert activation patterns

## Routing and Load Balancing

### Gating Mechanism
Standard MoE top-K gating scores determine expert activation for each input token.

### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network message count
2. **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent expert overloading

## Communication Overlap and Scheduling

### Compute-Communication Overlap
- **Implementation**: CUDA streams and asynchronous communication libraries (NCCL, MPI)
- **Mechanism**: While one batch processes on GPU, next batch transfers simultaneously
- **Goal**: Ensure data transfer does not block GPU computation

### Pipeline Scheduling
- **Multi-layer Processing**: Token outputs immediately route to next MoE layer
- **Fine-grained Pipeline**: Experts start processing partial batches upon arrival rather than waiting for complete batches
- **Throughput Increase**: Reduces idle time for each expert

## Scalability Considerations

### Large EP Regime
- **Definition**: EP ≥ 16 experts per parallel group
- **Primary Limiting Factor**: Network bandwidth
- **Mitigation**: Topology-aware routing and token batching
- **Compute Utilization**: One-expert-per-GPU ensures full GPU utilization while communication costs are masked

### Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within each expert's GPU when expert exceeds single-GPU memory
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates
- **Combination**: TP within experts + DP across replicas + EP across experts

## Implementation Details

### Hardware Requirements
- **GPU Type**: H100 with 64GB VRAM, 400TFlops compute power, 1.8TBps memory bandwidth
- **Target Utilization**: 60% MFU, 80% bandwidth utilization
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand, H100-class NVSwitch)

### Software Stack
- **Communication Backend**: NCCL for GPU-to-GPU, MPI for cross-node coordination
- **CUDA Streams**: For asynchronous compute-communication overlap
- **Scheduling**: Fine-grained pipeline scheduling with partial batch processing