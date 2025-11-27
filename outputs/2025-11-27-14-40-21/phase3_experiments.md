# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Model Type**: 61-layer Mixture-of-Experts (MoE)
- **Architecture**: 
  - First 3 layers are dense transformer layers
  - Remaining 58 layers are MoE layers
  - Each expert is a standard MLP (feed-forward network)
- **Precision**: FP8 for all computations
- **Batch Size**: Variable batch size (optimized per deployment scenario)
- **Sequence Length**: Variable sequence length (context-dependent)

### Dimensional Specifications
- **Token Dimension**: 7168 (embedding/hidden dimension)
- **MLA (Multi-Head Latent Attention)**:
  - Number of heads: 128
  - Dimension per head: 56
  - Total attention dimension: 128 × 56 = 7168
- **Expert MLP Hidden Size**: 18432 (4× hidden dimension expansion)

### Hardware Environment
- **GPUs**: Ample H100 GPU resources (no limits specified)
- **Single-card Computing Power**: 400 TFlops
- **MFU (Model FLOPS Utilization)**: 60%
- **VRAM Bandwidth**: 1.8 TBps
- **Bandwidth Utilization**: 80%
- **Single-card Video Memory Capacity**: 64GB

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism

#### GPU Allocation Strategy
- **GPUs Used**: Adequate GPUs (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert per layer
  - No GPU contains multiple experts from the same layer
  - Experts distributed across nodes based on topology

#### Routing Implementation
- **Dynamic Token Routing**: Input tokens routed to GPU holding corresponding expert
- **Asynchronous Communication**: Token batches sent asynchronously to minimize idle time
- **Parallel Execution**: All experts per layer compute in parallel
- **Load Balancing**: Dynamic adjustment to prevent expert overloading

#### Performance Characteristics
- **Throughput**: Maximized through parallel expert execution
- **Latency**: Minimized token processing time
- **Scalability**: Near-linear scaling with expert count
- **Resource Utilization**: Full GPU utilization achieved

## Experimental Methodology

### Inference-Only Setting
- **Evaluation Mode**: Inference-only (no training evaluation)
- **Focus**: Deployment efficiency and throughput optimization
- **Metrics**: Primarily throughput and latency measurements

### Communication Optimization
- **Cross-node Token Transfer**: Optimized for HPC networking
- **Topology-Aware Placement**: Minimizes network congestion
- **Overlapping Strategy**: Computation and communication overlap
- **Batching**: Token grouping by destination expert

## Key Experimental Insights

### Large EP Regime Validation
- **EP Degree**: 16+ experts per parallel group
- **Network Impact**: Communication costs effectively managed
- **Compute Saturation**: Full GPU utilization maintained
- **Scalability**: Demonstrated near-linear scaling

### Resource Efficiency
- **Memory Usage**: Optimized per-GPU memory allocation
- **Bandwidth Utilization**: 80% sustained utilization
- **Compute Utilization**: 60% MFU achieved
- **Expert Isolation**: Minimal inter-expert interference

## Deployment Configuration Summary

### Expert Distribution
- **Per-Layer Expert Count**: Variable based on model configuration
- **Total Expert Count**: Scales with number of MoE layers
- **Placement Strategy**: One-expert-per-GPU principle
- **Cross-Node Distribution**: Topology-optimized placement

### Communication Pattern
- **Token Routing**: Dynamic based on gating scores
- **Data Transfer**: Asynchronous cross-node communication
- **Scheduling**: Pipeline scheduling across layers
- **Load Balancing**: Real-time adjustment mechanisms

### Performance Optimization
- **Parallelism Level**: Maximum expert-level parallelism
- **Resource Contention**: Minimized through distribution
- **Network Efficiency**: Optimized for HPC environments
- **Throughput**: Maximized through concurrent execution