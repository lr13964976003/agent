# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 61-layer Mixture-of-Experts (MoE) model
- **Layer Distribution**: First 3 layers are dense, followed by 58 MoE layers
- **Expert Type**: Each expert is an MLP (Multi-Layer Perceptron)
- **Precision**: BF16 (BFloat16)
- **Batch Processing**: Variable batch size and sequence length

### Model Dimensions
- **Token Dimension**: 7168
- **Multi-Head Attention (MHA)**: 128 heads, each with 128 dimensions
- **MLP Hidden Size**: 2048

### Hardware Environment
- **GPUs**: H100 GPU resources (adequate supply, no limits)
- **Single-card Computing Power**: 400TFlops
- **Model FLOPS Utilization (MFU)**: 60%
- **VRAM Bandwidth**: 1.8TBps
- **Bandwidth Utilization**: 80%
- **Single-card Video Memory Capacity**: 64GB

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism

#### GPU Allocation Strategy
- **Principle**: One GPU per expert per layer
- **Implementation**: Each GPU hosts exactly one expert per layer
- **Resource Utilization**: Adequate GPUs provided (no resource constraints)

#### Routing Mechanism
- **Dynamic Routing**: Input tokens dynamically routed to GPU holding corresponding expert
- **Asynchronous Transfer**: Token batches sent asynchronously to ensure minimal idle time
- **Parallel Processing**: All experts per layer compute in parallel

#### Performance Optimization
- **Throughput Maximization**: Parallel expert computation across all GPUs
- **Latency Minimization**: Overlapping communication and computation
- **Load Balancing**: Continuous monitoring and dynamic adjustment

## Experimental Context

### Inference-Only Setting
- **Evaluation Type**: Inference-only (training not evaluated)
- **Focus**: Large-scale cross-node expert parallelism performance
- **Objective**: Validate proposed method's effectiveness in high-performance computing environment

### Baseline Comparison Context
- **Traditional Approach**: Multiple experts per GPU to reduce communication
- **Limitation**: Creates computational bottlenecks and limits expert parallelism
- **Proposed Solution**: Single expert per GPU to maximize parallelism

## Key Performance Characteristics

### Scalability Benefits
1. **Expert Independence**: Each expert runs in near-isolation on dedicated GPU
2. **Compute Saturation**: All GPUs fully utilized for expert computation
3. **Communication Overlap**: Token routing overlapped with computation
4. **Load Distribution**: Balanced across nodes to prevent hotspotting

### Network Optimization
- **Topology Awareness**: Expert placement considers node-to-node bandwidth/latency
- **Token Batching**: Reduces number of network messages
- **Bandwidth Utilization**: 80% of available 1.8TBps VRAM bandwidth
- **Low Latency**: Leveraging H100-class NVSwitch and InfiniBand fabrics

## Critical Deployment Parameters

### Resource Requirements
- **Minimum GPUs**: Equal to number of experts per layer (EP ≥ 16 for "large EP")
- **Memory per GPU**: Sufficient for single expert (64GB available)
- **Network**: High-bandwidth, low-latency interconnects
- **Compute**: 400TFlops per GPU minimum recommended

### Performance Metrics
- **MFU Achievement**: 60% Model FLOPS Utilization
- **Expert Parallelism**: One expert per GPU (maximum possible)
- **Communication Efficiency**: Overlapped with computation
- **Load Balance**: Dynamically maintained across all experts