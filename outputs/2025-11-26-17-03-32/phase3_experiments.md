# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 61-layer Mixture-of-Experts (MoE) model
- **Layer Distribution**: First 3 layers are dense, followed by MoE layers
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: BF16 (BFloat16) for efficient computation and memory usage

### Model Dimensions
- **Token Dimension**: 7168 (dimension of each input token)
- **Multi-Head Attention (MHA)**:
  - Number of heads: 128
  - Dimension per head: 128
- **MLP Hidden Size**: 2048 (feed-forward network hidden dimension)
- **Variable Parameters**:
  - Batch size: Variable
  - Sequence length: Variable

### Hardware Environment
- **GPU Type**: H100 GPU resources (adequate supply, no limits mentioned)
- **Single-card Computing Power**: 400TFlops
- **Memory Bandwidth**: 1.8TBps VRAM bandwidth
- **Memory Utilization**: 80% bandwidth utilization achieved
- **Model FLOPs Utilization (MFU)**: 60% target utilization
- **Single-card Video Memory Capacity**: 64GB per GPU

## Parallel Deployment Details

### Proposed Cross-Node Expert Parallelism
- **GPU Allocation**: Adequate GPUs provided (one GPU per expert per layer)
- **Per-GPU Assignment**:
  - Each GPU hosts exactly one expert per layer
  - No colocation of multiple experts on single GPU
  - Maximum expert-level parallelism achieved

### Routing Mechanism
- **Dynamic Routing**: Input tokens dynamically routed to GPU holding corresponding expert
- **Asynchronous Communication**: Token batches sent asynchronously to minimize idle time
- **Parallel Computation**: All experts per layer compute in parallel
- **Load Distribution**: Balanced workload across all available experts

## Performance Characteristics

### Throughput Optimization
- **Parallel Execution**: All experts process tokens simultaneously
- **Minimal Contention**: One expert per GPU eliminates intra-GPU resource competition
- **Communication Overlap**: Asynchronous token transfers hide communication latency
- **Near-linear Scaling**: Achieved through careful load balancing

### Bottleneck Analysis
- **Primary Bottleneck**: Network communication (shifted from computational contention)
- **Mitigation**: Topology-aware placement and routing
- **Secondary Factors**: Load balancing quality, token distribution efficiency

## Baseline Comparison

### Traditional Approach (Implied Baseline)
- **Expert Placement**: Multiple experts per GPU
- **Trade-off**: Reduced communication vs. computational bottlenecks
- **Limitation**: Expert-level parallelism restricted by GPU count
- **Resource Contention**: Multiple experts compete for same GPU resources

### Proposed Method Advantages
- **Scalability**: Linear scaling with available GPUs
- **Utilization**: Full GPU compute utilization per expert
- **Flexibility**: Accommodates variable batch and sequence lengths
- **Efficiency**: Eliminates intra-GPU expert contention

## Experimental Metrics

### Performance Measurement
- **Throughput**: Tokens processed per second across entire model
- **Latency**: Per-token processing time
- **Utilization**: GPU compute and memory utilization rates
- **Scalability**: Performance scaling with increasing expert count

### Resource Utilization
- **Memory Usage**: Optimal usage within 64GB per GPU constraint
- **Compute Utilization**: 60% MFU target achieved through balanced load
- **Network Efficiency**: 80% bandwidth utilization with asynchronous transfers
- **Load Balance**: Equal distribution across all experts

## Inference-Only Setting

### Evaluation Context
- **Mode**: Inference-only (no training evaluation mentioned)
- **Focus**: Maximum throughput and minimal latency
- **Optimization Target**: Real-time token processing efficiency

### Deployment Constraints
- **No Resource Limits**: Adequate H100 GPUs available
- **Scalability Test**: Demonstrates linear scaling with expert count
- **Performance Validation**: Validates theoretical benefits of large EP regime

## Results Summary

### Achieved Benefits
- **Maximum Expert Parallelism**: One expert per GPU ensures minimal contention
- **Balanced Load**: Topology-aware placement prevents network bottlenecks
- **Scalable Communication**: Asynchronous routing enables near-linear scaling
- **Large Model Support**: Integrates with TP and DP for memory-constrained scenarios

### Validation Outcomes
- **Linear Scaling**: Demonstrated with increasing expert count
- **Communication Efficiency**: Overlapped with computation
- **Resource Utilization**: Optimal use of available H100 resources
- **Practical Feasibility**: Validated in large-scale cluster environment