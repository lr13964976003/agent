# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Architecture Specifications
- **Total Layers**: 61-layer Mixture-of-Experts (MoE) model
- **Layer Distribution**: First 3 layers are dense, followed by MoE layers
- **Expert Architecture**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: BF16 (16-bit Brain Floating Point)

### 1.2 Dimensional Parameters
- **Token Dimension**: 7,168 (hidden state size)
- **Multi-Head Attention (MHA)**:
  - Number of heads: 128
  - Dimension per head: 128
  - Total attention dimensions: 128 × 128 = 16,384
- **MLP Hidden Size**: 2,048 (feed-forward network hidden dimension)
- **Variable Parameters**:
  - Batch size: Variable (adaptive based on workload)
  - Sequence length: Variable (adaptive based on input)

### 1.3 Deployment Mode
**Critical Setting**: **Inference-only** (not training)
- No gradient computation
- No weight updates
- Focus on maximizing throughput for inference requests

## 2. Hardware Environment

### 2.1 GPU Specifications
- **GPU Model**: H100 GPUs
- **Single-card computing power**: 400 TFLOPS
- **Single-card video memory capacity**: 64GB
- **VRAM Bandwidth**: 1.8 TB/s per GPU
- **Availability**: Adequate H100 GPU resources (no limits)

### 2.2 Performance Metrics
- **MFU (Model FLOPS Utilization)**: 60% achieved
- **Bandwidth utilization**: 80% achieved
- **Interconnect**: NVLink, InfiniBand, NVSwitch fabrics

## 3. Parallel Deployment Strategies

### 3.1 Proposed Cross-Node Expert Parallelism (Large EP)

#### Hardware Configuration
- **GPUs Used**: Adequate GPUs (one GPU per expert per layer)
- **Expert Placement**: Each GPU hosts exactly one expert per layer
- **EP Degree**: ≥16 (Large EP regime)

#### Implementation Details
- **Per-GPU Allocation**:
  - Single expert per GPU per layer
  - No expert replication within GPU
  - Cross-node distribution for all experts

#### Routing Strategy
- **Dynamic Token Routing**: Input tokens dynamically routed to GPU holding corresponding expert
- **Asynchronous Transmission**: Token batches sent asynchronously to minimize idle time
- **Batch Optimization**: Tokens grouped by destination expert to reduce network messages

#### Performance Characteristics
- **Parallelism Level**: All experts per layer compute in parallel
- **Throughput**: Maximized through concurrent expert execution
- **Latency**: Minimized token processing time

### 3.2 Traditional Baseline (Comparison Method)

#### Architecture (Implied from Background)
- **Expert Colocation**: Multiple experts per GPU
- **EP Degree**: Moderate (typically 4-8)
- **Communication Priority**: Minimize cross-node communication

#### Hardware Configuration
- **GPUs Used**: Fewer GPUs required (experts shared)
- **Per-GPU Allocation**: Multiple experts per GPU
- **Memory Efficiency**: Higher memory utilization per GPU

#### Implementation Approach
- **Expert Sharing**: 2-4 experts per GPU typical
- **Local Communication**: Intra-node communication preferred
- **Network Minimization**: Reduced cross-node transfers

#### Performance Trade-offs
- **Communication Savings**: Lower network overhead
- **Compute Bottlenecks**: Intra-GPU expert contention
- **Limited Parallelism**: Reduced expert-level concurrency

## 4. Comparison Framework

### 4.1 Metrics for Comparison
- **Throughput**: Tokens processed per second
- **Latency**: Time per token processing
- **Scalability**: Performance vs. number of experts
- **Resource Utilization**: GPU and network efficiency

### 4.2 Scaling Behavior
#### Proposed Method (Large EP)
- **Linear Scaling**: Near-linear throughput increase with EP
- **Network Limited**: Bandwidth becomes bottleneck at high EP
- **Compute Saturated**: GPUs fully utilized

#### Traditional Baseline
- **Diminishing Returns**: Limited by intra-GPU contention
- **Communication Optimal**: Lower network requirements
- **Compute Limited**: Expert sharing creates bottlenecks

## 5. Experimental Results (Inferred from Paper)

### 5.1 Performance Advantages
- **Expert Parallelism**: Maximum concurrent expert execution
- **Load Balancing**: Balanced workload across all GPUs
- **Scalability**: Effective scaling to large EP values (≥16)

### 5.2 Network Efficiency
- **Bandwidth Utilization**: 80% achieved
- **Communication Overlap**: Asynchronous transfer hiding latency
- **Topology Awareness**: Optimized expert placement

### 5.3 Hardware Utilization
- **MFU Achievement**: 60% Model FLOPS Utilization
- **GPU Saturation**: Full compute utilization per GPU
- **Memory Efficiency**: Optimal memory bandwidth usage

## 6. Deployment Considerations

### 6.1 Cluster Requirements
- **High-bandwidth Interconnect**: Essential for large EP
- **Abundant GPU Resources**: One GPU per expert requirement
- **Advanced Networking**: NVLink, InfiniBand capabilities

### 6.2 Optimization Strategies
- **Token Batching**: Group tokens by destination expert
- **Asynchronous Routing**: Non-blocking token transfers
- **Pipeline Scheduling**: Multi-layer overlap optimization

### 6.3 Scalability Thresholds
- **Minimum EP**: 16 experts for large EP classification
- **Optimal Range**: 16-64 experts for best performance
- **Upper Limits**: Network bandwidth becomes constraint