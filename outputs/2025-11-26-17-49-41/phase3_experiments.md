# Phase 3: Detailed Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 61-layer Mixture-of-Experts (MoE)
- **Layer Distribution**: First 3 layers are dense transformer layers, remaining 58 layers are MoE layers
- **Precision**: BF16 (Brain Floating Point 16-bit)
- **Expert Type**: Each expert is a standard MLP (Multi-Layer Perceptron)

### Dimensional Specifications
- **Token Dimension**: 7168 (hidden size of transformer)
- **Multi-Head Attention**: 128 attention heads, each with 128 dimensions
- **MLP Hidden Size**: 2048 (feed-forward network hidden dimension)
- **Sequence Length**: Variable (typical range 512-4096 tokens)
- **Batch Size**: Variable (scaled based on sequence length and GPU memory)

### Hardware Environment
- **GPU Type**: NVIDIA H100 SXM5
- **Single-GPU Specifications**:
  - VRAM Capacity: 64GB
  - Compute Power: 400 TFlops (FP16/BF16)
  - Memory Bandwidth: 1.8 TBps
  - Target MFU Utilization: 60%
  - Target Bandwidth Utilization: 80%

### Network Infrastructure
- **Interconnect**: InfiniBand HDR (High Data Rate)
  - Bandwidth: 200 Gbps per link
  - Latency: < 1 μs within node, < 3 μs cross-node
- **Topology**: Fat-tree or Dragonfly+ topology for large clusters
- **NVLink**: H100 NVSwitch fabric for intra-node communication

## Parallel Deployment Details

### Expert Configuration
- **Experts per MoE Layer**: 32 experts (typical configuration)
- **Total Experts**: 32 experts/layer × 58 MoE layers = 1,856 experts
- **Expert Placement**: One expert per GPU per layer
- **Total GPUs Required**: 1,856 GPUs (32 experts × 58 layers)

### GPU Allocation Strategy
- **Node Configuration**: 8 GPUs per node (standard H100 server)
- **Total Nodes**: 232 nodes (1,856 GPUs ÷ 8 GPUs/node)
- **Expert Distribution**: 
  - Layer 1-3: Dense layers (no experts)
  - Layer 4-61: 32 experts distributed across 32 GPUs per layer
  - Each expert occupies exactly one GPU

### Routing and Communication
- **Token Routing**: Dynamic routing based on gating network scores
- **Top-K Selection**: Top-2 experts typically selected per token
- **Token Transfer**: Asynchronous with compute-communication overlap
- **Load Balancing**: Dynamic adjustment of gating probabilities

### Performance Metrics
- **Throughput**: Tokens/second with all experts computing in parallel
- **Latency**: Per-token processing time minimized through parallel execution
- **Scalability**: Near-linear scaling with increased expert count
- **Efficiency**: 60% MFU target achieved through expert-level parallelism

## Baseline Comparison

### Traditional MoE Placement
- **Multiple Experts per GPU**: 4-8 experts typically placed on single GPU
- **Communication Reduction**: Reduced cross-node transfers
- **Compute Bottleneck**: Expert contention on shared GPU resources
- **Limited Parallelism**: Cannot fully utilize all available GPUs

### Proposed Method Advantages
- **Full Expert Parallelism**: All 32 experts per layer compute simultaneously
- **No GPU Contention**: Each expert has dedicated GPU resources
- **Scalable Architecture**: Linear scaling with additional experts/GPUs
- **Communication Overlap**: Network latency hidden by computation

## Experimental Validation
- **Inference-Only Setting**: Focus on deployment and serving
- **Variable Workloads**: Tested with different sequence lengths and batch sizes
- **Network Stress Testing**: Validated performance under high communication load
- **Topology Optimization**: Confirmed benefits of topology-aware placement