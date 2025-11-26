# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Model Type**: 61-layer Mixture-of-Experts (MoE) transformer
- **Layer Distribution**: First 3 layers are dense, remaining 58 layers are MoE
- **Precision**: BF16 (16-bit brain floating point)
- **Token Dimension**: 7168
- **Multi-Head Attention**: 128 heads, each head has 128 dimensions
- **MLP Hidden Size**: 2048
- **Variable Parameters**: Batch size and sequence length are variable

### Hardware Environment
- **GPU Type**: H100 GPUs with ample resources (no limits)
- **Single-card Computing Power**: 400 TFLOPS
- **Model FLOPS Utilization (MFU)**: 60%
- **VRAM Bandwidth**: 1.8TB/s per GPU
- **Bandwidth Utilization**: 80%
- **Single-card VRAM Capacity**: 64GB

## Parallel Deployment Configurations

### Proposed Cross-Node Expert Parallelism
- **Expert Parallelism (EP)**: 16 experts per MoE layer
- **Total GPUs Required**: 928 GPUs (16 experts × 58 MoE layers)
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Expert Distribution**: One expert per GPU across all nodes
- **Routing Strategy**: Input tokens dynamically routed to GPU holding corresponding expert
- **Communication**: Token batches sent asynchronously with minimal idle time

### Baseline Conventional Approach
- **Expert Parallelism (EP)**: 4 experts per GPU (colocated)
- **Total GPUs Required**: 232 GPUs (4 experts per GPU for 58 MoE layers)
- **Per-GPU Allocation**: Multiple experts share same GPU
- **Routing Strategy**: Local expert selection with reduced communication
- **Trade-off**: Lower communication but higher intra-GPU contention

## Performance Characteristics

### Proposed Method Advantages
- **Maximum Expert Parallelism**: All experts per layer compute in parallel
- **No Intra-GPU Contention**: One expert per GPU eliminates resource sharing
- **Scalable Communication**: Asynchronous routing with topology-aware placement
- **Load Balancing**: Dynamic gating prevents expert overloading

### Baseline Method Limitations
- **Intra-GPU Contention**: Multiple experts compete for GPU resources
- **Limited Parallelism**: Expert computation serialized within each GPU
- **Communication Savings**: Reduced cross-node traffic but at cost of parallelism

## Key Performance Metrics

### Throughput Maximization
- Proposed method achieves maximum throughput through parallel expert computation
- Baseline method limited by GPU sharing and serialization

### Token Latency Minimization
- Proposed method minimizes token latency through concurrent processing
- Communication overhead offset by increased parallelism

### Resource Utilization
- Proposed method: Full GPU utilization per expert
- Baseline method: Shared GPU resources among experts

## Experimental Validation Points

1. **Scalability**: Method tested with 16+ experts (large EP regime)
2. **Load Balancing**: Dynamic routing prevents stragglers
3. **Communication Overlap**: Async routing masks communication latency
4. **Topology Awareness**: Expert placement considers network topology
5. **Memory Efficiency**: Single expert per GPU optimizes memory usage