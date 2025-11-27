# Phase 1: Key Points Extraction

## Abstract (Retained Verbatim)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Points

### Core Innovation
- **One-Expert-Per-GPU Principle**: Deploy at most one expert per GPU to maximize computational concurrency
- **Large Expert Parallelism (EP >= 16)**: Minimum threshold for effective scaling
- **Cross-node Distribution**: Experts distributed across multiple nodes to minimize contention

### Model Architecture
- **61-layer MoE transformer**
- **First 3 layers are dense (non-MoE)**
- **Remaining 58 layers use MoE with expert MLPs**
- **Token dimension**: 7168
- **MLP hidden size**: 18432

### Multi-Head Latent Attention (MLA)
- **Heads**: 128 total
- **Head dimension**: 56 per head
- **Purpose**: Reduce KV cache memory by storing low-dimensional latent representations
- **Mechanism**: Compress X → K_latent (small dimension), then project per head

### Parallel Strategy
- **Expert Parallelism (EP)**: Primary strategy for MoE layers
- **Tensor Parallelism (TP)**: Applied within individual experts when needed
- **Data Parallelism (DP)**: Applied across complete model replicas
- **Topology-aware placement**: Considers node-to-node bandwidth and latency

### Deployment Configuration
- **Hardware**: H100 GPUs (64GB VRAM each)
- **Compute**: 400TFlops per GPU, 60% MFU utilization
- **Memory**: 1.8TBps VRAM bandwidth, 80% utilization
- **Precision**: BF16
- **Allocation**: One GPU per expert per layer (when possible)

### Communication Strategy
- **Asynchronous token routing**: Overlaps compute and communication
- **Token batching**: Groups tokens by destination expert
- **Pipeline scheduling**: Processes tokens immediately upon arrival
- **Load balancing**: Dynamic gating probability adjustment

### Key Metrics
- **Inference-only evaluation**
- **Variable batch and sequence lengths**
- **Focus on throughput maximization**
- **Near-linear scaling target**