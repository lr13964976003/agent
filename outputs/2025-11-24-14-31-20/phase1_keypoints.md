# Phase 1: Keypoints Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Keypoints Summary

### Core Problem Addressed
- Traditional MoE implementations place multiple experts per GPU to reduce communication, creating computational bottlenecks
- Need to balance communication overhead vs compute saturation in large clusters

### Proposed Solution
- **Large-scale cross-node expert parallelism** with at most one expert per GPU
- **Large EP regime**: EP ≥ 16 (large expert parallelism)
- Shift bottleneck from intra-GPU contention to network communication
- Exploit modern HPC networking (NVLink, InfiniBand, NVSwitch) to mitigate communication costs

### Key Innovations
1. **Single-expert-per-GPU deployment** ensuring minimal contention
2. **Topology-aware expert placement** minimizing network hotspots
3. **Asynchronous token routing** overlapping computation and communication
4. **Fine-grained pipeline scheduling** reducing GPU idle time

### Technical Details
- **Model Architecture**: 16-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: BF16
- **Batch Size**: 128 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 4096
- **MHA Heads**: 32 heads × 128 dimension = 4096
- **MLP Hidden Size**: 16384

### Baseline Configuration (for comparison)
- **TP=8, PP=2** using 16 H100 GPUs
- Each GPU holds 1/8 tensor-parallel shard for all layers
- Each pipeline stage spans 8 GPUs
- Experts colocated: 8 experts per layer per GPU

### Proposed Configuration
- **16 H100 GPUs** (one GPU per expert per layer)
- Each GPU hosts exactly one expert per layer
- Dynamic token routing to expert locations
- Asynchronous token batch transfers

### Results Achieved
- **Throughput**: 450,000 TPS (vs 120,000 TPS baseline)
- **Latency**: 2.2ms TPOT (vs 8.3ms baseline)
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

### Scalability Focus
- Designed for high-performance computing environments
- Optimized for H100 clusters with abundant GPU resources
- Near-linear scaling with sufficient network bandwidth
- Compatible with tensor parallelism (TP) and data parallelism (DP) for memory constraints