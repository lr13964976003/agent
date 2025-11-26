# Phase 1: Keypoints Extraction

## Original Abstract (Preserved)

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Points

### Core Innovation
- **Single-expert-per-GPU deployment**: Each GPU hosts at most one expert, maximizing expert-level parallelism
- **Large EP requirement**: Expert Parallelism (EP) must be ≥16 for optimal performance
- **Cross-node expert distribution**: Exploits distributed resources across nodes to minimize contention

### Model Architecture Details
- **61-layer MoE model** with first 3 layers dense, remaining MoE layers
- **Token dimension**: 7168
- **Multi-head attention**: 128 heads × 128 dimensions per head
- **MLP hidden size**: 2048
- **Precision**: BF16

### Hardware Requirements
- **GPUs**: H100 class
- **Single-card compute**: 400 TFLOPS
- **VRAM capacity**: 64GB per GPU
- **Bandwidth**: 1.8TB/s with 80% utilization
- **MFU**: 60% utilization target

### Deployment Scale
- **Proposed method**: 928 GPUs required (16 experts × 58 MoE layers)
- **Baseline conventional**: 232 GPUs required (4 experts per GPU × 58 MoE layers)
- **Communication**: Asynchronous token routing with CUDA streams, NCCL/MPI for cross-node

### Performance Benefits
- **Maximized expert parallelism**: No intra-GPU expert contention
- **Balanced load distribution**: Topology-aware placement prevents bottlenecks
- **Scalable communication overlap**: Near-linear scaling through async routing