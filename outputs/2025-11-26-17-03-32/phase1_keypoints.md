# Phase 1: Keypoints Extraction

## Abstract Keypoints
- Large-scale cross-node expert parallelism strategy for MoE models
- Deploy at most one expert per GPU (vs conventional multiple experts per GPU)
- Expert Parallelism (EP) >= 16 defined as "large EP"
- Maximizes computational parallelism and reduces expert-level contention
- Optimized for HPC and large GPU cluster environments

## Technical Keypoints

### Core Problem
- Traditional MoE parallelization assigns multiple experts to same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

### Proposed Solution
- Cross-node expert parallelism with at most one expert per GPU
- Shifts bottleneck from intra-GPU contention to network communication
- Leverages modern HPC networking (NVLink, InfiniBand, NVSwitch)

### Key Components
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Token batching, asynchronous routing, dynamic gating
3. **Communication Overlap**: Interleave computation and communication, pipeline scheduling

### Model Architecture Details
- 61-layer MoE model
- First 3 layers are dense, followed by MoE layers
- Token dimension: 7168
- MHA: 128 heads, 128 dimensions per head
- MLP hidden size: 2048
- Precision: BF16

### Performance Characteristics
- H100 GPUs with 400TFlops computing power
- 60% MFU utilization
- 1.8TBps VRAM bandwidth, 80% utilization
- 64GB single-card video memory

### Deployment Strategy
- Adequate GPUs (one GPU per expert per layer)
- Each GPU hosts exactly one expert per layer
- Dynamic token routing with asynchronous batch sending
- All experts per layer compute in parallel

### Scalability Advantages
- Maximized Expert Parallelism: Minimal contention, high compute efficiency
- Balanced Load: Topology-aware placement prevents bottlenecks
- Communication Overlap: Asynchronous routing enables near-linear scaling
- Large Model Compatibility: Integrates with TP and DP for memory constraints