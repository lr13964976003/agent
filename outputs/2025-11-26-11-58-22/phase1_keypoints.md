# Phase 1: Keypoints Extraction

## Core Innovation
- **Large-scale cross-node expert parallelism strategy** for Mixture-of-Experts (MoE) models
- **One expert per GPU deployment** - maximizes computational parallelism by distributing experts across nodes
- **Large EP definition**: Expert Parallelism (EP) of at least 16, qualifying as "large EP"

## Key Technical Contributions

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert per layer
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, GPU memory
- **61-layer MoE model**: First 3 layers are dense, followed by 58 MoE layers with 64 experts each

### 2. Routing and Load Balancing
- **Dynamic token routing**: Tokens asynchronously routed to destination experts
- **Token batching**: Groups tokens by destination expert to reduce network messages
- **Load balancing**: Dynamic gating probability adjustment to prevent expert overload

### 3. Communication Optimization
- **Compute-communication overlap**: Interleaves expert computation with token transfers
- **Pipeline scheduling**: Immediate routing between MoE layers
- **Asynchronous routing**: CUDA streams/NCCL for non-blocking data transfer

## Model Specifications
- **Architecture**: 61-layer transformer with MoE
- **Token dimension**: 7168
- **MHA heads**: 128 heads, 128 dimensions per head
- **MLP hidden size**: 2048
- **Precision**: BF16
- **Experts per layer**: 64 (for MoE layers 3-60)

## Experimental Configuration
- **Setting**: Inference-only
- **Hardware**: H100 GPUs (3904 total across 488 nodes)
- **Single-card compute**: 400TFlops
- **MFU utilization**: 60%
- **VRAM bandwidth**: 1.8TBps
- **Bandwidth utilization**: 80%
- **GPU memory**: 64GB per card

## Performance Advantages
1. **Maximized Expert Parallelism**: Minimal contention, high compute efficiency
2. **Balanced Load**: Topology-aware placement prevents bottlenecks
3. **Scalable Communication**: Near-linear scaling with asynchronous routing
4. **Large Model Compatibility**: Integrates with TP and DP for memory constraints

## Deployment Requirements
- **Total GPUs**: 3904 H100s (488 nodes × 8 GPUs/node)
- **Expert assignments**: 3712 experts (58 layers × 64 experts) + 3 dense layers
- **Network**: High-bandwidth, low-latency interconnects (NVLink, InfiniBand)
- **Memory**: Sufficient GPU memory for one expert per GPU