# Phase 1: Keypoints Extraction

## Core Problem
Traditional MoE implementations place multiple experts on the same GPU to reduce communication overhead, but this creates computational bottlenecks and limits expert-level parallelism.

## Key Innovation - Large Cross-Node Expert Parallelism
- Deploy at most one expert per GPU (one-expert-per-GPU principle)
- Push Expert Parallelism (EP) to 16 or beyond ("large EP" regime)
- Distribute experts across nodes to maximize compute concurrency
- Shift optimization focus from communication reduction to compute maximization

## Method Components
1. **Expert Placement Strategy**: Topology-aware placement ensuring one expert per GPU
2. **Routing & Load Balancing**: Dynamic token routing with balanced distribution
3. **Communication Overlap**: Asynchronous token routing with computation overlap

## Key Benefits
- **Maximized Expert Parallelism**: Each GPU hosts only one expert, eliminating intra-GPU contention
- **Scalable Performance**: Near-linear scaling in large EP regime (EP ≥ 16)
- **Reduced Latency**: 3.8× lower TPOT compared to baseline
- **Increased Throughput**: 3.75× higher TPS compared to baseline

## Technical Specifications
- Model: 16-layer MoE with 16 experts per layer
- Expert Type: MLP with hidden size 16384
- Precision: BF16
- Deployment: 16 H100 GPUs (1 expert per GPU per layer)
- Batch: 128 sequences × 10000 tokens per sequence
- Token dimension: 4096
- MHA: 32 heads × 128 dimensions per head

## Baseline Comparison
- **Baseline**: TP=8, PP=2 with 16 GPUs, multiple experts colocated per GPU
- **Proposed**: EP=16 with 16 GPUs, one expert per GPU per layer
- **Results**: Proposed achieves 450,000 TPS vs 120,000 TPS baseline

## Deployment Requirements
- HPC environment with high-bandwidth interconnects (NVLink, InfiniBand, H100 NVSwitch)
- Cross-node communication capabilities
- Support for asynchronous communication (NCCL/MPI)
- CUDA streams for overlapping compute and communication