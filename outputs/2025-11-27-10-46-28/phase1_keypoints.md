# Phase 1: Key Points Extraction

## Core Problem & Motivation
- Traditional MoE parallelization colocates multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits true expert parallelism
- As model/cluster sizes grow, this trade-off becomes suboptimal

## Proposed Solution
- **Large-scale cross-node expert parallelism strategy**
- Deploy **at most one expert per GPU** across nodes
- Push Expert Parallelism (EP) to **16 or more experts per parallel group** ("large EP")
- Shift bottleneck from expert-level contention to communication (mitigated through scheduling)

## Key Technical Components
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing & Load Balancing**: Dynamic gating, token batching, asynchronous routing
3. **Communication Overlap**: Interleave computation and communication using CUDA streams

## Model Specifications
- **61-layer MoE model** (first 3 layers dense, rest MoE)
- **Token dimension**: 7168
- **MLA heads**: 128 heads × 56 dimensions = 7168 total
- **MLP hidden size**: 18432
- **Precision**: BF16

## Hardware Environment
- **GPUs**: H100 (adequate resources, no limits)
- **Single-card compute**: 400TFlops at 60% MFU utilization
- **VRAM bandwidth**: 1.8TBps at 80% utilization  
- **Single-card memory**: 64GB

## Key Benefits
- **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
- **Balanced Load**: Topology-aware placement prevents network bottlenecks
- **Scalable Communication**: Asynchronous routing enables near-linear scaling
- **HPC/Very Large Cluster Optimized**: Designed for environments with abundant GPU resources