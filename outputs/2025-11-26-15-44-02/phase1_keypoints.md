# Phase 1: Keypoints Extraction

## Core Problem
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication, but creates computational bottlenecks and limits expert parallelism.

## Proposed Solution
**Large-scale cross-node expert parallelism** with one expert per GPU, pushing Expert Parallelism (EP) to 16+ ("large EP").

## Key Innovations
1. **Single-expert-per-GPU deployment**: Each GPU hosts at most one expert, maximizing compute parallelism
2. **Cross-node expert distribution**: Fully exploits distributed resources across nodes
3. **Communication-computation overlap**: Uses asynchronous routing and token batching to hide communication latency
4. **Topology-aware placement**: Considers node-to-node bandwidth, latency, and GPU memory capacity

## Technical Components
1. **Expert Placement Strategy**: Assigns experts across GPUs/nodes with one-expert-per-GPU principle
2. **Routing and Load Balancing**: Dynamic gating with token batching and load balancing
3. **Communication Scheduling**: Overlaps computation with cross-node token transfers
4. **Scalability Framework**: Optimized for large EP (≥16) with tensor/data parallelism integration

## Key Benefits
- Maximized expert-level parallelism with minimal contention
- Balanced load across nodes preventing bottlenecks
- Near-linear scaling through overlapping communication and computation
- Compatibility with large models requiring tensor/model parallelism

## Critical Dimensions
- **Token Dimension**: 7168
- **MHA Configuration**: 128 heads × 128 dimensions per head
- **MLP Hidden Size**: 2048
- **Model**: 61-layer MoE (first 3 layers dense, then MoE layers)
- **Precision**: BF16
- **Deployment**: One expert per GPU per layer (variable based on expert count)