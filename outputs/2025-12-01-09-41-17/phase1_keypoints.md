# Phase 1: Keypoints Extraction

## Main Contribution
Large-scale cross-node expert parallelism strategy for MoE models that deploys at most one expert per GPU, achieving EP ≥ 16 (large EP regime).

## Core Innovation
- Shifts bottleneck from intra-GPU contention to network communication
- Maximizes expert-level parallelism by distributing experts across nodes
- Each GPU hosts exactly one expert per layer when possible
- Targets high-performance computing environments with advanced networking

## Key Technical Components

### 1. Expert Placement Strategy
- Single-expert-per-GPU deployment principle
- Topology-aware distribution across nodes
- Considers bandwidth, latency, memory capacity, and routing patterns
- When E > G, experts replicated to maximize concurrency

### 2. Routing and Load Balancing
- Token batching by destination expert
- Asynchronous routing to overlap with computation
- Dynamic gating probability adjustment
- Prevents expert overloading and network congestion

### 3. Communication Overlap and Scheduling
- Interleaved computation and communication using CUDA streams/NCCL
- Pipeline scheduling for multi-layer MoE networks
- Immediate routing between layers without full batch waiting

## Performance Gains
- 3.75× higher throughput (450,000 vs 120,000 TPS)
- 3.8× lower latency (2.2ms vs 8.3ms TPOT)
- Near-linear scaling for EP ≥ 16
- Full GPU utilization with minimal contention

## Model Specifications
- 16-layer MoE, 64 experts per layer
- FP8 precision
- 128 sequences per batch, 128 tokens per sequence
- Token dimension: 1024
- MHA: 16 heads × 64 dimensions
- MOE hidden size: 2048

## Baseline Comparison
- Baseline: TP=8, PP=2 using 16 GPUs
- Proposed: 1 expert per GPU per layer using 16 GPUs
- Both configurations use adequate H100 GPUs