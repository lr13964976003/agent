# Phase 1: Keypoints Extraction

## Problem Statement
Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce inter-node communication, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
Large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models that maximizes computational parallelism by deploying at most one expert per GPU, ensuring Expert Parallelism (EP) is at least 16.

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert, eliminating intra-GPU contention
2. **Cross-Node Distribution**: Topology-aware expert placement across nodes considering bandwidth, latency, and memory capacity
3. **Asynchronous Token Routing**: Overlapping computation and communication through token batching and asynchronous routing
4. **Large EP Regime**: Optimized for EP ≥ 16, shifting bottleneck from compute contention to network communication

## Technical Details
- **Model**: 4-layer MoE with 16 experts per layer
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768

## Results
- **Throughput**: 3.75× higher than baseline (450,000 vs 120,000 tokens/second)
- **Latency**: 3.8× lower than baseline (2.2ms vs 8.3ms TPOT)
- **Deployment**: 16 H100 GPUs with one expert per GPU per layer

## Baseline Comparison
- **Baseline**: TP=8, PP=2 with 16 GPUs, 8 experts per GPU per layer
- **Proposed**: EP=16 with 16 GPUs, 1 expert per GPU per layer
- **Key Advantage**: Eliminates intra-GPU expert contention and enables true expert-level parallelism