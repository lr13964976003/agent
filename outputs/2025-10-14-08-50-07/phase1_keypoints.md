# Phase 1: Keypoints Extraction

## Problem Statement
- Traditional MoE parallelization assigns multiple experts per GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Trade-off becomes suboptimal as model and cluster sizes grow

## Proposed Solution
- Large-scale cross-node expert parallelism strategy for MoE models
- Deploy at most one expert per GPU
- Ensure Expert Parallelism (EP) ≥ 16 ("large EP")
- Shift optimization focus from reducing communication to maximizing compute concurrency

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert, maximizing independence
2. **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
3. **Token Sharding**: Efficient cross-node token routing with batching and asynchronous transfers
4. **Communication Overlap**: Interleaving computation and communication using CUDA streams
5. **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

## Experimental Setup
- Model: 4-layer MoE, 16 experts per layer, MLP experts
- Precision: FP16
- Batch: 1024 sequences × 10000 tokens
- Token dimension: 8192
- MHA: 16 heads × 512 dim per head
- MLP hidden: 32768

## Results
- **Baseline (TP=8, PP=2)**: 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency
- Hardware: 16 H100 GPUs

## Key Dimensions
- 16 experts per layer
- 8192 token dimension
- 32768 MLP hidden dimension
- 16 heads × 512 = 8192 MHA dimension