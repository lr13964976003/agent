# Phase 1: Keypoints Extraction

## Main Problem Addressed
Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce communication, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Core Contribution
Propose a large-scale cross-node expert parallelism strategy for MoE models that deploys at most one expert per GPU to maximize computational parallelism and reduce expert-level contention.

## Key Innovations

### 1. Large Expert Parallelism (EP ≥ 16)
- Define "large EP" as configurations where EP ≥ 16
- Distribute experts across as many devices as possible (ideally one per GPU)
- Minimize resource contention and maximize expert-level parallel execution

### 2. Single-Expert-Per-GPU Deployment
- Each GPU hosts at most one expert per layer
- If E ≤ G (experts ≤ GPUs): each expert assigned to distinct GPU
- If E > G: replicate experts to maximize concurrency while balancing memory
- Ensures experts process tokens without contention from other experts on same device

### 3. Cross-Node Distribution Strategy
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimize maximum number of tokens sent across any single link

### 4. Communication Optimization
- Token batching: Group tokens by destination expert to reduce network messages
- Asynchronous routing: Send token batches asynchronously to overlap with computation
- Overlapping compute and communication using CUDA streams/NCCL/MPI
- Fine-grained pipeline scheduling for multi-layer MoE networks

## Experimental Results
- Model: 4-layer MoE, 16 experts per layer, FP16 precision
- Setup: 16 H100 GPUs, batch size 1024, sequence length 10000
- Baseline (TP=8, PP=2): 120,000 TPS, 8.3ms TPOT
- Proposed method: 450,000 TPS, 2.2ms TPOT
- Performance gain: ~3.75× higher throughput, ~3.8× lower latency

## Technical Specifications
- Token dimension: 8192
- MHA: 16 heads, 512 dimension per head
- MLP hidden size: 32768
- Inference-only evaluation
- Cross-node expert placement with asynchronous token routing

## Scalability Benefits
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load Across Nodes: Topology-aware placement prevents bottlenecks
3. Scalable Communication Overlap: Asynchronous routing enables near-linear scaling
4. Compatibility with Large Models: Integrates with TP and DP for models exceeding single-GPU memory