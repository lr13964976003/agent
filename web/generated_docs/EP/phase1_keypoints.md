# Phase 1: Keypoints of Cross-Node Expert Parallelism Paper

## Core Problem
Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks and limiting expert parallelism.

## Key Innovation
Large-scale cross-node expert parallelism with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Definition - Large EP
EP ≥ 16 qualifies as "large EP" - the regime where this method becomes particularly effective.

## Primary Benefits
- **3.75× higher throughput** (450K vs 120K TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Eliminates intra-GPU expert contention
- Maximizes expert-level concurrency
- Shifts bottleneck from compute contention to network communication

## Technical Architecture
- **Expert Placement**: One expert per GPU maximum
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory
- **Routing**: Token batching, asynchronous routing, dynamic load balancing
- **Communication Overlap**: Interleaving compute and communication using CUDA streams/NCCL
- **Pipeline Scheduling**: Subsequent layers start with partial batches

## Deployment Context
- **Use Case**: Inference-only setting
- **Hardware**: 16 H100 GPUs
- **Model**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens per sequence
- **Dimensions**: Token=8192, MLP hidden=32768, MHA heads=16×512=8192

## Performance Comparison
| Method | GPUs | Expert/GPU | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8,PP=2) | 16 | 8 (shared) | 120,000 | 8.3ms |
| Proposed (EP=16) | 16 | 1 (dedicated) | 450,000 | 2.2ms |

## Scalability Considerations
- Network bandwidth becomes primary limiting factor in large EP regime
- Method integrates with TP and DP for models exceeding single-GPU memory
- Topology-aware routing prevents network bottlenecks