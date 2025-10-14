# Phase 1: Keypoints Extraction

## Main Problem
Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
Large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Uses Expert Parallelism (EP) ≥ 16 ("large EP")
- Maximizes computational parallelism by fully exploiting distributed resources
- Shifts bottleneck from intra-GPU contention to network communication

## Key Contributions
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Cross-Node Distribution**: Topology-aware placement strategy considering bandwidth, latency, memory capacity, and routing patterns
3. **Balanced Load**: Dynamic gating and load balancing to prevent network bottlenecks
4. **Communication Overlap**: Asynchronous token routing and pipeline scheduling for near-linear scaling
5. **Integration**: Compatible with tensor parallelism (TP) and data parallelism (DP) for large models

## Technical Specifications
- **Model**: 4-layer MoE with 16 experts per layer
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens
- **Token Dimension**: 8192
- **MHA**: 16 heads × 512 dimensions
- **MLP Hidden Size**: 32768

## Performance Results
- **Baseline (TP=8, PP=2)**: 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency