# Phase 1: Key Points Extraction

## Main Contribution
- Large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models
- Deploy at most one expert per GPU to maximize computational parallelism
- Expert Parallelism (EP) ≥ 16 defined as "large EP" regime

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Unlike conventional approaches thatcolocating multiple experts on same device
2. **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns
3. **Asynchronous Token Routing**: Overlap computation and communication to minimize idle time
4. **Large EP Optimization**: Specifically designed for EP ≥ 16 environments

## Technical Highlights
- Eliminates expert-level contention by isolating experts on separate GPUs
- Shifts bottleneck from intra-GPU contention to network communication
- Leverages modern HPC networking capabilities (NVLink, InfiniBand)
- Enables near-linear scaling in large GPU clusters

## Performance Claims
- 3.75× higher throughput (450,000 vs 120,000 TPS)
- 3.8× lower latency (2.2ms vs 8.3ms TPOT)
- Achieved on 16-layer MoE with 16 experts per layer
- BF16 precision, 128 sequences batch, 10,000 token sequences

## Critical Requirements
- Expert count must equal or exceed GPU count for optimal performance
- High-bandwidth cross-node communication infrastructure required
- Topology-aware expert placement essential
- CUDA streams for compute-communication overlap