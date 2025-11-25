# Phase 1: Keypoints Extraction

## Paper: Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

### Core Problem
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication, but this creates computational bottlenecks and limits expert-level parallelism as model/cluster sizes grow.

### Key Innovation
Proposes a cross-node expert parallelism method that prioritizes distributing experts across nodes with **at most one expert per GPU**, pushing Expert Parallelism (EP) to 16 or beyond (large EP).

### Main Contributions
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Topology-Aware Placement**: Considers node-to-node bandwidth, latency, GPU memory, and routing patterns
3. **Asynchronous Token Routing**: Overlaps communication with computation to minimize idle time
4. **Scalable Communication Overlap**: Near-linear scaling for EP ≥ 16 through pipelined scheduling
5. **Integration Capability**: Compatible with tensor parallelism (TP) and data parallelism (DP) for large models

### Performance Impact
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling with 16+ GPUs per expert layer

### Technical Specifications
- **Model**: 16-layer MoE, 16 experts per layer
- **Precision**: BF16
- **Batch**: 128 sequences, 10000 tokens/sequence
- **Token Dimension**: 4096
- **MHA**: 32 heads × 128 dimensions each
- **MLP Hidden**: 16384

### Deployment Strategy
- **Baseline**: TP=8, PP=2 with experts colocated on GPUs
- **Proposed**: One expert per GPU per layer, distributed across nodes
- **Parallelism**: Large EP (≥16) with topology-aware placement