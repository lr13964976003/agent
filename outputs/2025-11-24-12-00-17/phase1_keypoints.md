# Phase 1: Key Points Extraction - Large-Scale Cross-Node Expert Parallelism for MoE Models

## **Core Problem**
Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism as model and cluster sizes grow.

## **Proposed Solution**
A large-scale cross-node expert parallelism strategy that deploys at most one expert per GPU, maximizing computational parallelism by fully exploiting distributed resources and reducing expert-level contention.

## **Key Innovations**
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert, ensuring minimal contention and high compute efficiency
2. **Large Expert Parallelism (EP ≥ 16)**: Qualifies as "large EP" when each expert is distributed across at least 16 devices
3. **Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and token routing patterns
4. **Asynchronous Token Routing**: Overlapping computation and communication to minimize idle time

## **Technical Specifications**
- **Model**: 16-layer MoE with 16 experts per layer
- **Expert Type**: MLP-based experts
- **Dimensions**:
  - Token dimension: 4096
  - MLP hidden size: 16384
  - MHA heads: 32 with 128 dimensions each
- **Precision**: BF16
- **Batch**: 128 sequences of 10,000 tokens each

## **Performance Results**
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 8 experts per GPU + TP shard | 120,000 | 8.3ms |
| Proposed Method | 16 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: ~3.75× higher throughput, ~3.8× lower latency

## **Key Components**
1. **Expert Placement Strategy** - Assigning experts across GPUs and nodes
2. **Routing and Load Balancing** - Ensuring balanced input distribution
3. **Communication Overlap and Scheduling** - Minimizing cross-node transfer impact

## **Scalability Features**
- Compatible with tensor model parallelism (TP) within individual experts
- Integrates with data parallelism (DP) for synchronized weight updates
- Near-linear scaling for EP ≥ 16 in HPC environments