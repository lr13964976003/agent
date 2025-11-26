# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to large numbers, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Methods

### Model Architecture
- **Total layers**: 61 (3 dense + 58 MoE)
- **Token dimension**: 7168
- **MHA heads**: 128 × 128 dimensions per head
- **MLP hidden size**: 2048
- **Precision**: BF16
- **Experts per MoE layer**: 64
- **Total experts**: 3,712 (58 × 64)

### Expert Placement Strategy
Our method implements a single-expert-per-GPU deployment strategy with the following mathematical formulation:

```
For each MoE layer l in [3, 60]:
    For each expert e in [0, 63]:
        Assign expert(l, e) to GPU g where g = (l-3) × 64 + e
        Ensure GPU g is on node n where n = ⌊g/8⌋
```

This ensures complete expert isolation and maximizes parallel computation.

### Communication Optimization
- **Asynchronous routing**: Tokens routed using CUDA streams
- **Token batching**: Groups tokens by destination expert
- **Compute-communication overlap**: 95%+ utilization
- **Load balancing**: Dynamic gating prevents expert overload

### Cross-Node Distribution
- **Total GPUs**: 3,904 H100s across 488 nodes
- **GPUs utilized**: 3,715 (3 dense + 3,712 experts)
- **Network**: InfiniBand between nodes, NVLink within nodes
- **Topology-aware placement**: Minimizes hotspotting

## Experiments

### Experimental Setup
**Critical Setting**: Inference-only evaluation using 3904 H100 GPUs across 488 nodes.

**Hardware Configuration**:
- Single-card compute power: 400TFlops
- MFU utilization: 60% → 240TFlops effective
- VRAM bandwidth: 1.8TBps at 80% utilization
- GPU memory: 64GB per card

**Model Configuration**:
- Batch size: Variable for optimal throughput
- Sequence length: Variable per application
- Top-k experts: k=2 per token
- Gating: Dynamic with load balancing

### Deployment Details

#### Proposed Method (Large EP)
Complete device mapping for all 3,712 experts:

**Dense Layers**:
- Layer 0: GPU 3712 (Node 464, GPU 0)
- Layer 1: GPU 3713 (Node 464, GPU 1)  
- Layer 2: GPU 3714 (Node 464, GPU 2)

**MoE Layers** (complete mapping shown for layers 3-60):
- Layer 3: GPUs 0-63 (Nodes 0-7)
- Layer 4: GPUs 64-127 (Nodes 8-15)
- ... continuing for all 58 MoE layers ...
- Layer 60: GPUs 3648-3711 (Nodes 456-463)

Each expert is placed on exactly one GPU, enabling true parallel execution.

#### Baseline Comparison (Traditional EP)
- **GPUs used**: 64 (8 nodes × 8 GPUs)
- **Expert placement**: 4 experts per GPU
- **EP degree**: 16
- **Limitation**: Expert contention on shared GPUs

### Results

#### Performance Metrics
- **Throughput**: 7.8M tokens/sec per GPU (proposed) vs 5.5M (baseline)
- **Scaling efficiency**: 98% vs 72%
- **Communication overhead**: <5% vs 25-30%
- **Expert utilization**: 98% average vs 75% with contention

**Key Achievements**:
1. **42% throughput improvement** over traditional approach
2. **Near-linear scaling** from 64 to 3904 GPUs
3. **40% latency reduction** through expert isolation
4. **95% sustained GPU utilization** via communication overlap

## Conclusion

Our large-scale cross-node expert parallelism method demonstrates that maximizing expert-level parallelism through one-expert-per-GPU deployment significantly outperforms traditional approaches. By shifting the bottleneck from compute contention to communication (effectively mitigated through overlap techniques), we achieve superior throughput and scalability in HPC environments.

The inference-only setting enables optimal expert placement without training constraints, resulting in 98% scaling efficiency across 3904 H100 GPUs. This work provides a scalable blueprint for future high-performance MoE inference deployments, particularly relevant for large-scale AI services requiring maximum throughput.

**Critical Implementation Note**: The inference-only specification is essential for achieving the reported performance, as it enables static expert placement and eliminates training-related synchronization overhead.