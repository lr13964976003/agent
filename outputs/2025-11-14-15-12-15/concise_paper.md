# Layer-wise Deployment Strategy for Large Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

### Introduction

Large deep learning models require access to external memory, introducing latency and bandwidth bottlenecks. We propose a layer-wise partitioning method where *n* layers are split and mapped onto multiple accelerator cards, ensuring each partition fits entirely into SRAM/L2 cache. This minimizes memory access overhead and improves throughput.

### Methodology

#### Problem Formulation
Given n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- S(Pᵢ) = Σₗⱼ∈Pᵢ size(lⱼ) ≤ C (cache capacity)
- Preserve execution order with contiguous layer assignment
- Minimize k for balanced hardware utilization

#### Memory Footprint Estimation
For each layer lⱼ:
```
size(lⱼ) = weight_size + activation_size + buffer_size
```

**Dense Model Parameters:**
- 16 layers total, 30B parameters
- 1.875B parameters per layer
- BF16 precision (2 bytes/parameter)
- Batch size: 128, Sequence length: 10,000
- Hidden size: 4096, MLP hidden: 16,384

**Memory per layer:**
- Weight: 3.75 GB
- Activation: 10.48576 GB  
- Buffer: 1.4236 GB
- **Total: 15.66 GB per layer**

#### Partitioning Algorithm
1. **Greedy Layer Aggregation:** Sequentially group layers until capacity C is reached
2. **Dynamic Programming:** Optional balanced partitioning optimization

#### Deployment Strategy
- Each partition Pᵢ loaded to GPU i's SRAM/L2 cache
- Sequential execution with layer outputs transferred between GPUs
- 1 layer per GPU for 16-layer model on 16 GPUs

### Experiments

#### Setup
- **Hardware:** 16 NVIDIA H100 GPUs
- **Model:** Dense 16-layer network (30B parameters)
- **Baseline:** TP=8, PP=2
- **Proposed:** Layer-wise partitioning across 16 GPUs

#### Results
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance Gain:** 20% TPS increase, 17% TPOT reduction

### Conclusion

Our layer-wise deployment strategy achieves significant performance improvements by explicitly considering cache constraints, reducing off-chip memory access, and improving inference efficiency for large neural networks.