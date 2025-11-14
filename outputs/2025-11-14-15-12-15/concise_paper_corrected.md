# Efficient Layer-Wise Deployment for Large Neural Networks [CORRECTED VERSION]

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

Our method includes an analytical procedure to estimate the memory footprint of each partition and determine the optimal splitting scheme that fits the on-chip memory constraints. This approach facilitates scalable deployment of large models across multiple devices without sacrificing memory locality and efficiency.

## Methodology

### Problem Formulation
Given a 4-layer dense network with 30B parameters total:
- Model layers: L = {l₁, l₂, l₃, l₄}
- Total memory: 30B parameters × 2 bytes (BF16) = 60GB
- Per-layer memory: 60GB / 4 layers = 15GB weights per layer

The goal is to partition these 4 layers across 16 GPUs while ensuring optimal memory utilization.

### Memory Footprint Estimation
For each layer lⱼ:
- **Weights**: 7.5B parameters × 2 bytes = **15GB**
- **Activations**: 128 × 10,000 × 4,096 × 2 bytes = **10.48GB**
- **Buffers**: ~200MB temporary workspace
- **Total per layer**: ~25.7GB

### Partitioning Strategy
**Proposed Layer-wise Approach:**
- Map each layer to 4 GPUs (16 GPUs / 4 layers = 4 GPUs per layer)
- Within each layer: Tensor parallelism across 4 devices
- Between layers: Pipeline parallelism across 4 stages

**Greedy Layer Aggregation:**
1. Treat each of the 4 layers as a separate partition
2. Distribute each partition across 4 GPUs using tensor parallelism
3. Maintain contiguous layer ordering for correctness

### Baseline Comparison
**Standard Tensor + Pipeline Parallelism:**
- TP=8 (tensor parallelism across 8 GPUs)
- PP=2 (pipeline parallelism across 2 stages)
- Total: 8 × 2 = 16 GPUs fully utilized

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 4-layer dense network with 30B parameters
- **Precision**: BF16 (2 bytes/parameter)
- **Batch Size**: 128
- **Sequence Length**: 10,000 tokens
- **Hidden Size**: 4,096 (32 heads × 128 dimensions)
- **MLP Hidden Size**: 16,384

### Results [CORRECTED]

| Model           | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
|----------------|-----------------------|------|----------------|-----------|
| Dense (4-layer) | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078     |
| Dense (4-layer) | Proposed Layer-wise   | 16   | 15,360         | 0.065     |

### Performance Analysis
- **Throughput Improvement**: 20% increase (15,360 vs 12,800 TPS)
- **Latency Reduction**: 17% reduction (0.065ms vs 0.078ms TPOT)
- The proposed layer-wise approach achieves better on-chip memory utilization
- Reduced off-chip memory accesses contribute to performance gains

## Conclusion

We proposed a novel layer-wise deployment strategy for large neural networks that explicitly partitions model layers across multiple accelerator cards while optimizing for on-chip memory constraints. Experimental results on a 4-layer dense model with 30B parameters demonstrate up to 20% improvement in throughput over traditional tensor and pipeline parallelism approaches.

**Note**: This corrected version resolves the layer count discrepancy identified in the original paper. The model consistently has 4 layers throughout, not 16 as incorrectly stated in the original results table.