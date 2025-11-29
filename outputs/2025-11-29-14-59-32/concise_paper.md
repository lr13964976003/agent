# Layer-wise Deployment Strategy for Large Neural Networks: A Concise Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

## 2. Methodology

### 2.1 Problem Formulation

Given a large model composed of *n* layers L = {l₁, l₂, ..., lₙ}, the goal is to partition these layers into *k* disjoint groups P = {P₁, P₂, ..., Pₖ}, each assigned to a separate hardware accelerator card, such that:

- The memory footprint of each group Pᵢ does not exceed the capacity C of the SRAM or L2 cache available on the corresponding card
- The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order
- The number of partitions *k* is minimized or balanced to maximize hardware utilization

Formally, for each partition Pᵢ, the size S(Pᵢ) satisfies:

S(Pᵢ) = Σₗⱼ∈Pᵢ size(lⱼ) ≤ C

### 2.2 Memory Footprint Estimation

The memory footprint of each layer includes:

- **Weights**: 3.75 GB per layer (1.875B parameters × 2 bytes BF16)
- **Activations**: 10.48 GB per layer (128 batch × 10000 seq × 4096 hidden × 2 bytes)
- **Temporary Buffers**: 0.5 GB per layer
- **Total per layer**: 14.73 GB

### 2.3 Partitioning Algorithm

**Greedy Layer Aggregation:** Starting from the first layer l₁:
1. Initialize an empty partition Pᵢ
2. Iteratively add subsequent layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If adding lⱼ causes S(Pᵢ) > C, finalize Pᵢ with layers {lₛₜₐᵣₜ, ..., lⱼ₋₁}
4. Start a new partition Pᵢ₊₁ beginning from layer lⱼ
5. Repeat until all layers are assigned

### 2.4 Deployment Strategy

After partitioning, each group Pᵢ is deployed on a separate accelerator card:
- Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache
- Execute the layers sequentially on the assigned card
- Transfer intermediate outputs only when passing data between partitions on different cards

## 3. Experiments

### 3.1 Setup

**Hardware**: 16 NVIDIA H100 GPUs
**Model**: 16-layer dense network, 30B parameters, BF16 precision
**Configuration**: Batch size 128, sequence length 10000, 32 heads × 128 dim, MLP hidden 16384
**Baseline**: TP=8, PP=2 (16 GPUs total)

### 3.2 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance Improvement**: 20% TPS increase (12,800 → 15,360), 17% TPOT reduction (0.078ms → 0.065ms)

### 3.3 Analysis

The proposed deployment method achieves superior performance through:
- **Reduced Memory Access Latency**: Partitions fit in SRAM/L2 cache, minimizing off-chip DRAM accesses
- **Improved Throughput**: Faster memory access and parallel execution on multiple cards
- **Scalability**: Method adapts to varying model sizes and hardware configurations

## 4. Conclusion

We proposed a novel layer-wise deployment strategy for large neural network models that explicitly partitions model layers across multiple accelerator cards with the constraint that each partition fits entirely within the SRAM or L2 cache of the target hardware. This approach significantly reduces off-chip memory accesses and improves inference efficiency.

Experimental results demonstrate substantial performance gains over baseline tensor and pipeline parallelism setup, with up to 20% improvement in throughput (TPS) and corresponding reduction in latency (TPOT). The method enables efficient deployment of large models on multi-card systems while preserving low-latency access to model data.