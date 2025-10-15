# Layer-wise Deployment Strategy for Large Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

## 2. Methodology

### 2.1 Problem Formulation

Given a model with *n* layers L = {l₁, l₂, ..., lₙ}, partition into *k* disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C (cache capacity)
- Layers assigned contiguously in original order
- Minimize or balance *k* for hardware utilization

### 2.2 Memory Footprint Estimation

**size(lⱼ) = weight_size + activation_size + buffer_size**

Where:
- **weight_size**: parameters × datatype (FP16 = 2 bytes)
- **activation_size**: output dimensions × batch size
- **buffer_size**: operator workspace requirements

### 2.3 Partitioning Algorithms

**Greedy Layer Aggregation:**
1. Initialize empty partition Pᵢ
2. Add layers sequentially until S(Pᵢ) > C
3. Finalize Pᵢ, start new partition P_{i+1}
4. Repeat until all layers assigned

**Dynamic Programming (Optional):**
Optimize partition boundaries for balanced load and minimal partitions.

### 2.4 Deployment Strategy

1. Load weights and pre-allocate memory within SRAM/L2 cache
2. Execute layers sequentially on assigned card
3. Transfer intermediate outputs only between partitions

## 3. Experiments

### 3.1 Setup
- **Hardware:** 16 NVIDIA H100 GPUs
- **Model:** 16-layer dense network
- **Parameters:** FP16, batch=1024, seq_len=10000, 16 heads, head_dim=512, MLP_hidden=32768
- **Baseline:** TP=8, PP=2 configuration
- **Metrics:** Tokens Per Second (TPS), Time Per Output Token (TPOT)

### 3.2 Results

| Model | Method | GPUs | TPS | TPOT (ms) |
|--------|---------|------|-----|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance Gains:**
- **20% TPS improvement** (15,360 vs 12,800)
- **17% TPOT reduction** (0.065ms vs 0.078ms)

### 3.3 Analysis
The proposed method achieves significant performance gains through efficient on-chip memory utilization, reducing memory access latency compared to baseline tensor and pipeline parallelism approaches.

## 4. Conclusion

We proposed a layer-wise deployment strategy that explicitly partitions model layers across accelerator cards with cache capacity constraints. Experimental results demonstrate substantial performance improvements over traditional parallelism methods, with up to 20% throughput gains. This approach enables efficient deployment of large models while maximizing fast memory utilization and minimizing communication overhead.