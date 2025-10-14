# Layer-wise Deployment Strategy for Large Neural Networks: A Condensed Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache, minimizing memory access overhead and improving throughput during inference or training.

## 2. Methodology

### 2.1 Problem Formulation

Given a large model composed of *n* layers L = {l₁, l₂, ..., lₙ}, the goal is to partition these layers into *k* disjoint groups P = {P₁, P₂, ..., Pₖ}, each assigned to a separate hardware accelerator card, such that:

* The memory footprint of each group Pᵢ does not exceed the capacity C of the SRAM or L2 cache available on the corresponding card
* The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order
* The number of partitions *k* is minimized or balanced to maximize hardware utilization

Formally, for each partition Pᵢ, the size S(Pᵢ) satisfies:

```
S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C
```

### 2.2 Memory Footprint Estimation

The memory footprint of each layer includes:

* **Weights**: The parameter tensors stored for the layer (num_parameters × datatype_size)
* **Activations**: Intermediate outputs needed during inference or training (output_dimensions × batch_size)
* **Temporary Buffers**: Workspace memory required by operators during computation

```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

### 2.3 Partitioning Algorithm

**Greedy Layer Aggregation**: Starting from the first layer l₁, iteratively add subsequent layers to the current partition until adding the next layer would exceed the cache capacity C. Then finalize the current partition and start a new one.

**Dynamic Programming (Optional)**: For more balanced partitions, a DP approach can minimize the maximum partition size while respecting the cache capacity constraint.

### 2.4 Deployment Strategy

After partitioning, each group Pᵢ is deployed on a separate accelerator card with the following steps:

* Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache
* Execute the layers sequentially on the assigned card
* Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication

## 3. Experiments

### 3.1 Setup

**Hardware Platform**: 16 NVIDIA H100 GPUs

**Model**: 16-layer fully connected dense network with FP16 precision

**Configuration**:
- Batch size: 1024
- Sequence length: 10000
- Number of heads: 16
- Head dimension: 512
- MLP hidden size: 32768

**Baseline**: Standard tensor parallelism (TP=8) and pipeline parallelism (PP=2) setup

**Metrics**: Tokens Per Second (TPS) and Time Per Output Token (TPOT)

### 3.2 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### 3.3 Analysis

The proposed deployment method achieves a **20% increase in TPS** and a corresponding **17% reduction in TPOT** compared to the baseline. This improvement results from more efficient on-chip memory utilization, reducing memory access latency. The baseline TP=8, PP=2 approach does not consider on-chip memory constraints explicitly, leading to more off-chip memory accesses and communication delays.

## 4. Conclusion

We proposed a novel layer-wise deployment strategy for large neural network models that explicitly partitions the model layers across multiple accelerator cards with the constraint that each partition fits entirely within the SRAM or L2 cache of the target hardware. This approach significantly reduces off-chip memory accesses and improves inference efficiency. Experimental results demonstrate substantial performance gains over the baseline tensor and pipeline parallelism setup, with up to 20% improvement in throughput (TPS) and a corresponding reduction in latency (TPOT).