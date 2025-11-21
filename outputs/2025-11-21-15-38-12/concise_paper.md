# Layer-wise Deployment Strategy for Large Neural Network Models: A Concise Paper

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

Large models require access to external memory, introducing latency and bandwidth bottlenecks. Our layer-wise partitioning method splits n layers onto multiple accelerator cards, ensuring each layer group fits into SRAM/L2 cache. This minimizes memory access overhead and improves throughput.

## 2. Methodology

### 2.1 Problem Formulation
Given model layers L = {l₁, l₂, ..., lₙ}, partition into k groups P = {P₁, P₂, ..., Pₖ} such that:
- S(Pᵢ) ≤ C (cache capacity)
- Contiguous layer assignment
- Minimize k for optimal utilization

### 2.2 Memory Estimation
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```
- **Weight size**: num_parameters × datatype_size (BF16=2B)
- **Activation size**: batch_size × seq_len × hidden_size × layers_in_partition
- **Buffer size**: operator workspace (profiled)

### 2.3 Partitioning Algorithm
**Greedy approach:**
1. Initialize empty partition
2. Add layers sequentially until cache limit
3. Start new partition when limit exceeded
4. Ensure contiguous layer assignment

### 2.4 Deployment Strategy
- Load weights and pre-allocate memory in SRAM/L2 cache
- Execute layers sequentially on assigned card
- Transfer outputs only between partitions

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network, 30B parameters, BF16
- **Dimensions**: batch=128, seq_len=10000, heads=32, head_dim=128, mlp_hidden=16384
- **Baseline**: TP=8, PP=2

### 3.2 Results
| Method | GPUs | TPS | TPOT(ms) |
|--------|------|-----|----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed | 16 | 15,360 | 0.065 |

### 3.3 Performance
- **20% TPS improvement**
- **17% TPOT reduction**
- Achieved through reduced off-chip memory access

## 4. Conclusion

Our layer-wise deployment strategy partitions model layers across multiple GPUs while ensuring cache-fitting constraints. Experimental results demonstrate 20% throughput improvement over baseline tensor/pipeline parallelism, validating the effectiveness of cache-aware deployment for large models.