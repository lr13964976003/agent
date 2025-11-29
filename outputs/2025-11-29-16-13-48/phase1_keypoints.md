# Phase 1: Key Points Extraction

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points

### 1. Core Problem
- Large deep learning models require substantial memory capacity
- Off-chip memory accesses introduce latency and bandwidth bottlenecks
- Need to leverage fast but limited on-chip memory (SRAM/L2 cache)

### 2. Proposed Solution
- Layer-wise partitioning and distribution method
- Split model's *n* layers across multiple accelerator cards
- Ensure each partition fits entirely in SRAM or L2 cache
- Minimize memory access overhead and improve throughput

### 3. Technical Approach
- Memory footprint estimation: weights + activations + temporary buffers
- Partitioning algorithms: greedy or dynamic programming
- Deployment strategy: load partitions to separate accelerator cards
- Preserve execution order with contiguous layer assignment

### 4. Memory Footprint Formula
```
size(l_j) = weight_size(l_j) + activation_size(l_j) + buffer_size(l_j)
```

### 5. Constraints
- Each partition P_i must satisfy: S(P_i) ≤ C (cache capacity)
- Full execution order preserved
- Minimize number of partitions k

### 6. Experimental Setup
- Hardware: 16 NVIDIA H100 GPUs
- Models: 4-layer dense network
- Precision: BF16
- Model weight size: 30B
- Batch size: 128
- Sequence length: 10000
- Baseline: TP=8, PP=2

### 7. Key Results
- Dense model: 20% increase in TPS (12,800 → 15,360)
- Dense model: 17% reduction in TPOT (0.078ms → 0.065ms)
- Up to 31% improvement in throughput observed

### 8. Critical Dimensions
- Number of heads: 32
- Dimension per head: 128
- Hidden size of MLP: 16384
- Cache capacity C (not specified, but critical for partitioning)