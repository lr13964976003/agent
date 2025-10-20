# Condensed Paper: Layer-wise Deployment Strategy for Large-Scale Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

---

## 1. Problem Statement & Solution Overview

**Challenge**: Large neural networks exceed on-chip memory capacity, causing frequent off-chip memory access that introduces latency and bandwidth bottlenecks.

**Solution**: Layer-wise deployment strategy that partitions n model layers across k accelerator cards, ensuring each partition Pᵢ satisfies S(Pᵢ) ≤ C (cache capacity constraint).

## 2. Methodology

### 2.1 Memory Footprint Calculation
For each layer lⱼ:
- size(lⱼ) = weight_size + activation_size + buffer_size
- weight_size = parameter_count × datatype_size (FP16=2bytes)
- activation_size = batch_size × sequence_length × hidden_dimensions
- buffer_size = operator_workspace

### 2.2 Partitioning Algorithms

**Greedy Layer Aggregation** (primary method):
1. Initialize empty partition Pᵢ
2. Iteratively add layers until S(Pᵢ) > C
3. Finalize Pᵢ and start new partition
4. Continue until all layers assigned

**Dynamic Programming** (optional for balanced partitions):
- Minimize maximum partition size while respecting cache constraint

### 2.3 Deployment Process
1. Partition model into k groups P₁, P₂,..., Pₖ
2. Map each Pᵢ to device i-1 (0-indexed)
3. Load weights and pre-allocate memory in cache
4. Execute sequentially on assigned device
5. Transfer activations only between partitions

## 3. Experimental Evaluation

### 3.1 Setup
- **Platform**: 16 × NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Configuration**: FP16, batch=1024, seq_len=10000
- **Head config**: 16 heads × 512 dim = 8,192 hidden size
- **MLP**: 32,768 hidden dimension

### 3.2 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|---------|------|-----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance Gains**: 20% TPS increase, 17% TPOT reduction

### 3.3 Analysis
- **Root cause**: Efficient on-chip memory utilization
- **Mechanism**: Reduced off-chip memory accesses and communication delays
- **Comparison**: Baseline doesn't explicitly optimize for cache constraints

## 4. Key Advantages

- **Memory Efficiency**: 100% cache utilization per device
- **Performance**: Reduced latency via local memory access
- **Scalability**: Adapts to varying model sizes and hardware
- **Simplicity**: Direct layer-to-device mapping

## 5. Critical Implementation Parameters

For deployment success, the following must be precisely determined:
- Cache capacity C per device
- Layer-wise memory footprints
- Optimal partition count k
- Device-to-partition mapping scheme

---

*This condensed version retains all key technical details while removing redundant exposition and background material.*