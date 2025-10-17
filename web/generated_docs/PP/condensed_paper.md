# Condensed Paper: Layer-wise Deployment Strategy for Large Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

---

## 1. Introduction

The rapid growth of deep learning model sizes challenges efficient deployment on hardware with limited on-chip memory. Large models require external memory access, creating latency and bandwidth bottlenecks. Our layer-wise partitioning approach splits *n* model layers across accelerator cards, ensuring each partition fits within SRAM/L2 cache to minimize memory access overhead and maximize throughput.

## 2. Methodology

### 2.1 Problem Formulation
Given *n* layers *L* = {l₁, l₂, ..., lₙ}, partition into *k* groups *P* = {P₁, P₂, ..., Pₖ} such that:
- Each Pᵢ fits cache capacity *C*: S(Pᵢ) = Σₗⱼ∈Pᵢ size(lⱼ) ≤ *C*
- Layers assigned contiguously in original order
- Minimize partitions *k* for optimal hardware utilization

### 2.2 Memory Estimation
Layer size calculation: size(lⱼ) = weight_size + activation_size + buffer_size
- **Weights:** parameters × datatype (FP16 = 2 bytes)
- **Activations:** output dimensions × batch size
- **Buffers:** operator workspace requirements

### 2.3 Partitioning Algorithms
**Greedy Algorithm:**
1. Initialize empty partition
2. Add layers until cache capacity exceeded
3. Create new partition from overflow layer
4. Continue until all layers assigned

**Optional:** Dynamic programming for balanced partitions

### 2.4 Deployment Strategy
1. Calculate layer sizes pre-deployment
2. Apply partitioning algorithm
3. Load entire partitions into SRAM/L2 cache
4. Execute layers sequentially on assigned card
5. Transfer outputs only between partition boundaries

## 3. Experiments

### 3.1 Setup
- **Hardware:** 16 NVIDIA H100 GPUs
- **Models:** 16-layer dense network
- **Parameters:** 16 heads, 512 dims/head, MLP size 32768, batch 1024, sequence 10000, FP16
- **Baseline:** TP=8, PP=2 (standard tensor + pipeline parallelism)

### 3.2 Results
| Model | Method | GPUs | TPS | TPOT |
|-------|--------|------|-----|------|
| Dense 16-layer | Baseline TP=8, PP=2 | 16 | 12,800 | 0.078ms |
| Dense 16-layer | Proposed Layer-wise | 16 | 15,360 | 0.065ms |

**Improvements:** 20% TPS increase, 17% latency reduction

## 4. Conclusion

Our layer-wise cache-aware deployment strategy achieves significant performance gains (20% throughput improvement) by explicitly considering on-chip memory constraints during layer partitioning. The method efficiently scales large models across multiple accelerators while minimizing off-chip memory access, demonstrating superior hardware utilization compared to traditional parallelism approaches.

---

## Key Technical Details
- **Target memory:** SRAM/L2 cache per device
- **Contiguous layer assignment:** Preserves execution order
- **Edge case handling:** Intra-layer partitioning for oversized layers
- **Scalability:** Adapts to varying model sizes and hardware configurations