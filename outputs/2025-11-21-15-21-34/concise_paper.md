# Layer-Wise Deployment for Large Neural Networks: A Concise Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

---

## Core Methodology

### Problem Formulation
Given model with n layers L = {l₁, l₂, ..., lₙ} and cache capacity C per device, partition into k groups P = {P₁, P₂, ..., Pₖ} where:
- S(Pᵢ) = Σ size(lⱼ) ≤ C for each partition
- Layers assigned contiguously in original order
- Minimize k for optimal hardware utilization

### Memory Calculation
Layer memory footprint: size(lⱼ) = weight_size + activation_size + buffer_size
- weight_size: #parameters × 2 bytes (BF16)
- activation_size: output_dims × batch_size × 2 bytes
- buffer_size: operator workspace memory

### Partitioning Strategy
**Greedy Algorithm:** Sequentially add layers to partition until cache capacity exceeded
**Dynamic Programming:** Optimize partition boundaries for balanced load (optional)

### Deployment Process
1. Pre-calculate layer sizes and determine partitions
2. Load each partition entirely into device SRAM/L2 cache
3. Execute layers sequentially within partition
4. Transfer activations only between partitions

## Experimental Results

### Setup
- **Hardware:** 16× NVIDIA H100 GPUs
- **Model:** Dense 16-layer network, 30B parameters, BF16 precision
- **Dimensions:** Batch=128, SeqLen=10000, 32 heads×128 dim, MLP=16384
- **Baseline:** Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

### Performance Comparison
| Method | GPUs | TPS (tokens/s) | TPOT (ms) | Improvement |
|--------|------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | - |
| **Proposed Layer-wise** | 16 | **15,360** | **0.065** | +20% TPS, -17% TPOT |

### Key Benefits
- **20% throughput increase** via optimized cache utilization
- **17% latency reduction** by minimizing off-chip accesses
- **Scalable** across varying model sizes and hardware configurations
- **Practical** deployment on real hardware (H100 GPUs)

## Conclusion
The layer-wise deployment strategy achieves superior performance by explicitly considering on-chip memory constraints during model partitioning, demonstrating significant practical benefits for large-scale neural network deployment.