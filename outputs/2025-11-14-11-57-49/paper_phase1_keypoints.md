# Paper Keypoints - Phase 1

## Abstract (Retained)
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Methodological Contributions

### 1. Layer-wise Partitioning Strategy
- **Core idea**: Partition n layers into k disjoint groups P = {P₁, P₂, ..., Pₖ}
- **Constraint**: Each partition Pᵢ must fit within SRAM/L2 cache capacity C
- **Memory constraint**: S(Pᵢ) = Σ size(lⱼ) ≤ C for all layers lⱼ in Pᵢ
- **Order preservation**: Layers assigned contiguously in original order

### 2. Memory Footprint Estimation Formula
For each layer lⱼ:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```
- **weight_size**: Parameters × datatype (FP16 = 2 bytes)
- **activation_size**: Output feature map × batch size
- **buffer_size**: Workspace memory for operators

### 3. Partitioning Algorithms
- **Greedy Layer Aggregation**: Simple sequential accumulation until capacity C is reached
- **Dynamic Programming**: Optimizes for balanced partitions while respecting cache constraints

### 4. Deployment Strategy
- **Per-device loading**: Each partition Pᵢ loaded entirely into device's SRAM/L2 cache
- **Sequential execution**: Layers executed sequentially on assigned card
- **Minimal communication**: Only transfer intermediate outputs between partitions on different cards

## Experimental Setup - Key Parameters
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Precision**: FP16
- **Batch size**: 128
- **Sequence length**: 10000
- **Model dimensions**:
  - Number of heads: 32
  - Head dimension: 128
  - MLP hidden size: 16384
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

## Experimental Results - Key Metrics
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Gains
- **20% increase in TPS** (from 12,800 to 15,360 tokens/s)
- **17% reduction in TPOT** (from 0.078ms to 0.065ms)
- **Improvement source**: More efficient on-chip memory utilization reducing memory access latency