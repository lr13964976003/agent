# Phase One: Key Points Extraction

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points

### Core Problem
- Challenge: Large neural network models require more memory than available on-chip SRAM/L2 cache
- Impact: Off-chip memory access causes latency and bandwidth bottlenecks
- Solution: Layer-wise partitioning to fit within cache constraints

### Methodology Overview
1. **Memory-aware partitioning**: Split model layers ensuring each partition ≤ cache capacity
2. **Contiguous assignment**: Preserve layer execution order
3. **Dynamic allocation**: Map partitions to available hardware
4. **Minimize off-chip access**: Load entire partition into fast memory

### Technical Approach
- **Memory footprint calculation**: weights + activations + temporary buffers
- **Greedy algorithm**: Simple contiguous grouping approach
- **Dynamic programming**: Optional balanced partitioning for better load distribution
- **Deployment**: Sequential execution per partition, only inter-partition transfers between cards

### Experimental Results
- **Setup**: 16-layer dense model, 16 NVIDIA H100 GPUs
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Improvement**: 20% increase in TPS (15,360 vs 12,800), 17% reduction in TPOT (0.065ms vs 0.078ms)

### Key Innovation
- First method to explicitly consider SRAM/L2 cache size as hard constraint in layer partitioning
- Bridges gap between model architecture and hardware memory hierarchy
- Enables scalable deployment without sacrificing memory locality

### Critical Dimensions
- **Model**: 16 layers
- **Hardware**: 16 GPUs (NVIDIA H100)
- **Batch size**: 1024
- **Sequence length**: 10000
- **Precision**: FP16
- **Architecture**: Dense network with 16 heads, 512 head dimension, 32768 MLP hidden size