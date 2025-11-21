# Phase 2: Methodology Extraction

## Problem Formulation
Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:

**Constraint**: For each partition Pᵢ, memory footprint S(Pᵢ) ≤ C (cache capacity)

**Objective**: Minimize k while maintaining contiguous layer assignment

## Memory Estimation Formula
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

Where:
- weight_size = num_parameters × datatype_size
  - BF16: 2 bytes per parameter
  - FP16: 2 bytes per parameter
  - FP32: 4 bytes per parameter

- activation_size = batch_size × sequence_length × hidden_dim × num_layers_in_partition

- buffer_size = operator_workspace (profiled empirically)
```

## Greedy Partitioning Algorithm
```
Algorithm GreedyLayerPartition(L, C):
    partitions = []
    current_partition = []
    current_size = 0
    
    for each layer l in L:
        layer_size = estimate_memory(l)
        if current_size + layer_size ≤ C:
            current_partition.append(l)
            current_size += layer_size
        else:
            if current_partition:
                partitions.append(current_partition)
            current_partition = [l]
            current_size = layer_size
    
    if current_partition:
        partitions.append(current_partition)
    
    return partitions
```

## Deployment Pipeline
1. **Pre-deployment analysis**: Estimate memory for each layer
2. **Partition generation**: Apply greedy or DP algorithm
3. **Device mapping**: Assign each partition to a GPU
4. **Memory allocation**: Reserve SRAM/L2 cache for weights/activations
5. **Execution**: Sequential layer processing within each partition
6. **Inter-partition communication**: Transfer activations between GPUs

## Cache Capacity Calculation
- **H100 SRAM**: 50MB per GPU (L2 cache)
- **Total parameters**: 30B × 2 bytes (BF16) = 60GB
- **Per layer**: 60GB / 16 layers = 3.75GB per layer (exceeds cache)
- **Need multiple layers per partition** based on actual activation sizes

## Activation Memory Formula
```
Activation_memory = batch_size × seq_len × hidden_size × layers_in_partition
- batch_size: 128
- seq_len: 10000
- hidden_size: 4096 (32 heads × 128 dim)
- For 1 layer: 128 × 10000 × 4096 = 5.24GB
```

## Partition Balance
With 16 layers and 16 GPUs:
- **Target**: 1 layer per GPU
- **But**: 5.24GB activation + weights > 50MB cache
- **Solution**: Layer fusion + activation recomputation or smaller batch sizes

## Performance Metrics
- **Memory utilization**: % of cache used per partition
- **Communication overhead**: Inter-GPU transfer frequency
- **Load balance**: Variance in partition sizes