# Phase 2: Methodology Extraction

## Mathematical Foundation

### Problem Formulation
Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:

**Memory Constraint:**
```
S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C
```

**Layer Size Calculation:**
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

## Algorithmic Approach

### 1. Greedy Layer Aggregation Algorithm
```
Input: Model layers L = {l₁, l₂, ..., lₙ}, Cache capacity C
Output: Partitions P = {P₁, P₂, ..., Pₖ}

Initialize:
  i ← 1, j ← 1
  Pᵢ ← ∅
  S(Pᵢ) ← 0

While j ≤ n:
  If S(Pᵢ) + size(lⱼ) ≤ C:
    Pᵢ ← Pᵢ ∪ {lⱼ}
    S(Pᵢ) ← S(Pᵢ) + size(lⱼ)
    j ← j + 1
  Else:
    Finalize Pᵢ
    i ← i + 1
    Pᵢ ← ∅
    S(Pᵢ) ← 0

Finalize last partition Pᵢ
```

### 2. Memory Estimation Components

#### Weight Size Calculation
```
weight_size = num_parameters × datatype_size
```
- BF16: 2 bytes per parameter
- FP16: 2 bytes per parameter
- FP32: 4 bytes per parameter

#### Activation Size Calculation
```
activation_size = output_dimensions × batch_size × sequence_length × datatype_size
```

#### Buffer Size
- Derived from operator profiling or analytical models
- Workspace memory for computation kernels

## Deployment Architecture

### Hardware Configuration
- 16 NVIDIA H100 GPUs
- Each GPU has SRAM/L2 cache capacity C
- High-speed interconnect between GPUs

### Model Specifications (Dense Network)
- Total layers: 16
- Total parameters: 30B
- Precision: BF16 (2 bytes per parameter)
- Attention heads: 32
- Head dimension: 128
- MLP hidden size: 16384
- Batch size: 128
- Sequence length: 10000

### Partition Mapping Strategy
1. **Contiguous Layer Assignment**: Layers assigned in original order
2. **Cache-Constrained Grouping**: Each partition fits within SRAM/L2 cache
3. **Minimal Inter-Card Communication**: Only transfer between partition boundaries

### Execution Flow
```
For each partition Pᵢ on GPUᵢ:
  1. Load weights into SRAM/L2 cache
  2. Pre-allocate activation memory
  3. Pre-allocate buffer memory
  4. Execute layers sequentially:
     - Forward pass through all layers in Pᵢ
     - Store intermediate outputs
  5. Transfer outputs to next GPU (if applicable)
```

## Edge Case Handling

### Single Layer Exceeds Cache Capacity
- Apply intra-layer partitioning
- Use model compression techniques:
  - Quantization (reduce precision)
  - Pruning (remove redundant parameters)

### Variable Layer Sizes
- Adjust partitioning heuristics
- Consider layer size distribution
- Balance memory utilization across partitions

### Batch Size Optimization
- Tune batch size to reduce activation memory
- Trade-off between throughput and memory usage
- Dynamic batch size adjustment based on layer sizes

## Performance Optimization Techniques

### Memory Access Optimization
- Maximize on-chip memory utilization
- Minimize off-chip DRAM accesses
- Cache-friendly data layout

### Parallel Execution
- Multiple partitions execute concurrently
- Pipeline parallelism across GPUs
- Overlapping computation and communication

### Communication Optimization
- Efficient inter-GPU transfer protocols
- Minimize synchronization overhead
- Batch communication operations