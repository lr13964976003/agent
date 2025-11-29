# Methodology Extraction

## Problem Formulation

Given:
- Model with n layers: L = {l₁, l₂, ..., lₙ}
- Cache capacity per card: C
- Goal: Partition into k disjoint groups P = {P₁, P₂, ..., Pₖ}

Constraints:
1. Memory footprint: S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C
2. Execution order preservation: layers assigned contiguously
3. Minimize k for hardware utilization optimization

## Memory Footprint Calculation

For each layer lⱼ:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

### Component Details:
- **weight_size**: parameters × datatype_size
  - BF16/FP16: 2 bytes per parameter
  - FP32: 4 bytes per parameter
- **activation_size**: output_dims × batch_size × sequence_length × datatype_size
- **buffer_size**: operator-specific workspace (profiled or analytically derived)

## Partitioning Algorithms

### Algorithm 1: Greedy Layer Aggregation
```
Input: layers L = [l₁, l₂, ..., lₙ], cache capacity C
Output: partitions P = [P₁, P₂, ..., Pₖ]

1. i ← 1, start ← 1
2. while start ≤ n:
3.   Pᵢ ← ∅
4.   current_size ← 0
5.   for j from start to n:
6.     if current_size + size(lⱼ) ≤ C:
7.       Pᵢ ← Pᵢ ∪ {lⱼ}
8.       current_size ← current_size + size(lⱼ)
9.     else:
10.      break
11.  start ← j
12.  i ← i + 1
13. return P
```

### Algorithm 2: Dynamic Programming (Optional)
- Minimizes maximum partition size
- Achieves more balanced load distribution
- Respects cache capacity constraint C

## Deployment Pipeline

### Step 1: Pre-deployment Analysis
1. Profile each layer's memory requirements
2. Calculate total model footprint
3. Determine optimal partition boundaries

### Step 2: Partition Assignment
1. Map each Pᵢ to dedicated accelerator card
2. Pre-allocate SRAM/L2 cache memory
3. Load weights for assigned layers

### Step 3: Runtime Execution
1. Input data flows through partitions sequentially
2. Each card executes its assigned layers
3. Intermediate outputs transferred between cards via high-speed interconnect
4. Final output produced by last partition

## Edge Case Handling

### Single Layer Exceeding Cache Capacity
- Apply intra-layer tensor parallelism
- Implement model compression (quantization, pruning)
- Reduce batch size to decrease activation memory

### Variable Layer Sizes
- Adjust partitioning heuristics based on layer size distribution
- Implement adaptive partitioning for dynamic workloads
- Consider memory fragmentation effects

## Hardware Mapping Strategy

### Memory Hierarchy Utilization
- L1 cache: Frequently accessed data, small working sets
- L2 cache/SRAM: Entire partition weights and activations
- DRAM: Backup storage, overflow handling

### Communication Optimization
- Minimize inter-card data transfers
- Batch intermediate outputs for efficient transmission
- Overlap computation with communication where possible