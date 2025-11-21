# Phase 2: Methodology Extraction

## Problem Formulation

Given:
- Model with n layers: L = {l₁, l₂, ..., lₙ}
- Cache capacity per card: C
- Goal: Partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
  - S(Pᵢ) ≤ C for each partition
  - Layers assigned contiguously in original order
  - Minimize k

## Memory Footprint Calculation

Layer size calculation:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

Where:
- weight_size(lⱼ) = (#parameters × datatype_size) [BF16 = 2 bytes]
- activation_size(lⱼ) = (output_feature_map_dims × batch_size)
- buffer_size(lⱼ) = workspace_memory_for_operators

## Partitioning Algorithms

### 1. Greedy Layer Aggregation
```
Initialize: P₁ = ∅, current_size = 0, partition_index = 1
For each layer lⱼ in order:
    If current_size + size(lⱼ) ≤ C:
        Add lⱼ to Pᵢ
        current_size += size(lⱼ)
    Else:
        Finalize Pᵢ
        Start new partition Pᵢ₊₁ = {lⱼ}
        current_size = size(lⱼ)
```

### 2. Dynamic Programming (Optional)
```
State: dp[i] = minimum max partition size for layers 1..i
Transition: dp[i] = min_{0≤j<i} max(dp[j], sum_{k=j+1}^i size(lₖ))
Constraint: sum_{k=j+1}^i size(lₖ) ≤ C
```

## Deployment Pipeline

1. **Pre-deployment Analysis**:
   - Static calculation of layer sizes
   - Partition determination
   - Memory allocation planning

2. **Runtime Execution**:
   - Load weights for partition Pᵢ into card i's SRAM/L2
   - Pre-allocate activation and buffer memory
   - Execute layers sequentially within partition
   - Transfer activations between partitions via inter-card communication

3. **Memory Layout**:
   - Each partition: contiguous memory block
   - Weights: persistent during partition execution
   - Activations: dynamic allocation/deallocation per layer
   - Buffers: temporary workspace for operators

## Edge Case Handling

- Single layer exceeding C: apply intra-layer partitioning or compression
- Variable layer sizes: adjust heuristics to prevent under-utilization
- Batch size impact: tune to reduce activation footprint