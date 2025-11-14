# Methodology - Phase 2

## Problem Formulation

### Mathematical Model
Given a large model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:

**Primary Constraint**: Memory footprint constraint
```
S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C
```

Where:
- S(Pᵢ): Total memory footprint of partition Pᵢ
- C: Capacity of SRAM or L2 cache on target hardware
- size(lⱼ): Estimated memory footprint of layer lⱼ

**Secondary Constraints**:
1. Contiguous layer assignment (order preservation)
2. Minimize k (number of partitions)
3. Maximize hardware utilization

## Memory Footprint Estimation - Detailed

### Component Breakdown
For each layer lⱼ, the memory footprint is calculated as:

```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

### Detailed Component Calculations

#### 1. Weight Size Calculation
```
weight_size(lⱼ) = num_parameters(lⱼ) × datatype_size
```
Where datatype_size:
- FP16: 2 bytes
- FP32: 4 bytes
- INT8: 1 byte

#### 2. Activation Size Calculation
```
activation_size(lⱼ) = batch_size × sequence_length × hidden_dimension × num_heads × head_dimension
```

Based on experimental setup:
- batch_size = 128
- sequence_length = 10000
- hidden_dimension = 4096 (derived: 32 heads × 128 head_dim)
- num_heads = 32
- head_dimension = 128

#### 3. Buffer Size Estimation
```
buffer_size(lⱼ) = max_workspace_memory_required(lⱼ)
```
Determined through operator profiling or analytical models.

## Partitioning Algorithms - Detailed Implementation

### 3.1 Greedy Layer Aggregation Algorithm

```
Algorithm: GreedyLayerPartitioning
Input: layers = [l₁, l₂, ..., lₙ], cache_capacity = C
Output: partitions = [P₁, P₂, ..., Pₖ]

1: Initialize partitions = []
2: current_partition = []
3: current_size = 0
4: 
5: for each layer l in layers do
6:     layer_size = size(l)
7:     if current_size + layer_size ≤ C then
8:         Add l to current_partition
9:         current_size = current_size + layer_size
10:    else
11:        if current_partition is empty then
12:            // Handle oversized single layer
13:            Apply intra-layer partitioning or compression
14:        end if
15:        Add current_partition to partitions
16:        Start new partition with layer l
17:        current_size = layer_size
18:    end if
19: end for
20: 
21: if current_partition is not empty then
22:    Add current_partition to partitions
23: end if
24: 
25: return partitions
```

### 3.2 Dynamic Programming for Balanced Partitions (Optional)

```
Algorithm: DynamicProgrammingPartitioning
Input: layers = [l₁, l₂, ..., lₙ], cache_capacity = C
Output: optimal_partitions = [P₁, P₂, ..., Pₖ]

1: Let dp[i][j] = minimum maximum partition size for layers 1..i with j partitions
2: Initialize dp[0][0] = 0, all other dp[i][j] = ∞
3:
4: for i = 1 to n do
5:     for j = 1 to max_partitions do
6:         min_max_size = ∞
7:         for k = 1 to i do
8:             partition_sum = Σ_{m=k}^{i} size(lₘ)
9:             if partition_sum ≤ C then
10:                current_max = max(dp[k-1][j-1], partition_sum)
11:                min_max_size = min(min_max_size, current_max)
12:            end if
13:        end for
14:        dp[i][j] = min_max_size
15:    end for
16: end for
17:
18: Reconstruct partitions from dp table
19: return optimal_partitions
```

## Deployment Strategy - Step by Step

### Pre-deployment Phase
1. **Memory Profiling**: Calculate size(lⱼ) for all layers
2. **Partition Generation**: Apply chosen algorithm to create partitions
3. **Validation**: Ensure each S(Pᵢ) ≤ C
4. **Hardware Mapping**: Assign each Pᵢ to specific accelerator card

### Runtime Deployment Steps
1. **Device Initialization**
   - Initialize k accelerator cards
   - Allocate SRAM/L2 cache space for each partition

2. **Weight Loading**
   - Load all weights for partition Pᵢ into card i's on-chip memory
   - Pre-allocate activation and buffer memory

3. **Execution Flow**
   ```
   For each input batch:
   1. Send input to card 1 (partition P₁)
   2. Execute layers in P₁ sequentially
   3. Transfer output to card 2 (partition P₂)
   4. Continue until final partition Pₖ
   5. Return final output
   ```

4. **Communication Optimization**
   - Use high-bandwidth interconnects (NVLink, PCIe)
   - Minimize data transfer by keeping activations on-chip
   - Overlap computation and communication where possible

### Edge Cases Handling

#### Case 1: Single Layer Exceeding Cache Capacity
```
if size(lⱼ) > C:
    Options:
    1. Apply quantization (reduce datatype precision)
    2. Use intra-layer tensor parallelism
    3. Apply pruning to reduce parameter count
    4. Split layer across multiple devices (breaks cache constraint)
```

#### Case 2: Memory Under-utilization
```
if max(S(Pᵢ)) << C:
    Strategies:
    1. Increase batch size to utilize more memory
    2. Merge adjacent partitions
    3. Rebalance using dynamic programming approach
```

#### Case 3: Variable Layer Sizes
```
if high_variance(layer_sizes):
    Strategies:
    1. Use dynamic programming approach
    2. Group layers by similar sizes
    3. Apply padding or alignment techniques
```

## Implementation Considerations

### Memory Alignment
- Ensure memory allocations are aligned to cache line boundaries
- Account for memory fragmentation overhead

### Batch Size Tuning
- Adjust batch_size to fit within cache constraints
- Balance between memory efficiency and computational throughput

### Performance Monitoring
- Track cache hit/miss ratios
- Monitor inter-device communication overhead
- Measure end-to-end latency and throughput