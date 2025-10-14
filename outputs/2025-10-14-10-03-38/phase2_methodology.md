# Phase 2: Methodology Extraction

## 1. Problem Formulation

### Given:
- Model with n layers: L = {l₁, l₂, ..., lₙ}
- Cache capacity per device: C
- Goal: Partition into k disjoint groups P = {P₁, P₂, ..., Pₖ}

### Constraints:
- Memory footprint constraint: S(Pᵢ) ≤ C for each partition Pᵢ
- Contiguity constraint: Layers assigned in original order
- Optimization: Minimize k or balance load across devices

### Mathematical Formulation:
```
S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C
```

## 2. Memory Footprint Estimation

### Layer Memory Components:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

#### Weight Size Calculation:
- weight_size = num_parameters × datatype_size
- FP16: 2 bytes per parameter
- For dense layers: parameters = input_dim × output_dim

#### Activation Size Calculation:
- activation_size = output_feature_map_dimensions × batch_size
- For transformer layers: activation_size = sequence_length × hidden_size × batch_size

#### Buffer Size Calculation:
- Derived from operator profiling or analytical models
- Includes temporary workspace for matrix operations

## 3. Partitioning Algorithms

### 3.1 Greedy Layer Aggregation Algorithm

```
Algorithm: GreedyLayerPartition
Input: layers L = [l₁, l₂, ..., lₙ], cache_capacity C
Output: partitions P = [P₁, P₂, ..., Pₖ]

1: partitions ← []
2: current_partition ← []
3: current_size ← 0
4: 
5: for layer in L do
6:     layer_size ← calculate_layer_size(layer)
7:     if current_size + layer_size ≤ C then
8:         current_partition.append(layer)
9:         current_size ← current_size + layer_size
10:    else
11:        partitions.append(current_partition)
12:        current_partition ← [layer]
13:        current_size ← layer_size
14:    end if
15: end for
16: 
17: if current_partition not empty then
18:     partitions.append(current_partition)
19: end if
20: 
21: return partitions
```

### 3.2 Dynamic Programming for Balanced Partitions

```
Algorithm: DPPartition
Input: layers L = [l₁, l₂, ..., lₙ], cache_capacity C
Output: optimal partitions minimizing max partition size

State: dp[i][j] = minimum maximum partition size for first i layers using j partitions
Transition: 
- For each possible split point k < i
- Check if layers k+1 to i fit in cache
- Update dp[i][j] = min(max(dp[k][j-1], sum(k+1 to i)))
```

## 4. Deployment Strategy

### 4.1 Pre-deployment Steps
1. **Static Analysis**: Calculate memory footprint for each layer
2. **Partitioning**: Apply greedy or DP algorithm to determine layer groups
3. **Validation**: Ensure all partitions satisfy S(Pᵢ) ≤ C

### 4.2 Runtime Deployment
1. **Memory Allocation**: 
   - Pre-allocate weights, activations, and buffers in SRAM/L2 cache
   - No dynamic memory allocation during execution
2. **Execution Flow**:
   - Execute layers sequentially within each partition
   - Transfer intermediate outputs between partitions on different cards
   - Minimize inter-card communication

### 4.3 Edge Case Handling
- **Single Layer Too Large**: Apply intra-layer partitioning or model compression
- **Variable Layer Sizes**: Use adaptive partitioning heuristics
- **Batch Size Adjustment**: Tune batch size to reduce activation memory

## 5. Hardware Mapping

### Device Assignment:
- Each partition Pᵢ assigned to a dedicated accelerator card
- Cards execute partitions in pipeline fashion
- Intermediate outputs transferred via high-speed interconnect

### Memory Layout:
- Weights: Static allocation in SRAM/L2 cache
- Activations: Dynamic during forward pass, reused during backward pass
- Buffers: Temporary workspace cleared after each operation