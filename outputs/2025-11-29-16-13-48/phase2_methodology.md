# Phase 2: Methodology Extraction

## Problem Formulation

Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:

1. Memory footprint constraint: $S(P_i) = sum_{l_j in P_i} text{size}(l_j) leq C$
2. Execution order preserved: layers assigned contiguously in original order
3. Minimize partitions: minimize or balance $k$ to maximize hardware utilization

## Memory Footprint Estimation

For each layer $l_j$, calculate:
$$
text{size}(l_j) = text{weight_size}(l_j) + text{activation_size}(l_j) + text{buffer_size}(l_j)
$$

### Component Breakdown:
- **Weight size**: Number of parameters × datatype size (e.g., BF16 = 2 bytes)
- **Activation size**: Output feature map dimensions × batch size
- **Buffer size**: Workspace memory from profiling or analytical models

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation
Starting from first layer $l_1$:
1. Initialize empty partition $P_i$
2. Iteratively add layers $l_j$ to $P_i$, accumulating $S(P_i)$
3. If $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$
4. Start new partition $P_{i+1}$ from layer $l_j$
5. Repeat until all layers assigned

### 3.2 Dynamic Programming (Optional)
Optimize partition boundaries to minimize maximum partition size while respecting cache capacity constraint.

## Deployment Strategy

1. **Load Phase**: Load all weights and pre-allocate activation/buffer memory within SRAM/L2 cache
2. **Execution Phase**: Execute layers sequentially on assigned card
3. **Communication Phase**: Transfer intermediate outputs only between partitions on different cards

## Edge Case Handling

- **Single layer exceeds C**: Apply intra-layer partitioning or model compression (quantization, pruning)
- **Batch size optimization**: Tune to reduce activation memory footprint
- **Variable layer sizes**: Adjust partitioning heuristics to avoid under-utilization

## Implementation Considerations

- Memory footprint estimation can be static (pre-deployment) or dynamic (profiled)
- Each partition must fit entirely in target device's cache
- Minimize inter-card communication by keeping contiguous layers together
- Preserve model execution order to maintain correctness