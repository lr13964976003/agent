# Phase 2: Methodology Extraction

## Problem Formulation

Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:

### Constraints:
1. Memory footprint S(Pᵢ) ≤ C (cache capacity)
2. Layers assigned contiguously preserving execution order
3. Minimize k for optimal hardware utilization

### Formal Constraint:
```
S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C
```

## Memory Footprint Estimation Components

### 1. Weight Size
- Based on parameter count and datatype size
- BF16 = 2 bytes per parameter
- Calculation: `num_parameters × datatype_size`

### 2. Activation Size
- Depends on output feature map dimensions
- Varied with batch size and sequence length
- Calculation: `batch_size × sequence_length × hidden_size × datatype_size`

### 3. Buffer Size
- Workspace memory for operators
- Derived from profiling or analytical models
- Includes temporary storage for computations

## Partitioning Algorithms

### Greedy Layer Aggregation Algorithm
```
1. Initialize empty partition Pᵢ
2. iterate through layers lⱼ:
   a. add lⱼ to Pᵢ
   b. accumulate S(Pᵢ)
   c. if S(Pᵢ) > C:
      - finalize Pᵢ with layers {l_start, ..., l_{j-1}}
      - start new partition P_{i+1} from lⱼ
3. repeat until all layers assigned
```

### Dynamic Programming Approach (Optional)
- Minimizes maximum partition size
- Respects cache capacity constraint
- Achieves more balanced load distribution

## Deployment Strategy Steps

### 1. Pre-deployment Analysis
- Calculate memory footprint for each layer
- Determine optimal partition boundaries
- Validate cache capacity constraints

### 2. Hardware Mapping
- Assign each partition Pᵢ to separate accelerator card
- Load weights into SRAM/L2 cache
- Pre-allocate activation and buffer memory

### 3. Execution Flow
- Execute layers sequentially on assigned card
- Transfer intermediate outputs between partitions only
- Minimize inter-card communication

## Edge Case Handling

### Single Layer Exceeding Capacity
- Apply intra-layer partitioning
- Use model compression techniques (quantization, pruning)
- Reduce batch size to decrease activation memory

### Variable Layer Sizes
- Adjust partitioning heuristics
- Consider layer complexity in allocation
- Balance memory utilization across cards

## Memory Hierarchy Optimization

### SRAM/L2 Cache Benefits
- Access time: ~1-10 cycles vs DRAM: ~100-300 cycles
- Energy efficiency: ~10-100x better than DRAM
- Bandwidth: Higher sustained throughput

### Implementation Requirements
- Cache size awareness (typically 40-80MB per GPU)
- Memory alignment for optimal access patterns
- Synchronization mechanisms for inter-card transfers