# Phase 2: Methodology Extraction

## Problem Formulation
Given model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} where:
- Each Pᵢ assigned to separate accelerator card
- Memory footprint S(Pᵢ) ≤ cache capacity C
- Contiguous layer assignment preserved
- Minimize k while maximizing utilization

## Memory Footprint Calculation
For each layer lⱼ:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

### Weight Size
- Parameters × datatype size (BF16 = 2 bytes)
- For dense model: 30B parameters total across 16 layers

### Activation Size  
- Output feature map dimensions × batch size
- Configuration: batch_size = 128, sequence_length = 10000
- Hidden dimensions: 32 heads × 128 dim = 4096

### Buffer Size
- Operator workspace requirements
- Derived from profiling/analytical models

## Partitioning Algorithms

### Greedy Layer Aggregation
```
1. Initialize empty partition Pᵢ
2. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If S(Pᵢ) + size(lⱼ) > C:
   - Finalize Pᵢ with layers {l_start, ..., lⱼ₋₁}
   - Start new partition Pᵢ₊₁ from layer lⱼ
4. Repeat until all layers assigned
```

### Dynamic Programming (Optional)
- Optimize partition boundaries for balanced load
- Minimize maximum partition size while respecting S(Pᵢ) ≤ C

## Deployment Implementation

### Hardware Configuration
- 16 NVIDIA H100 GPUs
- Each with SRAM/L2 cache capacity C
- Inter-GPU communication for partition transfers

### Memory Allocation
```
For each partition Pᵢ on GPUᵢ:
1. Load all weights into SRAM/L2 cache
2. Pre-allocate activation memory
3. Pre-allocate buffer memory
4. Ensure total ≤ C
```

### Execution Flow
```
1. GPU₀ executes P₁ layers sequentially
2. Transfer intermediate output to GPU₁
3. GPU₁ executes P₂ layers sequentially
4. Continue through all partitions
5. Final output from GPUₖ₋₁
```

### Edge Case Handling
- Single layer exceeding C: apply intra-layer partitioning or compression
- Batch size tuning: reduce activation footprint
- Variable layer sizes: adjust heuristics for memory utilization