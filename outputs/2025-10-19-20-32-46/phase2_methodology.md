# Phase 2: Methodology Extraction

## Problem Formulation

Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} where:

**Constraints:**
- Each partition Pᵢ must satisfy: S(Pᵢ) ≤ C (cache capacity)
- Layers assigned contiguously in original order
- Minimize k or balance partitions for hardware utilization

**Memory constraint formula:**
S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C

## Memory Footprint Calculation

**Layer size components:**
- size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

**Specific calculations:**
1. **weight_size** = parameter_count × datatype_size (FP16 = 2 bytes)
2. **activation_size** = batch_size × sequence_length × hidden_dimensions
3. **buffer_size** = operator_workspace (from profiling/analytical models)

## Partitioning Algorithm Details

### Greedy Layer Aggregation Algorithm
```
Initialize: start = 1, i = 1
While start ≤ n:
    Initialize Pᵢ = ∅, cumulative_size = 0
    For j from start to n:
        If cumulative_size + size(lⱼ) ≤ C:
            Add lⱼ to Pᵢ
            cumulative_size += size(lⱼ)
        Else:
            Finalize Pᵢ with layers {l_start, ..., l_{j-1}}
            Set start = j
            i = i + 1
            Break
    If all layers processed:
        Finalize Pᵢ and terminate
```

### Dynamic Programming Formulation (Optional)
```
Minimize: max(S(Pᵢ)) for all i
Subject to: S(Pᵢ) ≤ C for all i
Variables: Partition boundaries {b₁, b₂, ..., b_{k-1}}
```

## Deployment Strategy Steps

1. **Pre-deployment Analysis**
   - Calculate size(lⱼ) for each layer j ∈ [1,n]
   - Determine k based on available devices and memory constraints

2. **Partition Mapping**
   - Assign P₁ to device 0, P₂ to device 1, ..., Pₖ to device k-1
   - Ensure 1-to-1 mapping between partitions and devices

3. **Memory Allocation**
   - Load all weights for partition Pᵢ into device i's SRAM/L2 cache
   - Pre-allocate activation and buffer memory within cache
   - Verify S(Pᵢ) ≤ C before execution

4. **Execution Flow**
   - Device 0 executes P₁ layers sequentially
   - Transfer intermediate outputs from device i to device i+1
   - Minimal inter-device communication (only between partitions)

## Edge Case Handling

### Case 1: Single Layer Exceeds Capacity
- Solution: Apply intra-layer partitioning or model compression
- Techniques: Quantization (reduce datatype size), pruning (reduce parameter count)

### Case 2: Variable Layer Sizes
- Solution: Adjust partitioning heuristics
- Approach: Group smaller layers together, split large layers if possible

### Case 3: Batch Size Impact
- Solution: Tune batch size to reduce activation memory
- Trade-off: Smaller batch = less activation memory but potentially reduced throughput

## Hardware-Specific Parameters

**NVIDIA H100 Specifications (from experiment):**
- SRAM/L2 cache capacity C = [derived from experiment]
- Device count: 16 GPUs
- Precision: FP16 (2 bytes per parameter)
- Model: 16-layer dense network

**Memory calculation for dense model:**
- Each dense layer: hidden_size parameters
- MLP hidden dimension: 32,768
- Head configuration: 16 heads × 512 dimensions = 8,192 total hidden size
- Batch size: 1024
- Sequence length: 10,000

## Performance Optimization Techniques

1. **Cache-aware placement**: Ensure 100% of partition memory resides in cache
2. **Sequential execution**: Layers within partition executed in order on single device
3. **Minimal communication**: Only transfer activations between partitions
4. **Parallel device utilization**: Different partitions execute simultaneously on different devices