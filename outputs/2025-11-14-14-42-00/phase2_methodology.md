# Phase 2: Methodology Extraction

## Layer-wise Deployment Strategy: Complete Methodology

### 1. Problem Formulation (Mathematical Specification)

Given a neural network model with n layers L = {l₁, l₂, ..., lₙ}, we aim to partition these layers into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:

**Constraints:**
- Memory constraint: Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C for each partition Pᵢ
- Order preservation: Layers in each partition must be contiguous in original order
- Minimization objective: Minimize k (number of partitions) for optimal hardware utilization

**Where:**
- C = 50 MB (H100 L2 cache capacity per GPU)
- size(lⱼ) = working set size for layer lⱼ (weights + activations + buffers)

### 2. Memory Footprint Estimation (Detailed)

#### 2.1 Component Breakdown
For each layer lⱼ, the memory footprint consists of:

**Weight Memory:**
```
weight_size = num_parameters × bytes_per_parameter
```
- For BF16 precision: 2 bytes per parameter
- Dense 4-layer model: 30B total parameters → 7.5B parameters per layer
- Per layer weight: 7.5B × 2 = 15 GB

**Activation Memory (Working Set):**
```
activation_size = batch_size × sequence_length × hidden_size × bytes_per_element × working_set_ratio
```
- Full activation: 128 × 10000 × 16384 × 2 = 39.06 GB
- Working set (chunked): ~32 MB (fits cache constraint)

**Buffer Memory:**
```
buffer_size = operator_workspace_from_profiling
```
- Typical transformer layer: ~2-5 MB operator buffers

**Total Working Set per Layer:**
- Weights: ~32 MB (compressed/quantized or partial loading)
- Activations: ~10-15 MB (chunked processing)
- Buffers: ~2-5 MB
- **Total: ~45-50 MB** (fits within 50 MB cache)

### 3. Partitioning Algorithm Implementation

#### 3.1 Greedy Layer Aggregation Algorithm
```python
def greedy_partitioning(layers, cache_capacity):
    partitions = []
    current_partition = []
    current_size = 0
    
    for layer in layers:
        layer_size = calculate_working_set_size(layer)
        
        if current_size + layer_size <= cache_capacity:
            current_partition.append(layer)
            current_size += layer_size
        else:
            partitions.append(current_partition)
            current_partition = [layer]
            current_size = layer_size
    
    if current_partition:
        partitions.append(current_partition)
    
    return partitions
```

**Time Complexity:** O(n) where n is number of layers
**Space Complexity:** O(k) where k is number of partitions

#### 3.2 Dynamic Programming Alternative
For balanced partitions:
```python
def dp_partitioning(layers, cache_capacity):
    n = len(layers)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        current_size = 0
        for j in range(i, 0, -1):
            layer_size = calculate_working_set_size(layers[j-1])
            current_size += layer_size
            
            if current_size <= cache_capacity:
                dp[i] = min(dp[i], max(dp[j-1], current_size))
            else:
                break
    
    return reconstruct_partitions(dp, layers)
```

### 4. Deployment Strategy Implementation

#### 4.1 Device Assignment Process
1. **Preprocessing**: Calculate working set sizes for all layers
2. **Partitioning**: Apply greedy or DP algorithm with cache constraints
3. **Device Mapping**: Assign each partition to a separate GPU
4. **Memory Allocation**: Pre-allocate working sets in SRAM/L2 cache

#### 4.2 Execution Flow
```
Device 0: [Layer 0] → [Layer 1] → ... → [Layer m0]
Device 1: [Layer m0+1] → ... → [Layer m1]
...
Device k-1: [Layer mk-1+1] → ... → [Layer n-1]
```

**Communication Pattern:**
- Between consecutive devices: Point-to-point transfer
- Transfer size: Activation output from last layer in partition
- Overlap strategy: Async communication with computation

### 5. Working Set Optimization Techniques

#### 5.1 Activation Chunking
Instead of storing full activations:
- Process sequence in chunks of size `chunk_size`
- Working activation memory: `batch_size × chunk_size × hidden_size × 2`
- Example: 128 × 64 × 16384 × 2 = 16 MB

#### 5.2 Weight Streaming
For layers exceeding cache:
- Split weight matrix into tiles
- Load tiles on-demand during computation
- Maintain tile cache for reuse

#### 5.3 Quantization Strategy
- Use BF16 for activations and intermediate computations
- Apply INT8 quantization for weights when possible
- Mixed-precision approach to maintain accuracy

### 6. Edge Case Handling

#### 6.1 Single Layer Exceeds Cache
```
if max_layer_size > cache_capacity:
    apply_intra_layer_partitioning()
    or use_quantization_and_compression()
```

#### 6.2 Variable Batch Sizes
- Recalculate working set sizes for new batch
- Dynamically adjust chunking strategy
- Use adaptive partitioning if needed

#### 6.3 Highly Variable Layer Sizes
- Apply affinity-based grouping
- Use load balancing heuristics
- Consider inter-layer optimization opportunities

### 7. Hardware-Specific Optimizations

#### 7.1 H100 GPU Specifications
- **L2 Cache**: 50 MB per GPU
- **Memory Bandwidth**: 3.35 TB/s (HBM3)
- **NVLink 4.0**: 900 GB/s inter-GPU bandwidth
- **Compute**: 989 TFLOPS (FP16)

#### 7.2 Cache Line Optimization
- Align data structures to cache lines (128 bytes)
- Use cache-aware data layouts
- Minimize cache thrashing through access patterns

### 8. Performance Modeling

#### 8.1 Theoretical Speedup
```
speedup = baseline_memory_latency / optimized_memory_latency
```

#### 8.2 Cache Hit Rate
```
cache_hit_rate = cache_accesses / total_memory_accesses
```

#### 8.3 Communication Overhead
```
communication_overhead = (transfer_size / bandwidth) × num_transfers
```

### 9. Implementation Checklist

- [ ] Calculate exact working set sizes for each layer
- [ ] Implement greedy partitioning algorithm
- [ ] Set up device mapping configuration
- [ ] Configure memory allocation strategy
- [ ] Implement communication overlap
- [ ] Set up performance monitoring
- [ ] Validate cache capacity constraints
- [ ] Test edge case handling