# Phase Two: Methodology Extraction

## Methodology Overview

### 1. Problem Formulation
Given a large model with *n* layers $L = {l_1, l_2, ..., l_n}$, partition into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$ such that:
- Each group $P_i$ assigned to separate accelerator card
- Memory footprint $S(P_i) \leq C$ (SRAM/L2 cache capacity)
- Preserve contiguous layer execution order
- Minimize number of partitions $k$

### 2. Memory Footprint Estimation Formula
For each layer $l_j$:
$$
\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)
$$

#### Detailed Components:
- **Weight size**: `number_of_parameters * datatype_size` (FP16 = 2 bytes)
- **Activation size**: `batch_size * sequence_length * hidden_dimensions`
- **Buffer size**: Profiled workspace memory for operators

###table>
| Component | Calculation | Dependencies |
|-----------|-------------|--------------|
| Weights | `params * 2 bytes` | Model architecture |
| Activations | `batch * seq_len * hidden` | Input dimensions |
| Buffers | Profiled values | Operator requirements |

### 3. Partitioning Algorithms

#### Greedy Layer Aggregation Algorithm
```
Initialize:
  partition_list = []
  current_partition = []
  current_size = 0

For each layer l_j in L:
  If current_size + size(l_j) <= C:
    Add l_j to current_partition
    current_size += size(l_j)
  Else:
    Finalize current_partition
    Start new partition with l_j
    current_size = size(l_j)
```

#### Dynamic Programming Approach (Optional)
- Objective: Minimize maximum partition size while respecting cache constraint
- Trade-off: More balanced partitions vs. computational overhead
- State: dp[i][j] = minimum partition cost for layers 1..i with j partitions

### 4. Deployment Strategy Pipeline

#### Pre-processing Phase
1. Analyze model architecture to extract layer specifications
2. Calculate memory footprint for each layer using static analysis or profiling
3. Determine cache capacity C for each target device
4. Execute partitioning algorithm to generate layer groups P_i

#### Runtime Deployment
1. **Device Assignment**: Map each partition P_i to accelerator card i
2. **Memory Allocation**: Pre-allocate weights, activations, and buffers within SRAM/L2
3. **Execution Flow**:
   - Load partition weights into device cache
   - Execute layers sequentially within partition
   - Transfer outputs to next partition device
   - Minimal inter-card communication (only between adjacent partitions)

### 5. Memory Footprint Calculation Example

For the experimental dense model (16-layer):
- **Batch size**: 1024
- **Sequence length**: 10000
- **Hidden size**: 16 heads × 512 head dimension = 8192
- **MLP hidden**: 32768
- **Precision**: FP16 (2 bytes)

#### Per-layer memory estimation:
- **Attention weights**: ~67MB (QKV + Output projections)
- **MLP weights**: ~1GB (2 linear layers 8192×32768)
- **Activations**: ~160MB (batch×seq×hidden)
- **Total per layer**: ~1.2GB

#### Cache constraint impact:
- If cache C = 8GB, optimal partition size = 6-7 layers per device
- Requires 16 layers ÷ 6-7 layers/device ≈ 3 devices minimum
- Paper uses 16 GPUs for parallel execution

### 6. Edge Case Handling

#### Oversized Single Layer
- **Condition**: Single layer size > C
- **Solutions**: 
  - Intra-layer partitioning (tensor parallelism)
  - Model compression (quantization, pruning)
  - Batch size reduction

#### Variable Layer Sizes
- **Problem**: Some layers significantly larger than others
- **Strategy**: Variable partition sizes, avoid under-utilization
- **Heuristic**: Balance partition sizes around mean layer size

### 7. Implementation Details

#### Memory Layout
```
Device Memory Map:
├─ [0x0000-0x1000): Weight tensors (read-only)
├─ [0x1000-0x4000): Activation buffers (read-write)
├─ [0x4000-0x5000): Temporary workspace
└─ [0x5000-0x8000): Communication buffers
```

#### Communication Protocol
- **Between partitions**: Point-to-point transfer via PCIe or NVLink
- **Transfer size**: Activation tensor (batch×seq×hidden)
- **Synchronization**: Barrier after each partition completion

### 8. Performance Model

#### Theoretical Speedup
```
Speedup = min(
  Sequential_time / Parallel_time,
  Memory_latency_reduction_factor
)

Where:
- Memory_latency_reduction = DRAM_latency / Cache_latency
- Parallel_time = max(partition_execution_time) + communication_overhead
```

#### Cache Hit Rate Impact
- **Perfect cache fit**: 100% SRAM/L2 hit rate
- **Baseline TP+PP**: ~60% cache hit rate (estimated)
- **Performance gain**: 20% TPS improvement demonstrated