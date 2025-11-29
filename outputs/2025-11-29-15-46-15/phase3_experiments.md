# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **GPUs**: 16 NVIDIA H100 GPUs
- **Interconnect**: High-speed GPU-to-GPU communication
- **Memory**: Each GPU has SRAM/L2 cache with capacity C

### Model Configuration
- **Type**: Dense 16-layer fully connected network
- **Total Parameters**: 30 billion (30B)
- **Precision**: BF16 (2 bytes per parameter)
- **Total Weight Memory**: ~60 GB (30B × 2 bytes)

### Hyperparameters
- **Batch Size**: 128
- **Sequence Length**: 10000
- **Attention Heads**: 32
- **Head Dimension**: 128
- **MLP Hidden Size**: 16384

## Baseline Comparison

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs**: 16 (8 × 2 = 16)

### Baseline Performance
- **Tokens Per Second (TPS)**: 12,800 tokens/s
- **Time Per Output Token (TPOT)**: 0.078 milliseconds

## Proposed Method Results

### Layer-wise Partitioning
- **Partition Strategy**: Cache-constrained layer grouping
- **Target**: Each partition fits within SRAM/L2 cache
- **Execution**: Sequential layer execution within each partition

### Performance Results
| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| TPS (tokens/s) | 12,800 | 15,360 | +20% |
| TPOT (ms) | 0.078 | 0.065 | -17% |

## Analysis of Results

### Performance Gains
1. **Throughput Improvement**: 20% increase in TPS
   - From 12,800 to 15,360 tokens per second
   - Equivalent to 2,560 additional tokens per second

2. **Latency Reduction**: 17% decrease in TPOT
   - From 0.078ms to 0.065ms per token
   - 0.013ms savings per token

### Root Cause Analysis

#### Memory Access Efficiency
- **Proposed Method**: Layers fit entirely in SRAM/L2 cache
- **Baseline**: Frequent off-chip memory accesses
- **Impact**: Reduced memory access latency

#### Communication Overhead
- **Proposed Method**: Minimal inter-GPU communication
- **Baseline**: Regular tensor parallelism synchronization
- **Impact**: Less communication delay

#### Cache Utilization
- **Proposed Method**: Maximized on-chip memory usage
- **Baseline**: Underutilized fast memory hierarchy
- **Impact**: Better memory locality

## Technical Implementation Details

### Memory Footprint Calculation
For the dense 16-layer model:
- **Per-layer weight memory**: ~3.75 GB (30B params ÷ 16 layers × 2 bytes)
- **Activation memory**: Depends on batch size and sequence length
- **Buffer memory**: Operator-specific workspace

### Partition Assignment
Given cache capacity C, the number of layers per partition:
```
layers_per_partition = floor(C / memory_per_layer)
```

### GPU Mapping
- **Total Partitions**: Determined by cache constraints
- **GPU Assignment**: One partition per GPU
- **Execution Order**: Sequential across GPUs

## Scalability Considerations

### Model Size Adaptation
- Method scales with model size
- Cache capacity determines partition size
- Dynamic partitioning for varying architectures

### Hardware Configuration
- Works with different GPU counts
- Adapts to varying cache sizes
- Compatible with different GPU architectures

## Reproducibility Factors

### Critical Parameters
- Exact cache capacity C
- Memory estimation accuracy
- Batch size optimization
- Layer size distribution

### Measurement Methodology
- TPS calculated over sustained execution
- TPOT averaged across all output tokens
- Multiple runs for statistical significance
- Warm-up periods excluded from measurements