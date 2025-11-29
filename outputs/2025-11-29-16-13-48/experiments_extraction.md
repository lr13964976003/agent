# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **16 NVIDIA H100 GPUs** (full utilization)
- High-bandwidth interconnect between GPUs
- Each GPU has SRAM/L2 cache capacity (40-80MB typical)

### Model Configuration
- **Dense Model**: 4-layer fully connected network
- **Weight Size**: 30B parameters total
- **Precision**: BF16 (2 bytes per parameter)
- **Total Weight Memory**: ~60GB (30B × 2 bytes)

### Test Parameters
- **Batch Size**: 128
- **Sequence Length**: 10,000
- **Number of Heads**: 32
- **Head Dimension**: 128
- **MLP Hidden Size**: 16,384

## Memory Calculation Details

### Per-Layer Memory Breakdown
```
Layer 1: ~15GB (weights) + activations + buffers
Layer 2: ~15GB (weights) + activations + buffers  
Layer 3: ~15GB (weights) + activations + buffers
Layer 4: ~15GB (weights) + activations + buffers
```

### Activation Memory
```
Activation_size = batch_size × sequence_length × hidden_size × datatype_size
= 128 × 10,000 × hidden_size × 2 bytes
```

## Baseline Configuration

### Tensor Parallelism + Pipeline Parallelism
- **TP=8, PP=2** (8 × 2 = 16 GPUs total)
- Tensor parallelism splits layers across 8 devices
- Pipeline parallelism creates 2 pipeline stages
- Does NOT consider on-chip memory constraints explicitly

## Results Analysis

### Performance Metrics
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (4-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (4-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Improvements Achieved
- **TPS Improvement**: 20% increase (15,360 vs 12,800)
- **TPOT Reduction**: 17% decrease (0.065ms vs 0.078ms)
- **Throughput Gain**: 3,560 additional tokens per second

## Performance Analysis

### Why Layer-wise Outperforms Baseline
1. **Reduced Memory Access Latency**: Fits partitions in SRAM/L2 cache
2. **Minimized Off-chip Accesses**: Avoids DRAM bottlenecks
3. **Better Locality**: Contiguous layer execution on same device
4. **Reduced Communication**: Fewer inter-device transfers

### Communication Pattern Comparison
- **Baseline**: Frequent tensor parallelism communication across 8 devices
- **Proposed**: Only boundary transfers between layer groups
- **Cache Efficiency**: 100% on-chip memory utilization per partition

## Scalability Considerations

### Model Size Scaling
- Method applicable to larger models (16, 32, 64+ layers)
- Cache capacity determines partition size, not total model size
- Can handle mixed layer sizes through dynamic programming

### Hardware Scaling
- Works with any number of available GPUs
- Partition count adapts to hardware resources
- Can combine with other parallelism strategies

## Experimental Validation

### Memory Footprint Verification
- Each partition validated to fit within cache capacity
- Runtime memory profiling confirms estimates
- No cache overflow detected during execution

### Reproducibility Requirements
- Exact layer memory calculations provided
- Hardware specifications documented
- Baseline configuration fully specified
- Performance metrics clearly defined