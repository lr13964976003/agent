# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- 16 NVIDIA H100 GPUs
- Full utilization: 16 GPUs (8 × 2 = 16)

### Model Specifications
- **Type**: Dense model - 16-layer fully connected network
- **Total Parameters**: 30B parameters
- **Precision**: BF16 (2 bytes per parameter)
- **Layer Distribution**: 30B parameters ÷ 16 layers = 1.875B parameters per layer

### Configuration Parameters
```
Batch Size: 128
Sequence Length: 10000
Number of Heads: 32
Dimension per Head: 128
Hidden Size of MLP: 16384
Total Hidden Dimension: 32 × 128 = 4096
```

### Memory Calculations

#### Per Layer Weight Memory
```
1.875B parameters × 2 bytes (BF16) = 3.75 GB per layer
```

#### Activation Memory (per layer)
```
Batch Size: 128
Sequence Length: 10000
Hidden Dim: 4096
Activation Memory = 128 × 10000 × 4096 × 2 bytes (BF16) = 10.48 GB
```

#### Total Per Layer Memory
```
Weights: 3.75 GB
Activations: 10.48 GB
Buffers: ~0.5 GB (estimated)
Total: ~14.73 GB per layer
```

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs**: 8 × 2 = 16 GPUs

### Proposed Layer-wise Partitioning
- **Target**: Fit partitions within SRAM/L2 cache capacity C
- **Method**: Greedy layer aggregation
- **Layers per GPU**: Variable based on cache capacity

### Performance Metrics

#### Results Comparison
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

#### Performance Improvement
- **TPS Increase**: 20% (12,800 → 15,360)
- **TPOT Reduction**: 17% (0.078ms → 0.065ms)

### Cache Capacity Requirements
Given 14.73 GB per layer total memory:
- **Single Layer**: Requires C ≥ 14.73 GB
- **Two Layers**: Requires C ≥ 29.46 GB
- **Optimal Partitioning**: Based on actual H100 SRAM/L2 cache size

### Deployment Constraints
- Each partition must fit within GPU SRAM/L2 cache
- Contiguous layer assignment preserved
- Inter-GPU communication minimized
- Memory access latency optimized