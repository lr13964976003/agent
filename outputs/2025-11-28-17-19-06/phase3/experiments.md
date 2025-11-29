# Experiments Extraction

## Experimental Setup

### Hardware Configuration
- Platform: 16 NVIDIA H100 GPUs
- Memory per GPU: 80GB HBM3
- Interconnect: NVLink 4.0, 900 GB/s bidirectional bandwidth
- Cache specifications per GPU:
  - L1 cache: 256 KB per SM
  - L2 cache: 50 MB
  - SRAM: Not explicitly specified (using L2 cache as proxy)

### Model Specifications
- **Dense Model**: 16-layer fully connected network
- **Total Parameters**: 30 billion (30B)
- **Precision**: BF16 (2 bytes per parameter)
- **Layer Configuration**:
  - Hidden size: 16384
  - Number of heads: 32
  - Head dimension: 128
  - MLP hidden size: 16384

### Runtime Parameters
- **Batch Size**: 128
- **Sequence Length**: 10,000
- **Total Input Tokens**: 1,280,000 per batch (128 × 10,000)

### Memory Calculations

#### Weight Memory per Layer
```
Dense layer weight size = hidden_size × hidden_size × 2 bytes
= 16384 × 16384 × 2 = 536,870,912 bytes ≈ 537 MB per layer

Total weight memory = 16 layers × 537 MB = 8.59 GB
```

#### Activation Memory per Layer
```
Activation size = batch_size × sequence_length × hidden_size × 2 bytes
= 128 × 10,000 × 16384 × 2 = 41,943,040,000 bytes ≈ 41.9 GB per layer
```

#### Total Memory per Layer
```
Total per layer = Weight + Activation + Buffer
≈ 537 MB + 41.9 GB + Buffer
≈ 42.4 GB + Buffer (dominated by activations)
```

## Baseline Configuration

### Tensor Parallelism + Pipeline Parallelism
- **TP=8**: Tensor parallelism across 8 GPUs
- **PP=2**: Pipeline parallelism across 2 stages
- **Total GPUs**: 8 × 2 = 16 GPUs
- **Memory Distribution**: Model split via tensor slicing and pipeline stages

## Proposed Method Configuration

### Layer-wise Partitioning
- **Cache Constraint**: L2 cache capacity (50 MB per GPU)
- **Challenge**: Single layer activation memory (41.9 GB) >> cache capacity
- **Solution**: Modified approach focusing on weight caching

### Practical Implementation
Given the massive activation sizes, the proposed method adapts to:
1. **Weight-optimized partitioning**: Focus on fitting layer weights in faster memory
2. **Activation streaming**: Manage activations through memory hierarchy
3. **Batch size optimization**: Potentially reduce batch size for cache fitting

## Performance Results

### Dense Model Results (16 layers)
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Improvements
- **TPS Improvement**: +20% (12,800 → 15,360)
- **TPOT Reduction**: -17% (0.078 → 0.065 ms)
- **Throughput Gain**: 3,560 additional tokens/second

## Analysis

### Baseline Limitations
1. **Memory Access Overhead**: TP/PP doesn't optimize for on-chip memory locality
2. **Communication Latency**: Frequent tensor slicing requires inter-GPU communication
3. **Load Imbalance**: Pipeline bubbles and uneven tensor distribution

### Proposed Method Advantages
1. **Cache Optimization**: Better utilization of fast on-chip memory
2. **Reduced Communication**: Layer-wise execution minimizes data transfers
3. **Improved Locality**: Sequential layer execution maximizes data reuse

### Scalability Considerations
- Method scales with model size and GPU count
- Cache-aware partitioning becomes critical for larger models
- Batch size adjustment enables cache fitting for various configurations

## Key Insights

1. **Memory Hierarchy Matters**: Explicit cache consideration provides significant gains
2. **Layer-wise Locality**: Sequential execution patterns optimize memory access
3. **Partitioning Efficiency**: Smart layer grouping outperforms naive parallelism
4. **Hardware-Software Co-design**: Matching algorithm to hardware characteristics crucial