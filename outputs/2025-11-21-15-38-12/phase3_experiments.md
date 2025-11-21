# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory**: 50MB L2 cache per GPU (SRAM)
- **Interconnect**: NVLink/NVSwitch for high-bandwidth communication

### Model Specifications
- **Type**: 16-layer fully connected dense network
- **Total parameters**: 30 billion (30B)
- **Precision**: BF16 (2 bytes per parameter)
- **Total model size**: 60 GB (30B × 2 bytes)

### Layer Architecture (per layer)
- **Hidden size**: 4096 (32 heads × 128 dimensions)
- **MLP intermediate size**: 16384
- **Sequence length**: 10000 tokens
- **Batch size**: 128 samples
- **Head count**: 32
- **Head dimension**: 128

### Memory Analysis (per layer)
```
Weights per layer: 30B params / 16 layers × 2 bytes = 3.75 GB
Activations per layer: 128 × 10000 × 4096 = 5.24 GB
Total per layer: 3.75 + 5.24 = 8.99 GB (excluding buffers)
Buffer overhead: ~0.5 GB (estimated)
Total per layer: ~9.5 GB
```

### Cache Constraint Challenge
- **Cache capacity**: 50 MB per GPU
- **Layer size**: 9.5 GB per layer
- **Challenge**: Single layer exceeds cache by 190x
- **Solution required**: Layer fusion, activation recomputation, or model compression

## Baseline Configuration
- **Method**: Tensor Parallelism + Pipeline Parallelism
- **Configuration**: TP=8, PP=2
- **Total GPUs**: 8 × 2 = 16 GPUs utilized
- **Distribution**: 
  - 8-way tensor parallelism within each pipeline stage
  - 2 pipeline stages across GPUs

## Proposed Method Configuration
- **Strategy**: Layer-wise partitioning with cache fitting
- **Implementation**: 
  - Partition layers into groups that fit cache constraints
  - Each partition assigned to dedicated GPU
  - Sequential execution within each partition
  - Inter-partition communication for activations

## Performance Results

### Dense Model Results
| Method                  | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------------------------|------|----------------|-----------|
| Baseline (TP=8, PP=2)   | 16   | 12,800         | 0.078     |
| Proposed Layer-wise     | 16   | 15,360         | 0.065     |

### Performance Improvement
- **Throughput gain**: (15,360 - 12,800) / 12,800 = 20% increase in TPS
- **Latency reduction**: (0.078 - 0.065) / 0.078 = 17% reduction in TPOT

## Analysis of Results
- **Efficiency gain**: Reduced off-chip memory access via cache utilization
- **Communication overhead**: Minimized by sequential layer processing
- **Scalability**: Linear scaling with additional GPUs for layer distribution

## Experimental Validations
- **Memory profiling**: Verified cache utilization per partition
- **Communication measurement**: Inter-GPU transfer frequency and bandwidth
- **Load balancing**: Equal distribution of layers across GPUs achieved

## Limitations Identified
- **Cache size constraint**: Requires careful optimization for large layers
- **Batch size sensitivity**: May need adjustment for cache fitting
- **Model architecture**: Best suited for uniform layer sizes

## Reproducibility Factors
- **Hardware**: Requires 16× H100 GPUs with NVLink
- **Software**: CUDA toolkit with optimized memory management
- **Environment**: Controlled thermal conditions for consistent cache performance