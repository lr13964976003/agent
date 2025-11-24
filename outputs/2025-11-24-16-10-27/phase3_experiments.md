# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **GPUs**: 16×NVIDIA H100 GPUs with NVLink + NVSwitch
- **Memory**: 80GB HBM3 per GPU (total 1.28TB)
- **Interconnect**: NVLink 4.0 (900GB/s bidirectional)
- **Network**: NVSwitch fabric for all-to-all communication
- **Precision**: BF16 (16-bit Brain Floating Point)

### Model Architecture
- **Type**: Dense Transformer
- **Layers**: 16 transformer layers (confirmed)
- **Hidden size**: 4096 (d_model)
- **Attention heads**: 32
- **Head dimension**: 128
- **MLP hidden size**: 16,384
- **Total parameters**: ~13B (estimated based on configuration)

### Runtime Parameters
- **Sequence length**: 100,000 tokens (fixed)
- **Batch size**: 128 (fixed)
- **Precision**: BF16 throughout
- **Warmup iterations**: 10 (for cache warmup)
- **Measurement iterations**: 100 (for stable measurements)
- **Synchronization**: CUDA synchronization after each iteration
- **Measurement methodology**: Average of 100 iterations after warmup

## Evaluation Metrics

### Primary Metrics
- **TPS (Tokens Per Second)**: Raw throughput measurement
  - Calculation: (batch_size × sequence_length) / inference_time
  - Higher values indicate better performance
- **TPOT (Time Per Output Token)**: Average latency per token
  - Calculation: inference_time / (batch_size × sequence_length)
  - Lower values indicate better performance

### Secondary Metrics
- **Memory utilization**: Peak memory usage per GPU
- **Communication overhead**: Time spent in NCCL operations
- **Compute efficiency**: GPU utilization percentage

## Benchmarking Results

### Baseline Configuration (TP=8, PP=2)
- **TPS**: 1.20M tokens/second
- **TPOT**: 0.85 ms
- **Memory usage**: ~78GB per GPU (near capacity)
- **Setup**:
  - 8-way tensor parallelism across model dimensions
  - 2-way pipeline parallelism across 16 layers (8 layers per stage)
  - No sequence parallelism (full sequence on each device)

### RA+SP Configuration (Ring Size=16)
- **TPS**: 1.45M tokens/second
- **TPOT**: 0.70 ms
- **Memory usage**: ~5GB per GPU (significant reduction)
- **Setup**:
  - 16-way sequence parallelism
  - 16-way ring attention across all GPUs
  - Each GPU processes 6,250 tokens (100,000/16)

## Performance Analysis

### Throughput Improvements
- **TPS improvement**: 20.8% (1.45M vs 1.20M)
- **TPOT reduction**: 17.6% (0.70ms vs 0.85ms)
- **Memory reduction**: 93.6% (5GB vs 78GB per GPU)

### Communication Analysis
- **Baseline communication**: All-gather operations for tensor parallelism
- **RA+SP communication**: Ring-based K/V exchanges
- **Communication volume per stage**: 6.25K tokens × 4096 dimensions × 2 (K+V)
- **Total communication**: 16 stages × 51.2MB = 819.2MB per GPU

### Scalability Characteristics
- **Optimal sequence length**: L > 16K tokens (as specified)
- **Scalability**: Benefits increase with sequence length and device count
- **Network efficiency**: Better utilization of NVLink bandwidth
- **Memory efficiency**: Linear reduction with device count

## Reproducibility Details

### Environment Setup
```bash
# CUDA version
CUDA 12.1 with cuDNN 8.9

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Precision settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

### Synchronization Protocol
1. **Warmup phase**: 10 iterations to warm CUDA caches
2. **Measurement phase**: 100 iterations for stable measurements
3. **Synchronization**: `cudaDeviceSynchronize()` after each iteration
4. **Timing**: `cudaEvent_t` for precise timing measurements

### Statistical Significance
- **Measurement iterations**: 100 (sufficient for statistical significance)
- **Standard deviation**: <2% across runs
- **Confidence interval**: 95% confidence for reported values
- **Outlier handling**: No outliers removed (consistent performance)

## Future Extensions
- **Training scenario**: Gradient communication to be added
- **Hierarchical topologies**: Combining intra-node rings with inter-node scheduling
- **Adaptive precision**: Dynamic precision adjustment based on layer sensitivity
- **Kernel fusion**: Enhanced fused kernels for projection and communication