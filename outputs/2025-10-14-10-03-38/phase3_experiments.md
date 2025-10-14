# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **GPUs**: 16 × NVIDIA H100 GPUs
- **Interconnect**: High-speed interconnect between GPUs (likely NVLink/NVSwitch)
- **Memory**: Each GPU has SRAM/L2 cache with capacity C (exact value not specified)

### Model Specifications

#### Dense Model (16-layer)
- **Architecture**: Fully connected dense network
- **Layers**: 16 transformer-style layers
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Sequence Length**: 10000
- **Model Dimensions**:
  - Number of heads: 16
  - Head dimension: 512
  - Hidden size: 16 × 512 = 8192
  - MLP hidden size: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total GPUs**: TP × PP = 8 × 2 = 16 GPUs
- **Strategy**: Standard tensor slicing and pipeline parallelism

### Proposed Configuration
- **Method**: Layer-wise partitioning
- **GPUs**: 16 GPUs
- **Strategy**: Partition 16 layers across 16 GPUs based on cache capacity

## Performance Metrics

### Measurement Details
- **Tokens Per Second (TPS)**: Number of output tokens generated per second
- **Time Per Output Token (TPOT)**: Average time to produce one output token (milliseconds)

### Results Table

| Model Type | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|------------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Analysis

### Improvement Metrics
- **TPS Improvement**: (15,360 - 12,800) / 12,800 = 20% increase
- **TPOT Reduction**: (0.078 - 0.065) / 0.078 = 16.67% ≈ 17% reduction

### Root Cause Analysis
- **Baseline Limitations**:
  - Does not consider on-chip memory constraints explicitly
  - More off-chip memory accesses due to tensor slicing
  - Communication delays between TP and PP stages
- **Proposed Advantages**:
  - Entire layer groups fit in SRAM/L2 cache
  - Minimal off-chip memory accesses
  - Reduced inter-card communication overhead
  - Better memory locality and cache utilization

### Memory Footprint Calculation (Estimated)
For the dense 16-layer model:
- **Per-layer parameters**: ~2.1B parameters per layer (estimated from dimensions)
- **Total model size**: ~33.6B parameters × 2 bytes = ~67.2 GB
- **Per-layer memory**: ~4.2 GB per layer (weights only)
- **With activations**: ~8-12 GB per layer (including activations and buffers)
- **Cache capacity C**: Must be >12 GB per layer to fit entire layers

### Experimental Validation
- **Test Environment**: Controlled inference benchmark
- **Measurement Method**: Average over multiple runs
- **Statistical Significance**: Results show consistent improvement across runs

## Reproducibility Details

### Key Parameters for Reproduction
1. **Model Architecture**: 16-layer transformer with specified dimensions
2. **Hardware**: 16× NVIDIA H100 with identical memory hierarchy
3. **Software**: CUDA-based implementation with custom partitioning logic
4. **Batch Configuration**: Fixed batch size 1024, sequence length 10000
5. **Precision**: FP16 throughout (weights, activations, computations)

### Baseline Implementation
- **Framework**: Likely Megatron-LM or similar for TP/PP
- **Communication**: NCCL for GPU-to-GPU transfers
- **Optimization**: Standard tensor parallelism optimizations applied

### Proposed Method Implementation
- **Partitioning**: Greedy algorithm for layer assignment
- **Memory Management**: Pre-allocation in SRAM/L2 cache
- **Communication**: Minimal inter-layer transfers between partitions