# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **Device**: 16 × NVIDIA H100 GPUs
- **Total GPUs**: 16
- **Memory Type**: SRAM/L2 cache (exact capacity not specified, but constraint applies)

### Model Configuration
- **Model Type**: Dense model (fully connected dense network)
- **Layer Count**: 16 layers
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Sequence Length**: 10,000
- **Head Configuration**: 
  - Number of heads: 16
  - Dimension per head: 512
  - Total hidden size: 16 × 512 = 8,192
- **MLP Configuration**: 
  - Hidden size of MLP: 32,768

### Baseline Configuration
- **Parallel Strategy**: Tensor Parallelism + Pipeline Parallelism
- **TP (Tensor Parallelism)**: 8
- **PP (Pipeline Parallelism)**: 2
- **Total GPU Utilization**: 8 × 2 = 16 GPUs (full platform)

### Performance Metrics
1. **TPS (Tokens Per Second)**: Number of output tokens generated per second
2. **TPOT (Time Per Output Token)**: Average time to produce single output token (milliseconds)

## Experimental Results

### Dense Model (16-layer) Performance Comparison

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|---------|------|-----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Analysis

#### Throughput Improvement
- **TPS Gain**: 15,360 - 12,800 = 2,560 tokens/s additional
- **Percentage Improvement**: (2,560/12,800) × 100 = **20% increase**

#### Latency Reduction
- **TPOT Reduction**: 0.078 - 0.065 = 0.013 ms per token
- **Percentage Reduction**: (0.013/0.078) × 100 ≈ **16.67% reduction** (reported as 17%)

#### Performance Attribution
- **Root Cause**: More efficient on-chip memory utilization
- **Mechanism**: Reduced off-chip memory accesses and communication delays
- **Comparison**: Baseline TP=8, PP=2 doesn't explicitly consider on-chip memory constraints

## Experimental Design Details

### Memory Calculation Verification
For dense model (16 layers):
- Each layer memory footprint = weight_size + activation_size + buffer_size
- Activation memory per layer = 1024 × 10,000 × hidden_size
- Weight memory per layer = hidden_size × hidden_size (for dense layers)
- Total model memory = Σ size(lⱼ) for j=1 to 16

### Partitioning Results
Since 16 layers distributed across 16 GPUs in proposed method:
- **Partitioning Scheme**: Likely 1 layer per GPU (16 partitions for 16 layers on 16 devices)
- **Cache Constraint**: Each single layer must fit within H100's L2 cache/SRAM
- **Verification**: size(single_layer) ≤ C (H100 cache capacity)

### Baseline vs Proposed Resource Mapping

#### Baseline (TP=8, PP=2)
- **Tensor Parallelism**: Splits each layer across 8 GPUs (8-way tensor parallel)
- **Pipeline Parallelism**: Splits model into 2 pipeline stages (2-way pipeline)
- **Total GPUs**: 8 × 2 = 16 GPUs utilized
- **Memory Access**: Frequent inter-GPU communication within layers and between stages

#### Proposed Layer-wise
- **Partitioning**: 16 layers → 16 partitions
- **Mapping**: 1 partition per GPU (1 layer per GPU)
- **Memory**: Each layer entirely in single GPU's cache
- **Communication**: Only between adjacent layers on different GPUs

## Reproducibility Parameters

### Fixed Parameters (must be maintained for reproduction)
- Model: 16-layer dense network
- Precision: FP16 (2 bytes)
- Batch size: 1024 (cannot be changed)
- Sequence length: 10,000 (fixed)
- Head count: 16
- Head dimension: 512
- MLP hidden: 32,768

### Measurable Outputs
- TPS measurement methodology
- TPOT calculation approach
- Warm-up iterations (not specified but required)
- Averaging methodology (not specified but implied)