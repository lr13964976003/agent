# Phase 3: Detailed Experiments

## Experimental Setup

### Hardware Configuration
- System: 16 NVIDIA H100 GPUs
- Precision: Mixed precision (FP16)
- Network: High-bandwidth interconnect for inter-GPU communication

### Model Specifications
- **Model Type**: 2-layer Dense Transformer
- **Batch Size**: 1024 (fixed)
- **Sequence Length**: 10000 (fixed)
- **Attention Heads**: 16 (fixed)
- **Head Dimension**: 512 (fixed)
- **MLP Hidden Size**: 32768 (fixed)
- **Total Embedding Dimension**: 8192 (16×512)

### Tested Configurations

#### Baseline Configuration
- **Method**: Tensor Parallelism + Pipeline Parallelism
- **Tensor Parallelism Degree (TP)**: 8
- **Pipeline Parallelism Degree (PP)**: 2
- **Total Devices**: TP × PP = 8 × 2 = 16 GPUs
- **Strategy**: 
  - Splits model layers across pipeline stages
  - Splits individual layers across tensor parallel groups

#### Proposed Configuration
- **Method**: Two-level Attention Partitioning
- **Partitioning**: m×n = 16
  - m = 4 (dimension partitions per head)
  - n = 4 (head partitions)
- **Total Devices**: m × n = 4 × 4 = 16 GPUs
- **Strategy**:
  - 4 head groups (n=4), each with 4 heads (16/4=4)
  - Each head dimension split into 4 slices (512/4=128)

## Performance Metrics

### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

### Results Table

| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 2-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 2-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |

## Performance Analysis

### Throughput Improvement
- **Absolute Improvement**: 1,580,000 - 1,200,000 = 380,000 tokens/sec
- **Relative Improvement**: (380,000/1,200,000) × 100 = **31.7% increase**

### Communication Overhead Reduction
- **Absolute Reduction**: 0.35 - 0.22 = 0.13 ms
- **Relative Reduction**: (0.13/0.35) × 100 = **37.1% decrease**

### Hardware Utilization
- **Baseline**: Uses 16 GPUs through TP+PP combination
- **Proposed**: Uses 16 GPUs through m×n partitioning
- **Efficiency Gain**: Better load balancing due to finer granularity

## Experimental Validations

### Reproducibility Conditions
- Fixed batch size ensures consistent GPU memory usage
- Fixed sequence length maintains attention computation complexity
- FP16 precision eliminates precision-related performance variations
- Identical model architecture ensures fair comparison

### Bottleneck Analysis
- **Baseline Limitations**:
  - Tensor parallelism limited by layer dimensions
  - Pipeline parallelism introduces bubble overhead
  - Coarser granularity leads to load imbalance

- **Proposed Advantages**:
  - Finer granularity enables better load distribution
  - Reduced communication through localized computation
  - Better scaling with increasing device count

## Scalability Implications

### Scaling Beyond 16 Devices
- Proposed method can scale to m×n devices where m×n > h
- Traditional head-wise partitioning limited to h devices maximum
- Enables deployment on very large clusters without architectural constraints

### Memory Efficiency
- Per-device memory footprint reduced by factor of m×n = 16
- Enables deployment of larger models on same hardware
- Reduces memory pressure for gradient storage during training