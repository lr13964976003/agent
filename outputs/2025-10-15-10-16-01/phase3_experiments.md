# Phase 3: Experiments Extraction - Helix Paper

## Complete Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability
- **Total devices**: 16 GPUs fully utilized

### Model Specifications
- **Model type**: 2-layer Dense Transformer
- **Fixed parameters**:
  - Batch size: 1024
  - Sequence length: 10000
  - Number of heads: 16
  - Dimension per head: 512
  - Hidden size of MLP: 32768
  - Total embedding dimension: 8192 (16 × 512)

### Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total devices**: TP × PP = 8 × 2 = 16 GPUs
- **Description**: Widely adopted method for large-scale model deployment

### Proposed Configuration
- **Method**: Two-level attention partitioning
- **Partitioning**: m × n = 16
- **m**: 4 (intra-head dimension splits)
- **n**: 4 (head groups)
- **Total partitions**: 16 (maps to 16 GPUs)
- **Heads per group**: h/n = 16/4 = 4 heads
- **Dimension per segment**: d/m = 512/4 = 128

### Evaluation Metrics

#### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
   - Measures overall system performance
   - Higher values indicate better performance

2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token
   - Measured in milliseconds (ms)
   - Lower values indicate reduced overhead

### Experimental Results

#### Dense Transformer Results
| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 2-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 2-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |

### Performance Analysis

#### Throughput Improvement
- **Absolute improvement**: 1,580,000 - 1,200,000 = 380,000 tokens/sec
- **Relative improvement**: (380,000/1,200,000) × 100 = **31.7% increase**

#### Communication Overhead Reduction
- **Absolute reduction**: 0.35 - 0.22 = 0.13 ms
- **Relative reduction**: (0.13/0.35) × 100 = **37.1% decrease**

### Hardware Utilization Analysis
- **Baseline**: Uses 16 GPUs with TP=8 + PP=2
- **Proposed**: Uses 16 GPUs with m×n=16 partitions
- **Utilization**: Both methods fully utilize 16 GPUs
- **Difference**: Proposed method achieves better load balancing through finer granularity

### Experimental Validations
- **Precision consistency**: Both methods use FP16 mixed precision
- **Batch size saturation**: Large batch size (1024) ensures GPU saturation
- **Sequence length**: Fixed at 10000 tokens for fair comparison
- **Model consistency**: Same 2-layer Dense Transformer used across tests

### Discussion Points
- **Scalability**: Proposed method scales beyond head count limitations
- **Load balancing**: Finer granularity enables better distribution
- **Communication efficiency**: Reduced synchronization costs through localized computation
- **Memory efficiency**: Each device handles smaller parameter subsets

### Experimental Reproducibility Parameters
- **Random seed**: Not specified in paper
- **Warmup steps**: Not specified in paper
- **Measurement duration**: Not specified in paper
- **Number of runs**: Not specified in paper
- **Statistical significance**: Not reported in paper

### Limitations Noted
- **Training scenario**: Not tested (only inference)
- **Adaptive partitioning**: Not explored (fixed m and n)
- **Hardware topology**: Not explicitly considered in partitioning decisions
- **Model types**: Only tested on 2-layer Dense Transformer