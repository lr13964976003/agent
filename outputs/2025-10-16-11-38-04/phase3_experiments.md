# Phase 3: Detailed Experiments Extraction - FA Pool

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Dense transformer
- **Hidden Dimension**: 4096
- **Attention Heads**: 32 heads × 128 dimensions/head = 4096 total
- **Feed-forward Dimension**: 16384 (4× hidden dimension)
- **Total Parameters**: ~13B
- **Batch Size**: 1024 sequences
- **Activation Function**: GELU
- **Normalization**: Pre-norm RMSNorm

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8-way across 8 GPUs
- **Pipeline Parallelism (PP)**: 2-way across 2 stages
- **Total Baseline GPUs**: 16 GPUs (8×2 grid)
- **GPU Model**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 + InfiniBand

### FA Pool Configuration
- **Base Layer**: 8 GPUs (maintaining 8-way tensor parallelism)
- **Attention Pool**: Dynamic allocation up to 32 GPUs
- **Maximum Total GPUs**: 40 (8 base + 32 pool)
- **Sequence Threshold**: 4096 tokens
- **Hardware**: Same as baseline (A100 80GB, NVLink 3.0, InfiniBand)

## Test Sequence Specifications

### Sequence Length Categories
1. **Short Sequences**: 512-2048 tokens
2. **Medium Sequences**: 2048-8192 tokens
3. **Long Sequences**: 8192-32768 tokens
4. **Very Long Sequences**: 32768+ tokens

### Test Points
- 512 tokens
- 1024 tokens
- 2048 tokens
- 4096 tokens (threshold)
- 8192 tokens
- 16384 tokens
- 32768 tokens

## Performance Results

### Time Per Output Token (TPOT) - Detailed
| Sequence Length | Baseline (ms) | FA Pool (ms) | Improvement | Pool GPUs |
|----------------|---------------|--------------|-------------|-----------|
| 512 tokens     | 45            | 41           | 1.1×        | 0         |
| 1024 tokens    | 52            | 46           | 1.1×        | 0         |
| 2048 tokens    | 78            | 56           | 1.4×        | 8         |
| 4096 tokens    | 145           | 89           | 1.6×        | 8         |
| 8192 tokens    | 245           | 117          | 2.1×        | 16        |
| 16384 tokens   | 892           | 279          | 3.2×        | 24        |
| 32768 tokens   | 3240          | 1080         | 3.0×        | 32        |

### Tokens Per Second (TPS) - Detailed
| Sequence Length | Baseline (TPS) | FA Pool (TPS) | Improvement | Pool GPUs |
|----------------|----------------|---------------|-------------|-----------|
| 512 tokens     | 22.2           | 26.7          | 1.2×        | 0         |
| 1024 tokens    | 19.2           | 21.7          | 1.1×        | 0         |
| 2048 tokens    | 25.6           | 41.0          | 1.6×        | 8         |
| 4096 tokens    | 27.6           | 44.9          | 1.6×        | 8         |
| 8192 tokens    | 33.4           | 83.5          | 2.5×        | 16        |
| 16384 tokens   | 18.3           | 51.2          | 2.8×        | 24        |
| 32768 tokens   | 10.1           | 30.3          | 3.0×        | 32        |

## Resource Utilization Analysis

### GPU Utilization by Component
- **Base Layer (8 GPUs)**:
  - FFN Computation: 85-90% utilization
  - Communication: 5-8% overhead
  - Idle Time: 2-5%

- **Attention Pool (variable GPUs)**:
  - Attention Computation: 85-92% utilization
  - Communication: 8-12% overhead
  - Synchronization: 2-3%

### Memory Usage Breakdown
| Component | Baseline (GB/GPU) | FA Pool Base (GB/GPU) | FA Pool (GB/GPU) |
|-----------|-------------------|------------------------|-------------------|
| Parameters | 1.6 | 1.6 | 0 |
| Activations | 35 | 32 | 25 |
| KV Cache | 25 | 20 | 15 |
| Attention | 10 | 8 | 5 |
| Total | 65 | 65 | 45 |

## Communication Overhead Analysis

### Baseline Communication
- **TP Communication**: All-reduce operations every attention layer
- **PP Communication**: Send/receive activations between stages
- **Total Overhead**: 20-25% of total time

### FA Pool Communication
- **KV Broadcast**: O(n×d) to all pool GPUs
- **Result Gather**: O(n×d/p) concatenation
- **Synchronization**: O(log p) hierarchical reduction
- **Total Overhead**: 10-15% of total time

## Scaling Characteristics

### Strong Scaling efficiency
- **8 pool GPUs**: 85% efficiency
- **16 pool GPUs**: 82% efficiency
- **24 pool GPUs**: 78% efficiency
- **32 pool GPUs**: 74% efficiency

### Optimal pool size determination
- **16 GPUs**: Best performance per GPU for 8K sequences
- **24 GPUs**: Best performance per GPU for 16K sequences
- **32 GPUs**: Diminishing returns beyond 24 GPUs

## Overhead Breakdown (8192 tokens)
| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Attention Computation | 88 | 75.2% |
| FFN Computation | 12 | 10.3% |
| Communication | 11 | 9.4% |
| Synchronization | 4 | 3.4% |
| Resource Management | 2 | 1.7% |
| Total | 117 | 100% |

## Comparison with Static Strategies

### Equivalent GPU Count Comparison
| Strategy | GPUs | TPOT@8K (ms) | TPS@8K | Efficiency |
|----------|------|--------------|--------|------------|
| TP=16, PP=2 | 32 | 245 | 33.4 | 100% |
| TP=8, PP=4 | 32 | 298 | 27.2 | 82% |
| FA Pool | 24 | 117 | 83.5 | 210% |

### Memory Efficiency Comparison
| Strategy | Memory/Seq (GB) | Total Memory (GB) | Memory Efficiency |
|----------|-----------------|-------------------|-------------------|
| TP=16, PP=2 | 1040 | 2080 | 1.0× |
| TP=8, PP=4 | 720 | 1440 | 1.4× |
| FA Pool | 360 | 1080 | 2.9× |

## Energy and Cost Analysis

### Power Consumption
- **Baseline (16 GPUs)**: 6.4 kW peak
- **FA Pool (24 GPUs)**: 9.6 kW peak
- **Performance per watt**: 1.8× improvement

### Cost Effectiveness
- **Hardware Cost Ratio**: 1.5× (24 vs 16 GPUs)
- **Performance Improvement**: 2.1× at 8K tokens
- **ROI**: Positive for sequence lengths > 4K tokens

## Experimental Validation

### Reproducibility Metrics
- **Standard Deviation**: <5% across 5 runs
- **Confidence Interval**: 95% for all reported metrics
- **System Stability**: No failures during 72-hour stress test

### Edge Case Testing
- **Empty Sequences**: Handled correctly with 0 pool GPUs
- **Maximum Length**: 65536 tokens tested successfully
- **Mixed Length Batches**: Pool size determined by max length

### Error Analysis
- **Measurement Error**: ±2ms for TPOT, ±0.5 TPS
- **Statistical Significance**: p < 0.01 for all improvements
- **System Overhead**: <3% measurement overhead