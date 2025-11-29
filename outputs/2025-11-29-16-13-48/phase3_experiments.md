# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- 16 NVIDIA H100 GPUs
- BF16 precision
- Target cache: SRAM/L2 cache per GPU

### Model Configuration
- **Dense model**: 4-layer fully connected network
- **Model weight size**: 30B parameters total
- **Batch size**: 128
- **Sequence length**: 10000
- **Number of heads**: 32
- **Dimension per head**: 128
- **Hidden size of MLP**: 16384

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs**: 16 (8 × 2 = 16)

### Proposed Method
- **Layer-wise partitioning**: Apply greedy/dynamic programming algorithm
- **Constraint**: Each partition must fit in GPU SRAM/L2 cache
- **Deployment**: 16 GPUs with optimized layer distribution

## Performance Metrics

### Key Measurements
1. **Tokens Per Second (TPS)**: Number of output tokens generated per second
2. **Time Per Output Token (TPOT)**: Average time to produce single token (milliseconds)

## Results

### Dense Model (4-layer)
| Method | GPUs | TPS (tokens/s) | TPOT (ms) | Improvement |
|--------|------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | - |
| Proposed Layer-wise | 16 | 15,360 | 0.065 | +20% TPS, -17% TPOT |

## Analysis

### Performance Gains
- **20% increase in TPS**: From 12,800 to 15,360 tokens/second
- **17% reduction in TPOT**: From 0.078ms to 0.065ms per token
- **Root cause**: More efficient on-chip memory utilization, reduced memory access latency

### Baseline Limitations
- TP=8, PP=2 doesn't explicitly consider on-chip memory constraints
- Results in more off-chip memory accesses
- Higher communication delays between devices

### Memory Utilization
- Proposed method ensures partitions fit entirely in SRAM/L2 cache
- Minimizes expensive off-chip DRAM accesses
- Better locality preservation during execution

## Experimental Validation

### Model Memory Breakdown
For 30B parameter model in BF16 (2 bytes per parameter):
- **Total weight memory**: ~60GB
- **Per layer average**: ~15GB (4 layers)
- **With activations and buffers**: Higher total per layer

### Partitioning Strategy
- Greedy algorithm applied to fit layers within cache constraints
- Dynamic programming for balanced load distribution
- Contiguous layer assignment preserved

## Key Findings

1. **Cache-conscious partitioning** significantly improves performance
2. **Reduced memory hierarchy traversal** leads to lower latency
3. **Parallel execution** on multiple cards increases throughput
4. **Scalability** demonstrated across 16 GPU configuration

## Limitations and Future Work

- Current evaluation limited to inference stage
- Training workload extension needed
- Adaptive partitioning for varying batch sizes
- Validation on larger, more complex models required