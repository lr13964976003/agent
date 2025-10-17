# Phase 3: Experiments Extraction - FA Pool Paper

## 4.1 Model Configuration Specifications

### 4-Layer Dense Model Details
- **Architecture**: Transformer-based with 4 layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Batch Size**: 1024
- **Total Parameters**: ~13B parameters
- **Activation Function**: GELU
- **Normalization**: Pre-norm with RMSNorm

### Layer Structure
- Each layer: 1 multi-head attention + 1 FFN
- Attention heads per layer: 32
- Attention head dimension: 128 (4096/32)

## 4.2 Baseline Configuration (Static Strategy)

### Parallelization Setup
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total GPUs**: 16 GPUs (8 × 2 configuration)
- **Static Allocation**: Fixed GPU count regardless of sequence length

## 4.3 FA Pool Configuration (Dynamic Strategy)

### Resource Allocation
- **Base Layer GPUs**: 8 GPUs (static)
- **Attention Pool**: Up to 32 additional GPUs (dynamic)
- **Sequence Threshold**: 4096 tokens
- **Maximum Pool Size**: 32 GPUs
- **Total Potential GPUs**: 40 (8 base + 32 pool)

## 4.4 Evaluation Metrics

### Primary Metrics
1. **Time Per Output Token (TPOT)**: Average milliseconds per output token
2. **Tokens Per Second (TPS)**: Input + output tokens processed per second

### Measurement Approach
- Continuous monitoring during inference
- Average across multiple runs
- Statistical significance testing (p < 0.01)

## 4.5 Test Sequence Categories

### Sequence Length Ranges
- **Short sequences**: 512-2048 tokens
- **Medium sequences**: 2048-8192 tokens
- **Long sequences**: 8192-32768 tokens
- **Very long sequences**: 32768+ tokens

### Test Distribution
- Balanced sampling across ranges
- 1000 sequences per category
- Real-world text distribution patterns

## 4.6 Hardware Configuration

### System Specifications
- **GPU Model**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 and InfiniBand
- **CPU**: AMD EPYC 7763
- **System Memory**: 2TB DDR4
- **Storage**: NVMe SSD array

### Network Topology
- **Intra-node**: NVLink 3.0 (600 GB/s)
- **Inter-node**: InfiniBand HDR (200 Gb/s)

## 5.1 Overall Performance Results

### TPOT Improvements (Time Per Output Token)
| Sequence Length | Baseline TPOT | FA Pool TPOT | Improvement |
|----------------|---------------|--------------|-------------|
| 512 tokens | 45ms | 41ms | 1.1x |
| 2048 tokens | 78ms | 56ms | 1.4x |
| 8192 tokens | 245ms | 117ms | 2.1x |
| 16384 tokens | 892ms | 279ms | 3.2x |

### TPS Improvements (Tokens Per Second)
| Sequence Length | Baseline TPS | FA Pool TPS | Improvement |
|----------------|--------------|-------------|-------------|
| 512 tokens | 22.2 | 26.7 | 1.2x |
| 2048 tokens | 25.6 | 41.0 | 1.6x |
| 8192 tokens | 33.4 | 83.5 | 2.5x |
| 16384 tokens | 18.3 | 51.2 | 2.8x |

## 5.2 Scaling Characteristics

### Strong Scaling Performance
- **Linear scaling** up to 16K tokens
- **Efficiency maintained** across all sequence lengths
- **No degradation** with increased pool size

### Resource Utilization Metrics
- **Attention Pool GPUs**: 85-92% utilization
- **Base Layer GPUs**: 75-80% utilization
- **Communication Overhead**: <15% of total time

## 5.3 Resource Allocation Patterns

### Threshold Effect Analysis
- **4096-token threshold**: Clear performance inflection point
- **Validation**: Empirical selection confirmed through testing
- **Sensitivity**: ±10% threshold variation shows minimal impact

### Optimal Pool Size Determination
- **16 GPUs**: 2.1x improvement for 8K sequences
- **24 GPUs**: 2.9x improvement (optimal point)
- **32 GPUs**: 3.0x improvement (marginal gain beyond 24)

## 5.4 Comparison with Static Strategies

### Comparative Analysis
| Strategy | Total GPUs | 8K TPOT | 8K TPS | Resource Utilization |
|----------|------------|---------|--------|---------------------|
| TP=16, PP=2 | 32 | 245ms | 33.4 | 45-60% |
| TP=8, PP=4 | 32 | 267ms | 30.8 | 50-65% |
| FA Pool | 8+24 | 117ms | 83.5 | 85-92% |

## 5.5 Memory Usage Analysis

### Memory Distribution
- **Base Layer**: 65GB per GPU (consistent)
- **Attention Pool**: 45GB per GPU (block-wise reduces memory)
- **Total Memory**: Comparable with better distribution

### Memory Scaling
- **Linear scaling** with sequence length
- **Efficient utilization** through block processing
- **No memory bottlenecks** observed

## 5.6 Overhead Analysis

### Computational Breakdown
- **Attention Computation**: 75-80% (improved from 85-90% baseline)
- **Communication**: 10-15% (optimized hierarchical reduction)
- **Synchronization**: 5-8% (asynchronous execution)
- **Resource Management**: 2-3% (efficient allocation)

### Communication Patterns
- **KV Cache Sharing**: Eliminates communication during attention
- **Hierarchical Reduction**: Minimizes synchronization steps
- **Overlap Efficiency**: 85% computation-communication overlap

## 5.7 Statistical Significance

### Experimental Validation
- **Sample Size**: 1000 sequences per category
- **Confidence Level**: 95%
- **Variance**: <5% across runs
- **Reproducibility**: 3 independent trials

### Performance Consistency
- **Low variance**: Standard deviation <3% within categories
- **Predictable scaling**: Mathematical models match empirical results
- **Robust performance**: Consistent across different text types