# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **GPUs**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability balance

### Model Specifications
- **Model Type**: 4-layer Dense Transformer
- **Fixed Parameters**:
  - Batch size: 128
  - Sequence length: 10000
  - Number of heads: 32
  - Dimension per head: 128
  - MLP hidden size: 16384

### Baseline Configuration
**Traditional Approach**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- Tensor Parallelism degree: 8
- Pipeline Parallelism degree: 2
- **Total devices utilized**: 16 GPUs
- This represents the widely adopted method for large-scale model deployment

### Proposed Method Configuration
- **Partitioning**: m×n = 16 partitions
- **Mapping**: Each partition assigned to one GPU (16 total)
- **Partition calculation**: Since h=32 heads, possible configurations include:
  - m=1, n=16 (16 head groups of 2 heads each)
  - m=2, n=8 (8 head groups of 4 heads each, 2 dimension slices)
  - m=4, n=4 (4 head groups of 8 heads each, 4 dimension slices)
  - m=8, n=2 (2 head groups of 16 heads each, 8 dimension slices)
  - m=16, n=1 (1 head group of 32 heads, 16 dimension slices)

## Performance Metrics

### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

### Results Comparison

| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |

## Performance Analysis

### Throughput Improvements
- **Absolute improvement**: 1,580,000 - 1,200,000 = 380,000 tokens/sec
- **Relative improvement**: 31.7% increase in throughput

### Communication Overhead Reduction
- **Absolute reduction**: 0.35 - 0.22 = 0.13 ms per token
- **Relative reduction**: 37.1% decrease in synchronization overhead

### Hardware Utilization
- **Baseline**: Uses 16 GPUs as 8×TP + 2×PP
- **Proposed**: Maps 16 partitions to 16 GPUs directly
- **Utilization**: Full GPU utilization achieved through fine-grained partitioning

## Experimental Validation

### Factors Contributing to Performance
1. **Load balancing**: Even distribution across head count and feature dimensions
2. **Communication efficiency**: Reduced cross-device synchronization
3. **Memory distribution**: Lower memory footprint per device
4. **Granularity**: Finer partitioning enables better hardware utilization

### Precision Impact
- FP16 precision maintains numerical stability while maximizing throughput
- Large batch size (128) ensures GPU saturation
- Configuration balances compute and memory bandwidth utilization

## Experimental Constraints

### Fixed Parameters
- Batch size held constant at 128 for fair comparison
- Sequence length fixed at 10000 across all tests
- Model architecture (4-layer Dense) unchanged
- Precision (FP16) consistent across methods

### Scalability Demonstrated
- Successfully scales to 16 GPUs
- Validated for inference workloads
- Framework supports both training and inference adaptation

## Reproducibility Details

### Key Configuration for Reproduction
- **Hardware**: 16× NVIDIA H100
- **Model**: 4-layer Dense Transformer
- **Parameters**: h=32, d=128, D=4096, MLP_hidden=16384
- **Precision**: FP16
- **Batch**: 128
- **Sequence**: 10000 tokens
- **Baseline**: TP=8, PP=2
- **Proposed**: m×n=16 partitions mapped to 16 devices

### Performance Expectation
- Expected throughput: 1.58M tokens/sec
- Expected TPOT: 0.22 ms
- Improvement over baseline: 31.7% throughput, 37.1% overhead reduction

## Limitations and Considerations

### Experimental Scope
- Validated for inference only (training extension mentioned as future work)
- Dense transformer architecture specifically tested
- Fixed model size and batch configuration
- Requires 16 GPUs for full comparison

### Practical Considerations
- Optimal m,n values depend on hardware topology
- Network bandwidth affects actual performance
- Model-specific characteristics may influence partitioning choices