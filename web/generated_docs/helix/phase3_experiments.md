# Helix: Two-Level Attention Partitioning - Experimental Details

## Experimental Setup

### Hardware Configuration
- **GPUs**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Software**: Compatible with existing model parallel frameworks

### Model Specifications
- **Architecture**: 2-layer Dense Transformer
- **Fixed Parameters**:
  - Batch size: 1024
  - Sequence length: 10000
  - Number of heads: 16
  - Dimension per head: 512
  - Hidden size of MLP: 32768

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total devices**: 16 GPUs (fully utilized)
- **Strategy**: Widely adopted large-scale model deployment

## Experimental Methods

### Tested Configurations
1. **Baseline**: TP=8 + PP=2 on 16 GPUs
2. **Proposed**: Two-level partitioning with m×n=16 on 16 GPUs
   - m: intra-head dimension splits
   - n: head-level splits  
   - Result: 16 partitions (m×n=16)

## Performance Metrics

### Throughput (TPS)
- **Definition**: Tokens processed per second
- **Baseline**: 1,200,000 tokens/sec
- **Proposed**: 1,580,000 tokens/sec
- **Improvement**: 31.7% increase ((1.58M-1.2M)/1.2M × 100)

### Time Per Output Token (TPOT)
- **Definition**: Average synchronization and communication overhead per token (milliseconds)
- **Baseline**: 0.35 ms
- **Proposed**: 0.22 ms
- **Improvement**: 37.1% reduction ((0.35-0.22)/0.35 × 100)

## Results Summary

| Model Type    | Method                | TPS (tokens/sec) | TPOT (ms) |
| ------------- | --------------------- | ---------------- | --------- |
| 2-layer Dense | Baseline (TP=8, PP=2) | 1,200,000        | 0.35      |
| 2-layer Dense | Proposed (m×n=16)     | 1,580,000        | 0.22      |

## Analysis

### Performance Gains
1. **Throughput Improvement**: 31.7% increase achieved by
   - Better load balancing across devices
   - Reduced communication overhead
   - Improved hardware utilization

2. **Communication Reduction**: 37.1% TPOT reduction indicates
   - More efficient communication patterns
   - Localized computation within partitions
   - Reduced cross-device synchronization

### Scalability Achievement
- **Full utilization**: 16/16 devices actively processing
- **Linear scaling**: Each partition processes 1/16 of computation
- **Memory efficiency**: Each device handles 1/16 of parameters

### Model Architecture Impact
- **Dense model**: Benefits significantly from two-level partitioning
- **Attention optimization**: Primary gains from MHA layer optimization
- **Batch efficiency**: Large batch (1024) ensures GPU saturation

## Limitations and Considerations

### Hardware Dependencies
- **GPU count**: Method scales with m×n ≤ available devices
- **Network bandwidth**: Intra-group communication requires sufficient bandwidth
- **Memory constraints**: Each partition must fit within device memory

### Model Characteristics
- **Head count**: Benefits models with sufficient heads (h ≥ n)
- **Head dimension**: Benefits models with large head dimensions (d ≥ m)
- **Architecture**: Specifically targets MHA layers in transformer models

## Reproducibility

### Experimental Controls
- **Fixed parameters**: All dimensions and batch sizes held constant
- **Precision**: FP16 throughout for fair comparison
- **Measurement**: Multiple runs averaged for stable metrics

### Deployment Configuration
- **Device mapping**: m×n partitions → 16 GPU devices
- **Communication**: Optimized placement for minimal inter-device traffic
- **Load balancing**: Equal partition sizes ensure uniform work distribution