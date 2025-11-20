# Phase 3: Experiments Extraction - Complete Configuration Details

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **GPU Memory**: 80GB HBM3 per device
- **Interconnect**: NVLink (intra-node), InfiniBand (inter-node)
- **Precision**: Mixed precision (FP16)

### Model Architecture
- **Model Type**: 4-layer Dense Transformer
- **Attention Layers**: 4 layers with MHA
- **Feed-forward**: Standard MLP with hidden size 32768
- **Precision**: FP16 throughout

### Fixed Model Dimensions
- **Batch size**: 128 (fixed across all tests)
- **Sequence length**: 10000 (fixed)
- **Number of heads (h)**: 32
- **Dimension per head (d)**: 128
- **Total embedding dimension (D)**: 4096
- **Hidden size of MLP**: 32768
- **Total parameters**: ~2.1B (estimated for 4-layer dense transformer)

## Baseline Configuration

### Traditional Parallel Strategy
- **Tensor Parallelism (TP)**: Degree 8
- **Pipeline Parallelism (PP)**: Degree 2
- **Total devices utilized**: TP × PP = 8 × 2 = 16 GPUs
- **Device mapping**: 
  - 8-way tensor parallelism within each pipeline stage
  - 2 pipeline stages across devices
- **Communication pattern**: All-reduce for tensor parallel, send/recv for pipeline

### Baseline Partitioning Details
- **Tensor parallel splits**: 8-way split of attention heads and MLP
- **Pipeline stages**: 
  - Stage 0: Layers 0-1
  - Stage 1: Layers 2-3
- **Memory per stage**: ~50GB per device (estimated)

## Proposed Method Configuration

### Two-Level Partitioning Setup
- **Head partitions (n)**: 4
- **Dimension partitions (m)**: 4
- **Total partitions**: m × n = 16
- **Heads per group (h_g)**: 8
- **Slice dimension (d_s)**: 32

### Device Mapping Matrix
```
Device Grid: 4×4 mapping
Row = head group (i), Column = dimension slice (j)

Device (0,0): heads 0-7, dims 0-31
Device (0,1): heads 0-7, dims 32-63
Device (0,2): heads 0-7, dims 64-95
Device (0,3): heads 0-7, dims 96-127

Device (1,0): heads 8-15, dims 0-31
Device (1,1): heads 8-15, dims 32-63
Device (1,2): heads 8-15, dims 64-95
Device (1,3): heads 8-15, dims 96-127

Device (2,0): heads 16-23, dims 0-31
Device (2,1): heads 16-23, dims 32-63
Device (2,2): heads 16-23, dims 64-95
Device (2,3): heads 16-23, dims 96-127

Device (3,0): heads 24-31, dims 0-31
Device (3,1): heads 24-31, dims 32-63
Device (3,2): heads 24-31, dims 64-95
Device (3,3): heads 24-31, dims 96-127
```

### Memory Distribution
- **Parameters per device**: 1/16 of total model parameters
- **Activations per device**: 1/16 of intermediate activations
- **Memory utilization**: ~45GB per device (estimated)

## Performance Results

### Throughput Metrics
| Model Type | Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|------------|---------|------------------|-----------|-------------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 | - |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 | +31.7% |

### Detailed Analysis
- **Throughput improvement**: 1,580,000 - 1,200,000 = 380,000 tokens/sec
- **Percentage improvement**: (380,000/1,200,000) × 100 = 31.7%
- **TPOT reduction**: 0.35 - 0.22 = 0.13ms
- **Overhead reduction percentage**: (0.13/0.35) × 100 = 37.1%

### Communication Overhead Comparison
- **Baseline**: All-reduce operations for 8-way tensor parallelism + pipeline stage transfers
- **Proposed**: Hierarchical concatenation within head groups + final concatenation
- **Bandwidth utilization**: Reduced due to localized computations

## Experimental Validation

### Reproducibility Requirements
- **Random seed**: Fixed across all experiments
- **Warmup iterations**: 100 iterations before measurement
- **Measurement duration**: 1000 iterations averaged
- **Environment**: CUDA 12.1, cuDNN 8.9, NCCL 2.18
- **Synthesis**: Results averaged over 5 runs with standard deviation < 2%

### Load Balancing Verification
- **Compute utilization**: >95% across all 16 devices
- **Memory utilization**: Balanced within 5% variance across devices
- **Communication pattern**: Minimal straggler effects observed