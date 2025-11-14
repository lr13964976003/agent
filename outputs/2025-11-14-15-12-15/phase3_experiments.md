# Phase 3: Experiments Extraction (CORRECTED)

## Experimental Setup for 4-Layer Dense Model

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory per GPU**: ~80GB HBM3, ~50MB L2 cache
- **Interconnect**: NVLink/PCIe for inter-GPU communication

### Model Specifications
- **Architecture**: 4-layer fully connected dense network
- **Total Parameters**: 30B
- **Precision**: BF16 (2 bytes per parameter)
- **Layer Distribution**: 7.5B parameters per layer
- **Memory per Layer**: ~25.7GB (15GB weights + 10.48GB activations + buffers)

### Hyperparameters
- **Batch Size**: 128
- **Sequence Length**: 10,000 tokens
- **Number of Heads**: 32
- **Head Dimension**: 128
- **Hidden Size**: 4,096 (32 × 128)
- **MLP Hidden Size**: 16,384

### Baseline Comparison
- **Baseline Configuration**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
  - Total GPUs: 8 × 2 = 16 GPUs
  - TP splits layers across 8 devices
  - PP splits model across 2 pipeline stages

### Corrected Results Table

| Model               | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
|---------------------|-----------------------|------|----------------|-----------|
| Dense (4-layer)     | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078     |
| Dense (4-layer)     | Proposed Layer-wise   | 16   | 15,360         | 0.065     |

### Performance Analysis
- **Throughput Improvement**: 20% increase (15,360 vs 12,800 TPS)
- **Latency Reduction**: 17% reduction (0.065ms vs 0.078ms TPOT)
- **Efficiency Gain**: From better on-chip memory utilization

### Key Findings
1. **Memory Efficiency**: Layer-wise approach reduces off-chip memory accesses
2. **Scalability**: Method works effectively for 4-layer model on 16 GPUs
3. **Trade-offs**: More balanced load distribution compared to TP+PP baseline
4. **Reproducibility**: Results consistent with theoretical memory calculations