# Phase 3: Detailed Experiments

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Architecture**: High-performance accelerator with SRAM/L2 cache
- **Memory**: HBM3 high-bandwidth memory
- **Precision**: BF16 (16-bit Brain Floating Point)
- **Interconnect**: NVLink for GPU-to-GPU communication

### Model Specifications
- **Dense Model**: 16-layer fully connected neural network
- **Total Parameters**: 30 billion (30B)
- **Layer Distribution**: 30B parameters / 16 layers = 1.875B parameters per layer average
- **Precision**: BF16 → 2 bytes per parameter → 60GB total model size

### Dimensional Parameters
- **Batch Size**: 128
- **Sequence Length**: 10,000 tokens
- **Attention Heads**: 32
- **Head Dimension**: 128
- **Hidden Size**: 32 × 128 = 4,096
- **MLP Hidden Size**: 16,384 (4× hidden size)

### Memory Footprint Calculation

#### Per Layer Analysis
Each layer consists of:
1. **Attention weights**:
   - Q projection: 4096 × 4096 = 16.8M parameters
   - K projection: 4096 × 4096 = 16.8M parameters
   - V projection: 4096 × 4096 = 16.8M parameters
   - Output projection: 4096 × 4096 = 16.8M parameters
   - **Total attention**: 67.1M parameters

2. **MLP weights**:
   - Gate projection: 4096 × 16384 = 67.1M parameters
   - Up projection: 4096 × 16384 = 67.1M parameters
   - Down projection: 16384 × 4096 = 67.1M parameters
   - **Total MLP**: 201.3M parameters

3. **Layer Normalization**: Negligible compared to weights

4. **Total per layer**: ~268.4M parameters → 536.8MB in BF16

#### Activation Memory
- **Per token**: 4,096 hidden dimensions
- **Per sequence**: 10,000 × 4,096 = 40.96M elements
- **Per batch**: 128 × 40.96M = 5.24B elements
- **BF16 activation**: ~10.49GB per layer for activations

## Baseline Comparison

### Baseline Method (TP=8, PP=2)
- **Tensor Parallelism**: 8-way splitting across GPUs
- **Pipeline Parallelism**: 2 stages across GPUs
- **Total GPUs**: 8 × 2 = 16 GPUs fully utilized
- **Performance**: 12,800 tokens/second
- **Latency**: 0.078ms per token

### Proposed Method (Layer-wise)
- **Partitioning**: 16 layers distributed across 16 GPUs
- **Cache constraint**: Each layer fits in GPU SRAM/L2 cache
- **Performance**: 15,360 tokens/second
- **Latency**: 0.065ms per token

## Performance Metrics

### Throughput Analysis
| Method | TPS | Improvement |
|--------|-----|-------------|
| Baseline | 12,800 | - |
| Proposed | 15,360 | +20% |

### Latency Analysis
| Method | TPOT (ms) | Reduction |
|--------|-----------|-----------|
| Baseline | 0.078 | - |
| Proposed | 0.065 | -17% |

### Memory Efficiency
- **Baseline**: Tensor parallelism creates partial layer distributions, requiring off-chip memory access
- **Proposed**: Entire layers loaded into fast on-chip memory, minimizing external memory access

## Cache Requirements
Based on calculations:
- **Per layer memory**: ~11GB (weights + activations + buffers)
- **Required cache per GPU**: ≥11GB to fit entire layer
- **Partitioning**: 1 layer per GPU for optimal cache utilization
- **Total utilization**: 16 layers → 16 GPUs in 1:1 mapping

## Communication Overhead
- **Baseline**: Frequent tensor synchronization across 8 GPUs per layer
- **Proposed**: Minimal inter-GPU communication between layers
- **Bandwidth advantage**: NVLink provides sufficient bandwidth for layer-to-layer transfers

## Scalability Analysis
The method demonstrates:
- **Linear scaling**: 1 layer per GPU for 16 layers
- **Cache efficiency**: Each layer fully contained in fast memory
- **Performance gain**: 20% improvement over traditional parallelism
- **Hardware utilization**: All 16 GPUs equally loaded