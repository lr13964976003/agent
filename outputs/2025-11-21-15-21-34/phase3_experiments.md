# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- Platform: 16 NVIDIA H100 GPUs
- Memory: Each with SRAM/L2 cache capacity C (exact value not specified)
- Interconnect: High-speed interconnect for inter-card communication

### Model Specifications
- **Dense Model**: 4-layer fully connected network
  - Total parameters: 30B
  - Precision: BF16 (2 bytes per parameter)
  - Architecture details:
    - Number of layers: 16 (as mentioned in results table)
    - Batch size: 128
    - Sequence length: 10000
    - Number of heads: 32
    - Head dimension: 128
    - MLP hidden size: 16384

### Memory Breakdown Estimation
Given 30B parameters in BF16:
- Total weight memory: 30B × 2 bytes = 60 GB
- Per layer weight memory: 60 GB / 16 layers = 3.75 GB/layer
- Activation memory per layer: ~batch_size × sequence_length × hidden_size × 2 bytes
  - = 128 × 10000 × (32×128) × 2 = 128 × 10000 × 4096 × 2 = 10.48 GB/layer
- Buffer memory: Additional workspace for operations (estimated ~10-20% of total)

### Baseline Configuration
- **Standard Approach**: Tensor Parallelism + Pipeline Parallelism
- **TP=8**: Tensor parallelism across 8 devices
- **PP=2**: Pipeline parallelism across 2 stages
- **Total GPUs**: 8 × 2 = 16 GPUs

### Proposed Configuration
- **Layer-wise partitioning**: Partition 16 layers across 16 GPUs
- Each partition: 1 layer per GPU (estimated ~3.75GB weights + 10.48GB activations = ~14.23GB total)
- Constraint: Must fit within H100 SRAM/L2 cache

## Performance Metrics

### Measured Results
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|---------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Improvements
- **TPS Improvement**: 15,360/12,800 = 1.20 (20% increase)
- **TPOT Reduction**: 0.078 - 0.065 = 0.013ms (17% reduction)
- **Efficiency Gain**: Achieved through better on-chip memory utilization

## Analysis

### Performance Factors
1. **On-chip Memory Utilization**: Proposed method ensures entire layer fits in cache
2. **Reduced Memory Access**: Minimizes off-chip DRAM accesses
3. **Communication Overhead**: Reduced compared to TP+PP baseline
4. **Parallelization Efficiency**: Better load balancing across GPUs

### Limitations
- Cache capacity C must accommodate single layer (weights + activations + buffers)
- Sequential dependency between layers requires careful pipelining
- Inter-card communication still needed between layers

## Validation Criteria
- Successfully demonstrates 20% throughput improvement
- Shows practical applicability on real hardware (H100 GPUs)
- Validates theoretical benefits of cache-aware partitioning