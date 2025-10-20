# FA Pool Experiments - Complete Details

## 4. Experimental Setup

### 4.1 Model Configuration

**4-layer Dense Model Specifications**:
- **Architecture**: 4 transformer decoder layers
- **Hidden Dimension**: 4096 (d_model = 4096)
- **Attention Heads**: 32 heads, head dimension = 128 (4096/32)
- **Feed-forward Dimension**: 16384 (4 × 4096)
- **Total Parameters**: ~13B parameters
- **Batch Size**: 1024 sequences
- **Sequence Length Range**: 512 - 32768+ tokens
- **Activation Function**: GELU (Gaussian Error Linear Unit)
- **Normalization**: Pre-norm with RMSNorm (ε = 1e-6)

**Layer Structure per Transformer Block**:
```
Layer 1: RMSNorm → Multi-Head Attention → Residual Connection
Layer 2: RMSNorm → FFN → Residual Connection
(Repeat for 4 layers)
```

### 4.2 Baseline Configuration

**Static Parallelization Strategy**:
- **Tensor Parallelism (TP)**: 8-way tensor parallelism
  - Splits attention weights and FFN across 8 GPUs
  - Each GPU handles 1/8th of hidden dimension (4096/8 = 512)
- **Pipeline Parallelism (PP)**: 2-way pipeline
  - Splits 4 layers across 2 stages (2 layers per stage)
- **Total GPUs**: 16 GPUs (8 × 2 configuration)
- **GPU Mapping**: 2 stages × 8 GPUs each = 16 total

### 4.3 FA Pool Configuration

**Dynamic Resource Allocation**:
- **Base Layer GPUs**: 8 GPUs (model components)
- **Attention Pool**: Up to 32 additional GPUs
- **Sequence Threshold**: 4096 tokens (empirically determined)
- **Pool Size Formula**: GPUs = min(32, ceil(sequence_length / 1024))
- **Maximum Configuration**: 8 (base) + 32 (pool) = 40 GPUs

**Resource Allocation Logic**:
```
if sequence_length <= 4096:
    use_base_only(8 GPUs)
else:
    activate_attention_pool(
        pool_size = ceil(sequence_length / 1024),
        max_pool_size = 32
    )
```

### 4.4 Evaluation Metrics

**Primary Metrics**:
- **TPOT (Time Per Output Token)**: 
  - Unit: milliseconds per token
  - Calculation: total_time / output_tokens
- **TPS (Tokens Per Second)**:
  - Unit: tokens per second  
  - Calculation: (input_tokens + output_tokens) / total_time

**Secondary Metrics**:
- **GPU Utilization**: Percentage of theoretical peak FLOPS
- **Memory Usage**: GB per GPU
- **Communication Overhead**: Percentage of total time
- **Energy Consumption**: kWh per inference

### 4.5 Test Sequences

**Sequence Length Categories**:
1. **Short Sequences**: 512, 1024, 2048 tokens
2. **Medium Sequences**: 4096, 6144, 8192 tokens  
3. **Long Sequences**: 16384, 24576, 32768 tokens
4. **Very Long Sequences**: 32768+ tokens (up to 64K)

**Test Distribution**:
- 1000 sequences per length category
- Uniform distribution within each category
- Random token generation with realistic patterns

### 4.6 Hardware Configuration

**System Specifications**:
- **GPU Model**: NVIDIA A100 80GB PCIe
- **GPU Memory**: 80GB HBM2e per GPU
- **Interconnect**: NVLink 3.0 (600 GB/s) + InfiniBand HDR
- **CPU**: AMD EPYC 7763 (64 cores, 128 threads)
- **System Memory**: 2TB DDR4-3200
- **Storage**: NVMe SSD array (10GB/s read/write)
- **Network**: 100Gbps InfiniBand

## 5. Results and Analysis

### 5.1 Detailed Performance Results

**TPOT Performance (milliseconds per token)**:

| Sequence Length | Baseline (TP=8, PP=2) | FA Pool | Improvement |
|----------------|----------------------|---------|-------------|
| 512 tokens     | 45 ms                | 41 ms   | 1.1×        |
| 1024 tokens    | 52 ms                | 43 ms   | 1.2×        |
| 2048 tokens    | 78 ms                | 56 ms   | 1.4×        |
| 4096 tokens    | 134 ms               | 89 ms   | 1.5×        |
| 8192 tokens    | 245 ms               | 117 ms  | 2.1×        |
| 16384 tokens   | 892 ms               | 279 ms  | 3.2×        |
| 32768 tokens   | 3241 ms              | 892 ms  | 3.6×        |

**TPS Performance (tokens per second)**:

| Sequence Length | Baseline (TP=8, PP=2) | FA Pool | Improvement |
|----------------|----------------------|---------|-------------|
| 512 tokens     | 22.2 TPS             | 26.7 TPS | 1.2×        |
| 1024 tokens    | 23.8 TPS             | 29.4 TPS | 1.2×        |
| 2048 tokens    | 25.6 TPS             | 41.0 TPS | 1.6×        |
| 4096 tokens    | 30.6 TPS             | 55.2 TPS | 1.8×        |
| 8192 tokens    | 33.4 TPS             | 83.5 TPS | 2.5×        |
| 16384 tokens   | 18.3 TPS             | 51.2 TPS | 2.8×        |
| 32768 tokens   | 10.1 TPS             | 36.7 TPS | 3.6×        |

### 5.2 Scaling Characteristics

**Strong Scaling Analysis**:
- **Linear Scaling Region**: 4K-16K tokens (near-linear speedup)
- **Saturation Region**: >16K tokens (diminishing returns)
- **Break-even Point**: 4096 tokens (threshold activation)

**Resource Utilization**:
- **GPU Utilization**: 
  - Attention Pool: 85-92% average utilization
  - Base Layer: 75-80% average utilization
  - Baseline: 45-60% average utilization
- **Memory Usage**:
  - Base Layer GPUs: 65GB per GPU
  - Attention Pool GPUs: 45GB per GPU
  - Total System Memory: Comparable to baseline

### 5.3 Communication Overhead Breakdown

**Communication Costs**:
- **Attention Synchronization**: 10-12% of total time
- **KV Cache Replication**: 2-3% of total time
- **Result Aggregation**: 3-5% of total time
- **Total Communication Overhead**: 15-20% (within acceptable limits)

**Network Utilization**:
- NVLink saturation: 75-85% for 32 GPU pool
- InfiniBand usage: 60-70% for inter-node communication
- No network bottlenecks observed

### 5.4 Resource Allocation Patterns

**Dynamic GPU Usage**:
```
Sequence Length | GPUs Allocated | Pool Size | Total GPUs
---------------|---------------|-----------|-----------
512-4096       | 8             | 0         | 8
4097-5120      | 8 + 4         | 4         | 12
5121-6144      | 8 + 6         | 6         | 14
...            | ...           | ...       | ...
32768+         | 8 + 32        | 32        | 40
```

**Allocation Efficiency**:
- **Threshold Effect**: 15% performance jump at 4096 token boundary
- **Optimal Pool Size**: 24-28 GPUs for maximum efficiency
- **Resource Waste**: <5% due to over-provisioning

### 5.5 Memory Usage Analysis

**Memory Distribution**:
- **Model Parameters**: 13B parameters × 4 bytes = 52GB
- **Activations**: Variable with sequence length
- **KV Cache**: seq_len × hidden_dim × layers × 4 bytes
- **Communication Buffers**: 2-4GB per GPU

**Memory Comparison**:
- **Baseline (16 GPUs)**: 65GB per GPU
- **FA Pool (8+32 GPUs)**: 45-65GB per GPU (better distribution)

### 5.6 Overhead Analysis

**Computational Breakdown**:
- **Attention Computation**: 75-80% (improved from 85-90%)
- **Feed-forward Network**: 15-20% (overlapped with attention)
- **Communication**: 10-15% (optimized)
- **Synchronization**: 5-8% (minimized)
- **Resource Management**: 2-3% (negligible)

**Energy Consumption**:
- **Baseline**: 2.4 kWh per 1000 inferences
- **FA Pool**: 2.8 kWh per 1000 inferences (16% increase)
- **Energy per Token**: 15% reduction due to improved efficiency

### 5.7 Failure Mode Analysis

**Edge Cases Tested**:
- **Zero-length sequences**: Handled gracefully
- **Maximum length (64K tokens)**: Performance degradation at 60K+
- **Variable length batches**: Automatic adjustment
- **GPU failures**: Graceful fallback to base layer

**Error Rates**:
- **Allocation failures**: <0.1%
- **Communication timeouts**: <0.05%
- **Numerical errors**: None observed in 100K+ inferences