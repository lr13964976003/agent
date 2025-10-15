# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **Platform:** 16 NVIDIA H100 GPUs
- **Memory:** Each GPU has SRAM/L2 cache capacity (exact capacity not specified in paper)
- **Configuration:** Fully utilizes all 16 GPUs for both baseline and proposed methods

### Model Specifications
- **Model Type 1:** Dense model - 16-layer fully connected dense network
- **Model Type 2:** (Implied from paper structure but not detailed in experiments)

### Model Parameters (Fixed across experiments)
- **Precision:** FP16 (2 bytes per parameter)
- **Batch Size:** 1024
- **Sequence Length:** 10,000
- **Number of Heads:** 16
- **Head Dimension:** 512 (per head)
- **MLP Hidden Size:** 32,768

### Baseline Configuration
- **Method:** Standard tensor parallelism (TP) + pipeline parallelism (PP)
- **Configuration:** TP=8, PP=2
- **GPU Utilization:** 8 × 2 = 16 GPUs (full utilization)

### Performance Metrics
1. **Tokens Per Second (TPS):** Number of output tokens generated per second
2. **Time Per Output Token (TPOT):** Average time to produce a single output token (milliseconds)

## Experimental Results

### Dense Model (16-layer) Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|---------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Analysis

#### Throughput Improvement
- **TPS Increase:** 15,360 vs 12,800 = **20% improvement**
- **Absolute Gain:** +2,560 tokens/second

#### Latency Reduction
- **TPOT Reduction:** 0.065ms vs 0.078ms = **17% reduction**
- **Absolute Improvement:** -0.013ms per token

#### Efficiency Gains
- **Memory Access Optimization:** Reduced off-chip memory accesses through SRAM/L2 cache utilization
- **Communication Overhead Reduction:** Minimized inter-card communication delays
- **Parallel Efficiency:** Better hardware utilization through balanced layer distribution

## Experimental Context

### Baseline Method Details
- **Tensor Parallelism (TP=8):** Splits individual layers across 8 devices
- **Pipeline Parallelism (PP=2):** Creates 2 pipeline stages across 16 GPUs
- **Total Configuration:** 8-way tensor parallelism × 2-way pipeline parallelism = 16 GPUs

### Proposed Method Implementation
- **Layer-wise Partitioning:** 16 layers distributed across 16 GPUs
- **Cache Constraint:** Each partition fits within individual GPU SRAM/L2 cache
- **Memory Footprint:** Calculated using weight_size + activation_size + buffer_size formula
- **Partitioning Algorithm:** Likely greedy approach given contiguous layer assignment

### Inference Stage Focus
- **Stage:** Inference only (training extension mentioned as future work)
- **Workload:** Forward pass only, no backward pass considerations
- **Memory Requirements:** Primarily weights and activations, no optimizer states

### Reproducibility Details
- **Precision Consistency:** FP16 maintained across both methods
- **Batch Size Consistency:** 1024 samples per batch
- **Model Architecture:** Identical 16-layer dense network for fair comparison
- **Hardware Consistency:** Same 16 H100 GPUs for both configurations

## Key Findings Summary

1. **Significant Performance Gain:** 20% throughput improvement demonstrates effectiveness of cache-aware partitioning
2. **Scalability Validation:** Method works effectively across 16-GPU configuration
3. **Memory Efficiency:** SRAM/L2 cache utilization provides measurable benefits over traditional approaches
4. **Practical Applicability:** Real-world deployment benefits confirmed through concrete metrics