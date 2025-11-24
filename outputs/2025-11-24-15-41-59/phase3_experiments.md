# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **GPU setup**: 16×H100 GPUs with NVLink and NVSwitch interconnects
- **Precision**: BF16 (BFloat16)
- **Mode**: Inference-only setting

### Model Architecture
- **Model**: Dense Transformer
- **Layers**: 16 transformer layers
- **Architecture**: Standard feed-forward transformer architecture

### Fixed Parameters
- **Batch size**: 128 (fixed across all experiments)
- **Sequence length**: 100,000 tokens (fixed)
- **Attention heads**: 32 heads
- **Head dimension**: 128 per head
- **MLP hidden size**: 16,384

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **No sequence parallelism** in baseline
- **No ring-based attention communication** in baseline

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Higher values indicate better performance
   - Unit: tokens/second

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Lower values indicate better performance
   - Unit: milliseconds (ms)

## Results Analysis

### Dense Transformer Results

| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

### Performance Improvements

#### Dense Model Analysis
- **TPS improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **Latency reduction**: 17.6% decrease (0.85ms → 0.70ms)
- **Throughput gain**: Substantial improvement in token processing rate

## Performance Analysis

### Root Cause Analysis
The improvements are attributed to:
1. **Ring-based communication**: Avoids peak bandwidth demands of all-to-all exchanges
2. **Memory savings**: Sequence parallelism reduces activation footprint
3. **Kernel efficiency**: Improved kernel scheduling due to reduced memory pressure

### Scalability Observations
- **Benefits scale with**: Sequence length L and number of devices P
- **Optimal scenarios**: Particularly effective for L > 16k tokens
- **Communication efficiency**: Lower peak bandwidth requirements enable better scaling

### Comparative Analysis
- **RA+SP vs TP+PP**: Consistently outperforms tensor+pipeline parallelism
- **Memory reduction**: Factor of P reduction in activation memory
- **Communication pattern**: Ring topology more efficient than all-to-all exchanges

## Experimental Validation

### Consistency
- **Cross-architecture benefits**: Demonstrated on dense transformer
- **Metric consistency**: Both TPS and TPOT show improvements
- **Statistical significance**: Clear performance gaps between methods

### Reproducibility
- **Fixed parameters**: All experimental parameters specified precisely
- **Hardware specification**: Exact GPU model and interconnect details provided
- **Precision**: BF16 used consistently across experiments