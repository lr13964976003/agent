# Phase 3: Experiments Extraction

## Experimental Setup
### Hardware Configuration
- **16 NVIDIA H100 GPUs** interconnected via NVLink and NVSwitch
- **Inference-only setting** (no training experiments)
- **Precision**: FP16
- **Batch size**: Fixed at 1024
- **Sequence length**: Fixed at 10000 tokens
- **Number of heads**: 16 heads
- **Dimension per head**: 512 (total hidden size = 16 × 512 = 8192)
- **MLP hidden size**: 32768

### Model Architecture
- **Dense Transformer**: 4 layers with standard feed-forward architecture
- **Baseline configuration**: Tensor Parallelism (TP) = 8, Pipeline Parallelism (PP) = 2
- **Proposed method**: Ring Attention + Sequence Parallelism (RA+SP)

## Evaluation Metrics
1. **TPS (Tokens Per Second)** - Raw throughput of tokens processed per second (higher is better)
2. **TPOT (Time Per Output Token)** - Average latency per output token, measured in milliseconds (lower is better)

## Experimental Results

### Performance Comparison Table
| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
|------------|-----------------------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85      |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**  |

### Performance Improvements
- **TPS Improvement**: 20.8% increase (1.45M vs 1.20M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.70ms vs 0.85ms)
- **Combined Benefits**: Higher throughput AND reduced latency achieved simultaneously

## Analysis of Results
### Latency Reduction Mechanisms
1. **Ring-based communication pattern** avoids peak bandwidth demands of all-to-all exchanges
2. **Memory savings** from sequence parallelism reduce activation footprint
3. **Improved kernel scheduling efficiency** due to reduced memory pressure

### Scalability Characteristics
- Performance benefits grow with sequence length L and number of devices P
- Particularly significant for L > 16k tokens (as claimed in implementation details)
- Benefits both throughput (TPS) and latency (TPOT) metrics

## Experimental Limitations
- **Inference-only evaluation** (no training experiments conducted)
- **Single model architecture tested** (4-layer dense transformer)
- **Fixed hyperparameters** (sequence length=10k, batch=1024, heads=16, etc.)
- **16 GPU configuration only** (scalability to larger clusters not demonstrated)

## Future Work Mentioned
- Extension to training scenarios with gradient communication
- Exploration of hierarchical topologies combining ring-based intra-node with bandwidth-aware inter-node scheduling
- Integration with adaptive precision and kernel fusion techniques for further performance improvements