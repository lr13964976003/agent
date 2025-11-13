# Phase 3: Experimental Details Extraction

## Experimental Setup

### Hardware Configuration
- **GPUs**: 16× NVIDIA H100 GPUs
- **Interconnect**: NVLink + NVSwitch
- **Total devices**: P = 16
- **Setting**: Inference-only evaluation

### Model Architecture
- **Type**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer

### Fixed Parameters
- **Precision**: FP16
- **Batch size**: B = 1024
- **Sequence length**: L = 10000 tokens
- **Attention heads**: H = 16 heads
- **Head dimension**: d_h = 512
- **Model dimension**: d_model = H × d_h = 16 × 512 = 8192
- **MLP hidden size**: 32768

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token in milliseconds
   - Lower values indicate better performance

## Experimental Results

### Dense Transformer Results
| Method | Configuration | TPS (tokens/s) | TPOT (ms) | Improvement |
|--------|---------------|----------------|-----------|-------------|
| Baseline | TP=8, PP=2 | 1.20M | 0.85 | - |
| RA+SP | Ring Attention + Sequence Parallelism | **1.45M** | **0.70** | +20.8% TPS, -17.6% TPOT |

### Performance Analysis

#### Throughput Improvements
- **TPS increase**: 1.45M vs 1.20M (20.8% improvement)
- **Raw improvement**: +250K tokens/second

#### Latency Reductions
- **TPOT reduction**: 0.70ms vs 0.85ms (17.6% reduction)
- **Absolute reduction**: -0.15ms per token

## Baseline Configuration Details

### Baseline Method
- **Tensor Parallelism**: TP = 8 (splits model tensors across 8 GPUs)
- **Pipeline Parallelism**: PP = 2 (splits layers across 2 pipeline stages)
- **Sequence Parallelism**: Not used
- **Ring Attention**: Not used

### Communication Patterns
- **Baseline**: All-to-all communication for tensor parallelism
- **Proposed**: Ring-based communication for attention computation

## Experimental Validation

### Consistency Check
- **Dense model**: Consistent improvement across measurements
- **No degradation**: No case where RA+SP performed worse

### Scalability Evidence
- **Long sequences**: Benefits increase with L > 16k
- **Device scaling**: Performance improves with more devices P

## Experimental Limitations

### Scope
- **Inference only**: No training experiments
- **Fixed batch**: All experiments use B = 1024
- **Fixed sequence**: All experiments use L = 10000
- **Single architecture**: Only dense transformer tested

### Future Extensions
- **Training scenarios**: Gradient communication overhead
- **Variable sequences**: Different sequence lengths
- **Larger models**: More layers, different architectures