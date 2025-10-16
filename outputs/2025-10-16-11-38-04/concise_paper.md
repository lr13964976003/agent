# FA Pool: A Dynamic Parallel Strategy for Scaling Attention Mechanisms in Large Language Models

## Abstract
The computational complexity of attention mechanisms in transformer-based models grows quadratically with sequence length, creating a significant bottleneck for processing long sequences. We propose FA Pool (Flash Attention Pool), a novel dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds. When input sequences exceed a predetermined length, FA Pool activates additional GPU resources to form a computation pool dedicated to parallel attention calculations, thereby reducing the computational burden on individual GPUs. Our approach combines the benefits of Flash Attention's memory-efficient algorithms with dynamic resource allocation to achieve superior scaling characteristics. Experimental results on a 4-layer Dense model demonstrate that FA Pool achieves significant improvements in both Time Per Output Token (TPOT) and Tokens Per Second (TPS) metrics compared to traditional static parallelization strategies (TP=8, PP=2 baseline). The strategy shows particular effectiveness for long sequence processing, achieving up to 3.2x improvement in TPOT and 2.8x improvement in TPS for sequences exceeding 8K tokens.

## 1. Methodology

### 1.1 Model Architecture
- **Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Model Parameters**: ~13B
- **Batch Size**: 1024
- **Activation Function**: GELU
- **Normalization**: Pre-norm RMSNorm

### 1.2 System Architecture
- **Base Layer**: 8 GPUs maintaining core components
  - Embedding layer
  - Positional encoding
  - Output projection
  - Feed-forward networks (all 4 layers)
- **Attention Pool**: Up to 32 additional GPUs
  - Dedicated to attention computation
  - Activated when sequence length ≥ 4096 tokens
- **Resource Manager**: Monitors sequence length and allocates GPUs

### 1.3 Dynamic Resource Allocation
```python
pool_gpus = 0 if seq_len < 4096 else min(ceil(seq_len / 512), 32)
```

### 1.4 Attention Parallelization
- **Block Size**: `b = ceil(sequence_length / num_pool_gpus)`
- **Partitioning**: Query split along sequence dimension, K/V replicated
- **Algorithm**: FlashAttention per block with KV cache sharing
- **Result Aggregation**: Concatenation across pool GPUs

### 1.5 Communication Optimization
- **KV Cache Sharing**: Replicated across pool GPUs
- **Asynchronous Execution**: Overlaps attention with FFN computation
- **Hierarchical Reduction**: Tree-based result gathering

## 2. Experimental Setup

### 2.1 Baseline Configuration
- **Tensor Parallelism**: 8-way (8 GPUs)
- **Pipeline Parallelism**: 2-way (2 stages)
- **Total GPUs**: 16 GPUs
- **GPU Model**: NVIDIA A100 80GB

### 2.2 FA Pool Configuration
- **Base GPUs**: 8 (maintaining tensor parallelism)
- **Pool GPUs**: 0-32 (dynamic based on sequence length)
- **Maximum Total**: 40 GPUs
- **Sequence Threshold**: 4096 tokens

### 2.3 Evaluation Metrics
- **Time Per Output Token (TPOT)**: milliseconds per token
- **Tokens Per Second (TPS)**: tokens processed per second

## 3. Results

### 3.1 Performance Improvements
| Sequence Length | TPOT Improvement | TPS Improvement | Pool GPUs |
|----------------|------------------|-----------------|-----------|
| 512 tokens     | 1.1×             | 1.2×            | 0         |
| 2048 tokens    | 1.4×             | 1.6×            | 8         |
| 8192 tokens    | 2.1×             | 2.5×            | 16        |
| 16384 tokens   | 3.2×             | 2.8×            | 24        |

### 3.2 Resource Utilization
- **Base GPUs**: 85-90% utilization for FFN
- **Pool GPUs**: 85-92% utilization for attention
- **Communication Overhead**: 10-15% of total time

### 3.3 Memory Usage
- **Base Layer**: 65GB per GPU
- **Attention Pool**: 45GB per GPU
- **Total Memory**: Optimized distribution across 8-40 GPUs

### 3.4 Scaling Characteristics
- **Linear scaling**: Up to 16K tokens with 24 GPUs
- **Efficiency**: 78% at 24 pool GPUs
- **Optimal range**: 16-24 GPUs for most long sequences

## 4. Conclusion
FA Pool demonstrates that dynamic resource allocation based on sequence length can significantly improve performance for long sequences in transformer models. The strategy achieves up to 3.2× improvement in TPOT and 2.8× improvement in TPS for sequences exceeding 8K tokens, while maintaining efficient resource utilization and memory distribution across 8-40 GPUs.