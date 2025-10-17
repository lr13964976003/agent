# FA Pool: A Dynamic Parallel Strategy for Scaling Attention Mechanisms in Large Language Models

## Abstract

The computational complexity of attention mechanisms in transformer-based models grows quadratically with sequence length, creating a significant bottleneck for processing long sequences. We propose FA Pool (Flash Attention Pool), a novel dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds. When input sequences exceed a predetermined length, FA Pool activates additional GPU resources to form a computation pool dedicated to parallel attention calculations, thereby reducing the computational burden on individual GPUs. Our approach combines the benefits of Flash Attention's memory-efficient algorithms with dynamic resource allocation to achieve superior scaling characteristics. Experimental results on a 4-layer Dense model demonstrate that FA Pool achieves significant improvements in both Time Per Output Token (TPOT) and Tokens Per Second (TPS) metrics compared to traditional static parallelization strategies (TP=8, PP=2 baseline). The strategy shows particular effectiveness for long sequence processing, achieving up to 3.2x improvement in TPOT and 2.8x improvement in TPS for sequences exceeding 8K tokens.

## 1. Introduction

Large language models (LLMs) face significant challenges due to the O(n²) complexity of attention mechanisms in transformer architectures. Traditional static parallelization strategies (Tensor Parallelism and Pipeline Parallelism) lead to suboptimal resource utilization, particularly with variable sequence lengths. This rigidity causes either resource underutilization for short sequences or computational bottlenecks for long sequences.

We introduce FA Pool, a dynamic parallel strategy that addresses these limitations through:

1. **Adaptive Resource Allocation**: Dynamically adjusting GPU resources based on sequence length thresholds
2. **Parallel Attention Computation**: Distributing attention calculations across a pool of GPUs when sequences exceed critical lengths
3. **Maintaining Model Coherence**: Preserving feed-forward network integrity while parallelizing attention mechanisms
4. **Optimizing Communication Overhead**: Minimizing inter-GPU communication through intelligent workload distribution

## 2. FA Pool Methodology

### 2.1 System Architecture

**Base Layer**: 8 GPUs containing core components:
- Embedding layer
- Positional encoding
- Output layer
- Feed-forward network (FFN) layer

**Attention Pool**: Dynamically allocated up to 32 GPUs for attention computation

**Resource Manager**: Monitors sequence length threshold (4096 tokens) and manages GPU allocation/deallocation

### 2.2 Dynamic Resource Allocation Strategy

1. **Sequence Length Monitoring**: Continuous input monitoring
2. **Threshold Detection**: Compare against 4096 token threshold
3. **Resource Activation**: When exceeded, activate additional GPUs for attention pool
4. **Workload Distribution**: Partition attention computation across pool GPUs
5. **Result Aggregation**: Synchronize and collect results
6. **Resource Deactivation**: Release resources when below threshold

### 2.3 Attention Parallelization

Block-wise parallelization within attention pool:

```
Block size calculation: b = ceil(n / p)
For each GPU i in pool:
   Q_i = Q[i*b:(i+1)*b], K_i = K[i*b:(i+1)*b], V_i = V[i*b:(i+1)*b]
   O_i = FlashAttention(Q_i, K, V)
O = concat(O_0, O_1, ..., O_p-1)
```

### 2.4 Communication Optimization

- **KV Cache Sharing**: Replicate keys/values across pool GPUs
- **Asynchronous Execution**: Overlap attention computation with FFN operations
- **Hierarchical Reduction**: Tree-based reduction pattern for result aggregation

### 2.5 Model Configuration

- **Layers**: 4 transformer layers (each with multi-head attention + FFN)
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Batch Size**: 1024
- **Model Parameters**: ~13B parameters
- **Activation**: GELU, Pre-norm with RMSNorm

## 3. Experimental Setup and Results

### 3.1 Baseline Configuration
- **Tensor Parallelism**: 8-way
- **Pipeline Parallelism**: 2-way
- **Total GPUs**: 16 (baseline)

### 3.2 FA Pool Configuration
- **Base Layer**: 8 GPUs
- **Attention Pool**: Up to 32 additional GPUs
- **Sequence Threshold**: 4096 tokens

### 3.3 Performance Results

**TPOT Improvements**:
- 512 tokens: 45ms → 41ms (1.1x)
- 2048 tokens: 78ms → 56ms (1.4x)
- 8192 tokens: 245ms → 117ms (2.1x)
- 16384 tokens: 892ms → 279ms (3.2x)

**TPS Improvements**:
- 512 tokens: 22.2 → 26.7 TPS (1.2x)
- 2048 tokens: 25.6 → 41.0 TPS (1.6x)
- 8192 tokens: 33.4 → 83.5 TPS (2.5x)
- 16384 tokens: 18.3 → 51.2 TPS (2.8x)

### 3.4 System Characteristics

- **GPU Utilization**: 85-92% vs 45-60% (baseline)
- **Communication Overhead**: <15% total computation time
- **Memory Usage**: 65GB base, 45GB pool GPUs
- **Optimal Pool Size**: 24 GPUs (performance plateau)
- **Scaling**: Near-linear up to 16K tokens

## 4. Conclusion

FA Pool achieves significant performance improvements through dynamic resource allocation based on sequence length, particularly effective for long sequences. The strategy demonstrates up to 3.2x TPOT and 2.8x TPS improvements compared to static parallelization strategies while maintaining efficient resource utilization and model coherence.