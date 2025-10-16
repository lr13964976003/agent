# Phase 2: Detailed Methodology Extraction - FA Pool

## System Architecture

### Base Layer Configuration
- **GPUs**: 8 GPUs dedicated to base layer operations
- **Components**:
  - Embedding layer
  - Positional encoding
  - Output projection layer
  - Feed-forward networks (FFN) for all 4 layers
  - Layer normalization (RMSNorm)
  - Residual connections

### Attention Pool Configuration
- **Maximum GPUs**: 32 additional GPUs
- **Activation Threshold**: Sequence length ≥ 4096 tokens
- **Block Size Calculation**: `b = ceil(n / p)` where n=sequence length, p=pool GPUs
- **KV Cache**: Replicated across pool GPUs to avoid communication

### Model Architecture Details
- **Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32 (128 dimensions per head)
- **Feed-forward Dimension**: 16384 (4× hidden dimension)
- **Activation Function**: GELU
- **Normalization**: Pre-norm with RMSNorm
- **Batch Size**: 1024
- **Total Parameters**: ~13B

## Dynamic Resource Allocation Strategy

### 1. Sequence Length Monitoring
- Real-time monitoring during inference
- Batch-level sequence length detection
- Maximum sequence length in batch determines allocation

### 2. Threshold Detection
```python
threshold = 4096  # tokens
if max_sequence_length >= threshold:
    activate_attention_pool()
```

### 3. GPU Allocation Algorithm
```python
def allocate_gpus(sequence_length):
    base_gpus = 8  # Fixed
    if sequence_length < 4096:
        pool_gpus = 0
    elif sequence_length < 8192:
        pool_gpus = 8
    elif sequence_length < 16384:
        pool_gpus = 16
    elif sequence_length < 32768:
        pool_gpus = 24
    else:
        pool_gpus = 32
    return base_gpus, pool_gpus
```

### 4. Attention Parallelization Algorithm
```
Input: Query Q (batch, seq_len, hidden), Key K, Value V, pool_gpus p
Output: Attention output O

1. Determine block size: b = ceil(seq_len / p)
2. For each GPU i in 0..p-1:
   - Q_i = Q[:, i*b:(i+1)*b, :]  # (batch, b, hidden)
   - K_i = K  # Full K copied to all GPUs
   - V_i = V  # Full V copied to all GPUs
   - O_i = FlashAttention(Q_i, K_i, V_i)  # (batch, b, hidden)
3. Concatenate results: O = concat(O_0, O_1, ..., O_p-1)
4. Return O
```

## Communication Patterns

### Data Distribution
- **Q**: Split along sequence dimension
- **K, V**: Replicated across all pool GPUs
- **Attention Output**: Gather operation to concatenate results

### Synchronization Points
1. **Pre-computation**: Ensure all pool GPUs have received K/V data
2. **Post-computation**: Synchronize before concatenation
3. **Result transfer**: Send attention output back to base layer

### Communication Optimization
- **Asynchronous execution**: FFN computation on base layer overlaps with attention in pool
- **Hierarchical reduction**: Tree-based gathering (log(p) steps vs p-1)
- **Memory layout**: Contiguous memory blocks for efficient transfers

## Threshold Determination Method

### Empirical Formula
```
threshold = argmin_t (Attention_Time(t) > FFN_Time + Communication_Overhead)
```

### Measured Values
- **FFN Time**: ~35ms for 4096 tokens
- **Attention Time**: 45ms for 4096 tokens (baseline)
- **Communication Overhead**: ~8ms for 8 pool GPUs
- **Selected Threshold**: 4096 tokens (where attention becomes bottleneck)

## Memory Layout

### Base Layer Memory (per GPU)
- **Model Parameters**: ~1.6GB per GPU (13B/8)
- **Activations**: ~32GB
- **KV Cache**: ~20GB (for 4096 tokens)
- **Total**: ~65GB per GPU

### Attention Pool Memory (per GPU)
- **Model Parameters**: 0GB (parameters remain on base layer)
- **Activations**: ~25GB
- **KV Cache**: ~15GB (replicated K/V)
- **Attention Blocks**: ~5GB
- **Total**: ~45GB per GPU

## Performance Model

### Time Complexity
```
Baseline: O(n²) attention + O(n) feedforward
FA Pool: O(n²/p) attention + O(n) feedforward + O(log p) communication
```

### Resource Utilization
- **Base GPUs**: 85-90% (FFN computations)
- **Pool GPUs**: 85-92% (attention computations)
- **Overall**: Significantly higher than static allocation

### Scaling Characteristics
- **Linear scaling**: Up to 16K tokens with 24 GPUs
- **Plateau**: Performance gains diminish beyond 24 GPUs
- **Optimal efficiency**: 16-24 pool GPUs for most sequences