# Phase 2: Methodology Extraction - FA Pool Paper

## 3. FA Pool Methodology

### 3.1 System Architecture

**Base Layer**: Primary computational layer containing:
- Embedding layer
- Positional encoding
- Output layer
- Feed-forward network (FFN) layer

**Attention Pool**: Dynamically allocated set of GPUs dedicated to attention computation

**FFN Layer**: Feed-forward network computations that remain on the base layer

**Resource Manager**: Monitors sequence length and allocates/deallocates GPU resources for attention pool

### 3.2 Dynamic Resource Allocation Strategy

The FA Pool strategy operates through the following mechanism:

1. **Sequence Length Monitoring**: Continuously monitor input sequence length during inference
2. **Threshold Detection**: Compare sequence length against predefined thresholds (4096 tokens)
3. **Resource Activation**: When sequence length exceeds threshold, activate additional GPUs for attention pool. Otherwise, activate only Attention with same number of FFNs as before.
4. **Workload Distribution**: Partition attention computation across available pool GPUs
5. **Result Aggregation**: Collect and synchronize results from pool GPUs
6. **Resource Deactivation**: Release pool resources when sequence length drops below threshold

### 3.3 Attention Parallelization

Within the attention pool, implement block-wise parallelization strategy:

```
Input: Query Q, Key K, Value V, sequence length n, number of pool GPUs p
Output: Attention output O

1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool:
   - Extract block: Q_i = Q[i*b:(i+1)*b], K_i = K[i*b:(i+1)*b], V_i = V[i*b:(i+1)*b]
   - Compute local attention: O_i = FlashAttention(Q_i, K, V)
3. Synchronize and aggregate results: O = concat(O_0, O_1, ..., O_p-1)
4. Return final output O
```

### 3.4 Communication Optimization

**KV Cache Sharing**: Keys and values replicated across pool GPUs to avoid communication during attention computation

**Asynchronous Execution**: Attention computation overlaps with FFN operations on base layer

**Hierarchical Reduction**: Results aggregated using tree-based reduction pattern to minimize communication steps

### 3.5 Threshold Determination

**Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)**

where Overhead accounts for communication and synchronization costs.

### Model Configuration Details

**Layer Structure**: 4 transformer layers, each with:
- One multi-head attention mechanism
- One feed-forward network (FFN)

**Dimensions**:
- Hidden dimension: 4096
- Attention heads: 32
- Feed-forward dimension: 16384
- Batch size: 1024

**Hardware Requirements**:
- Base layer: 8 GPUs
- Attention pool: up to 32 additional GPUs
- GPU model: NVIDIA A100 80GB
- Interconnect: NVLink 3.0 and InfiniBand