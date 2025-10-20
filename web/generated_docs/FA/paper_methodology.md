# FA Pool Methodology - Technical Details

## 3. FA Pool Methodology

### 3.1 System Architecture

**Base Layer Configuration**:
- Primary computational layer with 8 GPUs
- Contains core model components:
  - Embedding layer (4096 hidden dimension)
  - Positional encoding
  - Output projection layers
  - Feed-forward network (FFN) layers
  - Layer normalization (RMSNorm)

**Attention Pool Structure**:
- Dynamically allocated GPU pool (max 32 additional GPUs)
- Dedicated to attention computation only
- Parallel attention calculation when sequence length > 4096 tokens
- Block-wise parallelization strategy

**Resource Manager Components**:
- Sequence length monitoring system
- Threshold detection (4096 token threshold)
- GPU allocation/deallocation engine
- Workload distribution scheduler
- Result aggregation coordinator

### 3.2 Dynamic Resource Allocation Strategy

**Threshold Determination Formula**:
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
```
- Empirically determined threshold: 4096 tokens
- Overhead includes communication and synchronization costs

**Allocation Process**:
1. **Sequence Length Monitoring**: Continuous monitoring during inference
2. **Threshold Detection**: Compare current sequence length against 4096
3. **Resource Activation**: When length > 4096, activate additional GPUs
4. **Pool Size Calculation**: Number of GPUs = ceil(sequence_length / 1024)
5. **Resource Deactivation**: Release when sequence length drops below threshold

### 3.3 Attention Parallelization Algorithm

**Block-wise Parallelization Strategy**:
```
Input: 
- Query Q: (batch_size, seq_len, hidden_dim)
- Key K: (batch_size, seq_len, hidden_dim)  
- Value V: (batch_size, seq_len, hidden_dim)
- Sequence length: n
- Number of pool GPUs: p

Algorithm:
1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool (0 ≤ i < p):
   - Extract blocks:
     * Q_i = Q[:, i*b:(i+1)*b, :]  # (batch_size, b, hidden_dim)
     * K_i = K[:, i*b:(i+1)*b, :]  # (batch_size, b, hidden_dim)
     * V_i = V[:, i*b:(i+1)*b, :]  # (batch_size, b, hidden_dim)
   - Compute local attention using FlashAttention:
     * O_i = FlashAttention(Q_i, K, V)
3. Synchronize and aggregate:
   - O = concat(O_0, O_1, ..., O_p-1) along sequence dimension
4. Return final output: O (batch_size, seq_len, hidden_dim)
```

### 3.4 Communication Optimization

**KV Cache Sharing Strategy**:
- Keys and values are fully replicated across all pool GPUs
- Eliminates communication during attention computation
- Memory trade-off: Higher memory usage for reduced communication

**Asynchronous Execution Pattern**:
- Attention computation overlaps with FFN operations on base layer
- Pipeline scheduling:
  1. Attention pool begins computation on current layer
  2. Base layer processes FFN for previous layer simultaneously
  3. Results synchronized at layer boundaries

**Hierarchical Reduction Pattern**:
- Tree-based reduction for result aggregation
- Reduces communication steps from O(p) to O(log p)
- Implementation:
  - Binary tree reduction across pool GPUs
  - 2-step synchronization process
  - Final result broadcast to base layer

### 3.5 Model Layer Specifications

**Layer Structure Details**:
```
Each Transformer Layer:
- Multi-Head Attention:
  * Hidden dimension: 4096
  * Number of heads: 32
  * Head dimension: 4096/32 = 128
  * Attention computation: softmax(QK^T/√128)V

- Feed-Forward Network:
  * Input dimension: 4096
  * Hidden dimension: 16384 (4× hidden dimension)
  * Activation: GELU
  * Output dimension: 4096
  * Remains on base layer (8 GPUs)

- Normalization:
  * Pre-norm architecture
  * RMSNorm with epsilon=1e-6
  * Applied before attention and FFN
```

### 3.6 Memory Management

**Base Layer Memory**:
- Model parameters: ~13B parameters
- Per GPU memory: 65GB
- Includes: model weights, activations, KV cache, optimizer states

**Attention Pool Memory**:
- Per GPU memory: 45GB (reduced due to block-wise computation)
- Includes: attention weights, block activations, cached keys/values
- Dynamic allocation based on sequence length

**Communication Buffer Sizes**:
- Attention output buffer: seq_len × hidden_dim × 4 bytes (float32)
- KV cache per GPU: seq_len × hidden_dim × 4 bytes × 2 (K and V)
- Synchronization buffers: 2 × hidden_dim × 4 bytes

### 3.7 Implementation Details

**Flash Attention Integration**:
- Uses FlashAttention-2 algorithm for memory-efficient computation
- Block size: 128×128 tiles for GPU efficiency
- Memory savings: O(n²) → O(n) memory usage

**GPU Coordination**:
- CUDA streams for asynchronous execution
- NCCL for inter-GPU communication
- Custom kernels for block-wise attention computation

**Error Handling**:
- Automatic fallback to base layer if pool allocation fails
- Graceful degradation for edge cases (very short sequences)
- Timeout mechanisms for pool synchronization