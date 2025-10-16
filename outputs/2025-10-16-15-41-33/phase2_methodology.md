# Phase 2: Detailed Methodology Extraction - FA Pool Paper

## 1. System Architecture Overview

### Component Structure
- **Base Layer**: Primary computational layer with 8 GPUs
  - Contains: Embedding layer, positional encoding, output layers, and FFN computations
  - Maintains model coherence during dynamic allocation
  - GPU count: Fixed at 8 GPUs

- **Attention Pool**: Dynamic GPU set (up to 32 additional GPUs)
  - Dedicated to attention computation only
  - Activated when sequence length > 4096 tokens
  - GPU count: Variable from 0 to 32 based on sequence length

- **Resource Manager**: Central controller
  - Monitors sequence length in real-time
  - Handles GPU allocation and deallocation
  - Manages workload distribution across pool GPUs

## 2. Dynamic Resource Allocation Strategy

### 2.1 Activation Process
```
Input: Current sequence length n
Output: Number of pool GPUs p to activate

1. if n <= 4096: p = 0
2. else: p = min(ceil(n/1024), 32)
3. Allocate p GPUs for attention pool
4. Initialize KV cache on all pool GPUs
5. Distribute attention workload across p GPUs
```

### 2.2 Workload Distribution Algorithm
```
Input: Query Q, Key K, Value V, sequence length n, pool GPUs p
Output: Distributed attention computation

Block Size Calculation:
b = ceil(n / p)

GPU Assignment:
For GPU i (0 <= i < p):
  - Q_i = Q[i*b : min((i+1)*b, n)]
  - K_i = K[0:n]  // Full key cache (replicated)
  - V_i = V[0:n]  // Full value cache (replicated)
  - Compute: O_i = FlashAttention(Q_i, K_i, V_i)

Result Aggregation:
O = concat(O_0, O_1, ..., O_{p-1})
```

## 3. Attention Parallelization Details

### 3.1 Mathematical Formulation
For multi-head attention with h=32 heads:
```
Head dimension: d_h = hidden_dim / h = 4096 / 32 = 128

Per-head computation:
Attention(Q_h, K_h, V_h) = softmax(Q_h K_h^T / sqrt(d_h)) V_h

Where:
- Q_h: (batch_size, seq_len, d_h)
- K_h: (batch_size, seq_len, d_h)  
- V_h: (batch_size, seq_len, d_h)
```

### 3.2 Parallel Flash Attention
```
Algorithm: ParallelFlashAttention
Input: Q, K, V, block_size b, GPU_id i, total_GPUs p
Output: O_i (partial attention output)

// Local computation on GPU i
local_seq_start = i * b
local_seq_end = min((i+1)*b, seq_len)

// Extract local query
Q_local = Q[local_seq_start:local_seq_end]

// Initialize output buffer
O_local = zeros(local_seq_end-local_seq_start, hidden_dim)

// Process attention in blocks (Flash Attention)
block_size_flash = 256  // Flash attention block size
for j in range(0, seq_len, block_size_flash):
    K_block = K[j:j+block_size_flash]
    V_block = V[j:j+block_size_flash]
    
    // Compute local attention
    scores = Q_local @ K_block^T / sqrt(128)
    weights = softmax(scores, dim=-1)
    O_local += weights @ V_block

return O_local
```

## 4. Communication Optimization

### 4.1 KV Cache Management
- **Replication Strategy**: Full K and V matrices replicated on each pool GPU
- **Memory Cost**: seq_len * hidden_dim * 2 * 4 bytes = seq_len * 32768 bytes
- **Benefit**: Eliminates communication during attention computation

### 4.2 Result Aggregation
```
Hierarchical Reduction Pattern:
Level 1: Pairwise reduction (16 pairs → 16 GPUs)
Level 2: Pairwise reduction (8 pairs → 8 GPUs)  
Level 3: Pairwise reduction (4 pairs → 4 GPUs)
Level 4: Pairwise reduction (2 pairs → 2 GPUs)
Level 5: Final reduction (1 pair → 1 GPU)

Total reduction steps: log2(p) where p is pool size
```

### 4.3 Asynchronous Execution
```
Timeline:
1. Base layer: Start FFN computation for layer n
2. Attention pool: Compute attention for layer n (overlaps with FFN)
3. Synchronize: Wait for both attention and FFN completion
4. Continue to layer n+1
```

## 5. Threshold Determination

### 5.1 Mathematical Formulation
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)

Where:
- Attention_Time(t) = α * t²  // Quadratic in sequence length
- FFN_Time = β * seq_len    // Linear in sequence length  
- Overhead = γ * log2(p)    // Communication overhead

Empirical values (from experiments):
- α = 0.000012 ms/token²
- β = 0.008 ms/token
- γ = 0.5 ms

Solving: t = 4096 tokens (threshold)
```

### 5.2 Dynamic Adjustment
```
Runtime threshold tuning:
if current_TPOT > 1.5 * baseline_TPOT:
    threshold = threshold * 0.9  // Lower threshold
elif pool_GPU_utilization < 60%:
    threshold = threshold * 1.1  // Raise threshold
```

## 6. Memory Layout and Data Flow

### 6.1 Memory Allocation per GPU
```
Base Layer GPU (8 total):
- Model parameters: ~1.6GB (13B/8)
- Activations: 65GB
- KV cache: seq_len * 4096 * 4 bytes
- Working memory: 5GB
Total: ~72GB per base GPU

Attention Pool GPU (up to 32):
- Model parameters: 0GB (no parameters stored)
- Activations: 45GB
- KV cache: seq_len * 4096 * 2 * 4 bytes (replicated)
- Working memory: 10GB
Total: ~55GB per pool GPU
```

### 6.2 Data Flow Sequence
```
1. Input processing (Base Layer):
   - Token embedding + positional encoding
   - Initial layer normalization

2. Attention computation (Attention Pool):
   - Query extraction from base layer
   - Distributed attention calculation
   - Result aggregation back to base layer

3. FFN computation (Base Layer):
   - Parallel with attention computation
   - Uses column-wise and row-wise tensor parallelism

4. Output generation (Base Layer):
   - Final layer normalization
   - Output projection and token generation
```

## 7. Model Synchronization

### 7.1 Layer-wise Synchronization
```
For each transformer layer:
1. Base layer: Prepare Q, K, V matrices
2. Attention pool: Receive Q, K, V (broadcast from base)
3. Attention pool: Compute distributed attention
4. Attention pool: Send result back to base layer
5. Base layer: Receive attention output
6. Base layer: Compute FFN (overlapping with attention)
7. Synchronize: Ensure both attention and FFN complete
```

### 7.2 Gradient Synchronization (Training Mode)
```
Note: Current implementation focuses on inference
Training extension would require:
- Gradient accumulation across pool GPUs
- Parameter server synchronization for attention weights
- Backpropagation through distributed computation
```