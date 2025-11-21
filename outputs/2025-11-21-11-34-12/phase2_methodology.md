# Phase 2: Methodology Extraction

## Two-Level Partitioning Method

### 1. Multi-Head Attention Foundation

Given input tensor X ∈ ℝ^(B×L×D) where:
- B: batch size
- L: sequence length  
- D: embedding dimension

The MHA layer projects X into query, key, and value tensors:
```
Q, K, V = XW_Q, XW_K, XW_V
```
where W_Q, W_K, W_V ∈ ℝ^(D×D)

### 2. Head Dimension Partitioning

The embedding dimension D is split into h heads, each with dimension d = D/h
- Total heads: h = 32 (from experimental setup)
- Heads per group: h_g = h/n where n = 4 (for m×n=16)
- Therefore: h_g = 32/4 = 8 heads per group

### 3. Intra-Head Dimension Partitioning

Each head's feature dimension d is further sliced:
- Per-head dimension: d = 128 (from experimental setup)
- Dimension slices: d_s = d/m where m = 4 (for m×n=16)
- Therefore: d_s = 128/4 = 32 dimensions per slice

### 4. Weight Matrix Partitioning

Each projection matrix W ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice
- Block size: W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g) = ℝ^(32×8 × 32×8) = ℝ^(256×256)

### 5. Per-Partition Computation

Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)
V^(i,j) = X W_V^(i,j)

Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)
```

### 6. Softmax Computation Details

The scaled dot-product attention for each partition includes precise softmax:
```
score_matrix = Q^(i,j) (K^(i,j))^T / √d_s ∈ ℝ^(B×L×L)
attention_weights = softmax(score_matrix) ∈ ℝ^(B×L×L)
Attention^(i,j) = attention_weights V^(i,j) ∈ ℝ^(B×L×d_s·h_g)
```

### 7. Hierarchical Aggregation

**Phase 1 - Intra-group concatenation:**
- Concatenate m=4 dimension slices within each head group
- Result per group: ℝ^(B×L×d·h_g) = ℝ^(B×L×128×8)

**Phase 2 - Inter-group concatenation:**
- Concatenate n=4 head groups along head dimension
- Final output: ℝ^(B×L×D) = ℝ^(B×L×4096)

### 8. Communication Patterns

- **Input Distribution**: Broadcast X to all m×n=16 devices
- **Intra-group Communication**: All-gather within each head group (4 devices per group)
- **Output Collection**: Concatenation along pre-defined device topology
- **Communication Complexity**: O(B×L×d·h_g) per group vs O(B×L×D) for naive approaches

### 9. Memory Requirements

Per-device memory footprint:
- Parameters: (D×D)/(m×n) = 4096×4096/16 = 1M parameters per matrix
- Activations: Reduced proportional to partition size
- Total parameter reduction: 16× compared to single device

### 10. Implementation Integration

- Compatible with existing model parallel frameworks
- Requires custom tensor partitioning primitives
- Gradient synchronization for training scenarios
- Adaptive partitioning based on hardware topology (number of heads vs. devices)
- CUDA-aware communication libraries (NCCL) for efficient all-gather operations