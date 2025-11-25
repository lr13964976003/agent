# Phase 2: Detailed Methodology - Helix Two-Level Attention Partitioning

## Complete Mathematical Formulation

### Input Specifications
Given input tensor: X ∈ ℝ^(B×L×D)
Where:
- B = 128 (batch size)
- L = 10000 (sequence length)
- D = 4096 (embedding dimension)

### Multi-Head Attention Structure
- Number of heads: h = 32
- Dimension per head: d = 128
- Total heads: h = 32
- Total dimension: D = h × d = 32 × 128 = 4096

### Two-Level Partitioning Scheme

#### Partitioning Parameters
- **Head-level partitioning**: n = 4 groups
  - Heads per group: h_g = h/n = 32/4 = 8 heads
- **Dimension-level partitioning**: m = 4 slices per head
  - Dimension per slice: d_s = d/m = 128/4 = 32 dimensions
- **Total partitions**: m × n = 4 × 4 = 16 partitions

### Weight Matrix Partitioning

#### Query, Key, Value Projection Matrices
Each weight matrix W_Q, W_K, W_V ∈ ℝ^(D×D) = ℝ^(4096×4096)

#### Partition Structure
Each matrix is partitioned into blocks W^(i,j) where:
- i ∈ [1, n] = [1, 4] indexes head groups
- j ∈ [1, m] = [1, 4] indexes dimension slices
- Each block: W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g) = ℝ^(32×8 × 32×8) = ℝ^(256×256)

### Computation Flow on Each Partition (i,j)

#### Step 1: Input Projection
Each device handling partition (i,j) computes:
```
Q^(i,j) = X · W_Q^(i,j) ∈ ℝ^(B×L×d_s×h_g) = ℝ^(128×10000×32×8)
K^(i,j) = X · W_K^(i,j) ∈ ℝ^(128×10000×32×8)
V^(i,j) = X · W_V^(i,j) ∈ ℝ^(128×10000×32×8)
```

#### Step 2: Attention Computation
```
Attention^(i,j) = softmax(Q^(i,j) · (K^(i,j))^T / √d_s) · V^(i,j)
                = softmax(ℝ^(128×10000×10000) · ℝ^(128×10000×32×8))
                ∈ ℝ^(128×10000×32×8)
```

#### Step 3: Hierarchical Aggregation

**Phase 1: Intra-group concatenation**
For each head group i ∈ [1,4]:
```
Attention^(i) = Concatenate_{j=1}^m (Attention^(i,j))
              = Concatenate_{j=1}^4 (ℝ^(128×10000×32×8))
              ∈ ℝ^(128×10000×128×8)
```

**Phase 2: Inter-group concatenation**
```
Output = Concatenate_{i=1}^n (Attention^(i))
       = Concatenate_{i=1}^4 (ℝ^(128×10000×128×8))
       ∈ ℝ^(128×10000×4096)
```

### Memory Analysis

#### Parameters per Device
- Each device stores 1/(m×n) = 1/16 of total parameters
- Total parameters in MHA layer: 3 × D × D = 3 × 4096 × 4096 = 50,331,648
- Parameters per device: 50,331,648 / 16 = 3,145,728 parameters
- FP16 storage: 3,145,728 × 2 bytes = 6,291,456 bytes ≈ 6 MB

#### Activations per Device
- Activation tensor size: B × L × d_s × h_g
- = 128 × 10000 × 32 × 8 = 327,680,000 elements
- FP16 storage: 327,680,000 × 2 bytes = 655,360,000 bytes ≈ 625 MB

#### Communication Patterns

**Intra-group communication (4 devices per group)**:
- Each group has 4 devices sharing 8 heads
- Communication for dimension slice concatenation
- Bandwidth: 4 × 625 MB = 2.5 GB per group

**Inter-group communication**:
- Final concatenation across 4 groups
- Direct device-to-device communication based on network topology

### Implementation Constraints

#### Precision Requirements
- All computations use FP16 precision
- Reduce memory usage by 50% compared to FP32
- Maintain numerical stability through careful scaling

#### Synchronization Points
1. **Input broadcast**: Input X broadcast to all 16 devices
2. **Phase 1 synchronization**: Devices within each group synchronize after dimension concatenation
3. **Phase 2 synchronization**: Final output assembly requires inter-group communication

#### Device Mapping Strategy
- Device (i,j) handles partition (i,j)
- i ∈ [0,3] for head groups
- j ∈ [0,3] for dimension slices
- Enables optimal locality for intra-group communication