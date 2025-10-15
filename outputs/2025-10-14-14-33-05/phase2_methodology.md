# Phase 2: Detailed Methodology

## Mathematical Formulation

### Input Specifications
- Input tensor: X ∈ ℝ^(B×L×D)
  - B: batch size = 1024
  - L: sequence length = 10000
  - D: embedding dimension = 8192 (h×d = 16×512)

### MHA Parameters
- h: number of heads = 16
- d: dimension per head = 512
- n: number of head partitions = 4 (for m×n=16)
- m: number of dimension partitions per head = 4 (for m×n=16)
- h_g: heads per group = h/n = 16/4 = 4
- d_s: slice dimension per partition = d/m = 512/4 = 128

### Weight Matrix Partitioning
For each projection matrix W_Q, W_K, W_V ∈ ℝ^(D×D):
- Partitioned into blocks W^(i,j) where:
  - i ∈ [1,n] = [1,2,3,4] (head group index)
  - j ∈ [1,m] = [1,2,3,4] (dimension slice index)
  - Each block W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g) = ℝ^(128×4 × 128×4) = ℝ^(512×512)

### Computation Flow

#### Step 1: Input Projection
Each device (i,j) computes:
- Q^(i,j) = X W_Q^(i,j) ∈ ℝ^(B×L×d_s·h_g) = ℝ^(1024×10000×512)
- K^(i,j) = X W_K^(i,j) ∈ ℝ^(1024×10000×512)
- V^(i,j) = X W_V^(i,j) ∈ ℝ^(1024×10000×512)

#### Step 2: Attention Computation
Each device computes scaled dot-product attention:
- Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)
- Where d_s = 128 (slice dimension)
- Output: Attention^(i,j) ∈ ℝ^(1024×10000×512)

#### Step 3: Hierarchical Aggregation
1. **Intra-group concatenation**: For each head group i, concatenate m dimension slices
   - Concat_j Attention^(i,j) for j ∈ [1,4] → ℝ^(1024×10000×2048)
   
2. **Inter-group concatenation**: Concatenate n head groups
   - Concat_i [Concat_j Attention^(i,j)] for i ∈ [1,4] → ℝ^(1024×10000×8192)

## Communication Pattern

### Device Mapping
- Total devices: 16 (4×4 grid)
- Device (i,j) handles partition (i,j)
- Grid mapping:
  - Rows: head groups (i=1,2,3,4)
  - Columns: dimension slices (j=1,2,3,4)

### Communication Steps
1. **Input broadcast**: All devices receive full input X ∈ ℝ^(1024×10000×8192)
2. **Intra-group communication**: Devices in same row (same head group) exchange partial results
3. **Final concatenation**: No additional communication needed if devices are arranged hierarchically

## Memory Requirements
- Each device stores:
  - 3 weight matrices: W_Q^(i,j), W_K^(i,j), W_V^(i,j) ∈ ℝ^(512×512) each
  - Total parameters per device: 3×512×512 = 786,432 parameters
  - Compared to full model: 3×8192×8192 = 201,326,592 parameters (256× reduction per device)

## Baseline Comparison
### Baseline: TP=8 + PP=2
- Tensor Parallelism degree: 8
- Pipeline Parallelism degree: 2
- Total devices: 8×2 = 16
- Different partitioning strategy but same hardware count for fair comparison