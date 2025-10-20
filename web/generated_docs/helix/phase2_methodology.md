# Helix: Two-Level Attention Partitioning - Detailed Methodology

## Method Overview
- **Two-level partitioning** for MHA mechanism
- **Fine-grained distribution** beyond conventional head-wise splitting
- **m×n partitions** where n=head splits, m=intra-head dimension splits

## Mathematical Foundation

### Input Tensor
- Shape: X ∈ ℝ^(B×L×D)
  - B: batch size
  - L: sequence length  
  - D: embedding dimension = h×d
  - h: number of heads
  - d: dimension per head

### Weight Matrices
- W_Q, W_K, W_V ∈ ℝ^(D×D)
- Each matrix partitioned into blocks W^(i,j)
  - i ∈ [1,n]: head group index
  - j ∈ [1,m]: dimension slice index
  - W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

### Partitioning Parameters
- **h_g = h/n**: heads per group
- **d_s = d/m**: slice dimension per partition
- **Total partitions**: m×n

## Implementation Details

### Step 1: Weight Matrix Partitioning
Each W_Q/W_K/W_V matrix is partitioned:
1. **Output dimension**: split into h heads
2. **Within each head**: split into m dimension slices
3. **Result**: m×n blocks W^(i,j) per projection

### Step 2: Device Assignment
Each device handles partition (i,j) computing:
- **Q^(i,j) = X W_Q^(i,j)**: Query projection for slice (i,j)
- **K^(i,j) = X W_K^(i,j)**: Key projection for slice (i,j)
- **V^(i,j) = X W_V^(i,j)**: Value projection for slice (i,j)

### Step 3: Attention Computation
Each device computes:
```
Attention^(i,j) = softmax(Q^(i,j) × (K^(i,j))ᵀ / √d_s) × V^(i,j)
```

### Step 4: Hierarchical Aggregation
Two-stage concatenation:
1. **Within head group**: Concatenate m dimension slices j=1..m for each head group i
2. **Across head groups**: Concatenate n head groups i=1..n for final MHA output

## Communication Patterns

### Input Distribution
- Each device receives corresponding input slice for its partition
- Broadcast X to all devices initially

### Intra-group Communication
- Devices within same head group communicate to concatenate dimension slices
- Localized communication reduces bandwidth

### Output Aggregation
- Dimension slices within each head group concatenated
- Head group outputs concatenated without additional communication if placement optimized

## Memory Efficiency
- **Parameter storage**: Each device stores only m×n fraction of total parameters
- **Activation storage**: Each device stores only m×n fraction of intermediate activations
- **Reduced memory footprint** enables larger models/batches

## Precision and Training
- **Mixed precision supported** (FP16/BF16 for efficiency)
- **Gradient synchronization** adapts to partitioning scheme
- **Compatible with existing frameworks** via custom tensor partitioning

## Hardware Topology Considerations
- **Choice of m and n** depends on:
  - Number of available devices
  - Network bandwidth between devices
  - Memory capacity per device
- **Optimal mapping**: Place partitions to minimize inter-node communication
- **Scalability**: Linear scaling with m×n up to hardware limits