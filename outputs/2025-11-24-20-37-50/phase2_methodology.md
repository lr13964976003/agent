# Phase 2: Methodology Extraction

## Method Overview

### Two-Level Partitioning Scheme
Our proposed method partitions the Multi-Head Attention (MHA) layer along two dimensions:

1. **Head Dimension Partitioning** - divides total `h` heads into `n` groups, each containing `h/n` heads
2. **Intra-Head Dimension Partitioning** - splits each head's feature dimension `d` into `m` segments of size `d/m`

This creates `m × n` partitions total, each corresponding to a unique `(head group, dimension slice)` pair.

## Detailed Partitioning Process

### Step 1: Weight Matrix Partitioning

Each projection matrix `W ∈ ℝ^(D×D)` (for Q, K, V) is partitioned as follows:

- **Output dimension**: split into `h` heads
- **Input/output dimension per head**: split into `m` slices
- **Partition blocks**: `W^(i,j)` where `i ∈ [1,n]` indexes head group, `j ∈ [1,m]` indexes dimension slice
- **Partition size**: `W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)` where:
  - `h_g = h/n` (heads per group)
  - `d_s = d/m` (slice dimension per partition)

### Step 2: Input Tensor Processing

Each device handling partition `(i,j)` receives corresponding input tensor slices:

- **Query projection**: `Q^(i,j) = X W_Q^(i,j)`
- **Key projection**: `K^(i,j) = X W_K^(i,j)`  
- **Value projection**: `V^(i,j) = X W_V^(i,j)`

### Step 3: Attention Computation

Each device computes scaled dot-product attention for its assigned slice:

```
Attention^(i,j) = softmax((Q^(i,j) (K^(i,j))^T) / sqrt(d_s)) V^(i,j)
```

### Step 4: Result Aggregation

Outputs are aggregated hierarchically:

1. **Dimension concatenation**: Within each head group `i`, concatenate slices `j=1,...,m` along feature dimension
2. **Head concatenation**: Concatenate outputs from all head groups `i=1,...,n` along head dimension

```
Output = Concat_{i=1}^n (Concat_{j=1}^m Attention^(i,j))
```

## Communication Pattern

### Data Distribution
- Each device receives its corresponding input slice for projections

### Intra-group Communication
- Partial results within same head group require concatenation (devices in same group)

### Final Assembly
- After dimension-wise concatenation, head group outputs are concatenated
- Minimal additional communication needed if devices are placed accordingly

## Implementation Requirements

### Framework Integration
- Requires custom tensor partitioning primitives
- Compatible with existing model parallel frameworks
- Supports both training and inference modes

### Hyperparameter Selection
- Choice of `m` and `n` depends on:
  - Hardware topology
  - Network bandwidth
  - Total available devices
  - Model characteristics (h, d, D)

### Memory Requirements
- Each device stores: 
  - `1/(m×n)` fraction of MHA parameters
  - Corresponding fraction of intermediate activations

## Mathematical Formulation

Given:
- Input: `X ∈ ℝ^(B×L×D)`
- Total heads: `h`
- Head dimension: `d = D/h`
- Head partitions: `n`
- Dimension slices: `m`
- Partition sizes: `h_g = h/n`, `d_s = d/m`

Weight matrices:
- `W_Q^(i,j), W_K^(i,j), W_V^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)`

Output reconstruction:
- Dimension reconstruction: `Head_i = [Attention^(i,1); ...; Attention^(i,m)]`
- Final output: `Output = [Head_1; ...; Head_n]`

## Precision and Performance
- Supports mixed precision (FP16)
- Optimized for large batch sizes (128+ in experiments)
- Compatible with standard transformer architectures