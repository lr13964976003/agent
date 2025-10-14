# Phase 2: Methodology Extraction

## Overview of Two-Level Partitioning Method

### Core Concept
Proposed method introduces **two-level partitioning** for Multi-Head Attention (MHA) mechanism that goes beyond conventional head-wise splitting by adding intra-head dimension partitioning.

### Partitioning Dimensions
1. **Head Dimension Partitioning**: h heads → n groups (h_g = h/n heads per group)
2. **Intra-Head Dimension Partitioning**: d dimension per head → m segments (d_s = d/m per segment)
3. **Total Partitions**: m × n partitions → m × n devices

## Mathematical Formulation

### Input Specifications
- Input tensor: X ∈ ℝ^(B×L×D)
  - B: batch size
  - L: sequence length  
  - D: embedding dimension
- Total heads: h
- Dimension per head: d = D/h

### Partitioning Parameters
- n: number of head partitions
- m: number of dimension partitions per head
- h_g = h/n: heads per group (must be integer)
- d_s = d/m: slice dimension per partition (must be integer)

### Weight Matrix Partitioning

#### Projection Matrices
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n]: head group index
- j ∈ [1,m]: intra-head dimension slice index
- W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

#### Partition Structure
- Each W^(i,j) corresponds to portion of input/output feature spaces
- Blocks assigned to individual devices for parallel processing

## Computation Flow

### Per-Device Computation
Each device handling partition (i,j) performs:

1. **Input Projection**:
   - Q^(i,j) = X W_Q^(i,j)
   - K^(i,j) = X W_K^(i,j) 
   - V^(i,j) = X W_V^(i,j)

2. **Scaled Dot-Product Attention**:
   - Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)

### Output Aggregation
Hierarchical concatenation process:

1. **Intra-group concatenation**: Concatenate m dimension slices within each head group
   - For each head group i: Concat_j=1^m Attention^(i,j)

2. **Inter-group concatenation**: Concatenate n head groups
   - Final output: Concat_i=1^n (Concat_j=1^m Attention^(i,j))

## Communication Patterns

### Required Communications
1. **Input Distribution**: Each device receives corresponding input slice for projections
2. **Intra-group Communication**: Partial results concatenated within head groups
3. **Final Assembly**: Head group outputs concatenated (can be optimized with device placement)

### Communication Efficiency
- Hierarchical partitioning reduces communication overhead vs naive full-dimension splits
- Localized intra-head dimension partitions minimize cross-device synchronization

## Implementation Details

### Integration Requirements
- Compatible with existing model parallel frameworks
- Requires custom tensor partitioning and communication primitives
- Supports both training and inference (with adapted gradient synchronization)

### Parameter Selection
- Choice of m and n depends on:
  - Hardware topology
  - Network bandwidth
  - Model characteristics (h, d)
  - Total available devices (m × n)

### Memory Benefits
- Each device stores only fraction of MHA parameters
- Reduced intermediate activation memory footprint
- Enables deployment of larger models on fixed hardware

## Algorithm Summary

```
Input: X ∈ ℝ^(B×L×D), parameters m, n
Output: MHA(X) ∈ ℝ^(B×L×D)

1. Validate: h divisible by n, d divisible by m
2. Partition weights W_Q, W_K, W_V into m×n blocks
3. For each device (i,j) in parallel:
   a. Compute Q^(i,j), K^(i,j), V^(i,j)
   b. Compute Attention^(i,j)
4. Aggregate results:
   a. Concatenate dimension slices within groups
   b. Concatenate head groups
5. Return final output
```

## Key Constraints
- h must be divisible by n (integer heads per group)
- d must be divisible by m (integer dimension per slice)
- Total devices must equal m × n
- All devices must have identical computational capabilities