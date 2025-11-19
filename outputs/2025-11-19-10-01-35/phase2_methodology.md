# Phase 2: Methodology Extraction

## Multi-Head Attention Background

### Input Tensor
- Input: $X \in \mathbb{R}^{B \times L \times D}$
- Where: $B$ = batch size, $L$ = sequence length, $D$ = embedding dimension

### MHA Projections
- $Q, K, V = XW_Q, XW_K, XW_V$
- Weight matrices: $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$
- $D$ split into $h$ heads, each with dimension $d = D / h$

### Attention Computation Per Head
- Each head $i$ computes: $\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right) V_i$
- All head outputs concatenated for final output

## Two-Level Partitioning Scheme

### Partitioning Parameters
- $h$: total number of heads
- $d$: dimension per head
- $n$: number of head partitions (head groups)
- $m$: number of intra-head dimension slices
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition

### Step 1: Weight Matrix Partitioning
Each projection matrix $W \in \mathbb{R}^{D \times D}$ (for Q, K, V) is partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1, n]$: indexes the head group
- $j \in [1, m]$: indexes the intra-head dimension slice
- Each block: $W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$

### Step 2: Device Assignment
- Total partitions: $m \times n$
- Each device handles one partition $(i,j)$
- Each device receives corresponding input tensor slices

### Step 3: Computation on Each Partition
Each device $(i,j)$ computes:
- $Q^{(i,j)} = X W_Q^{(i,j)}$
- $K^{(i,j)} = X W_K^{(i,j)}$
- $V^{(i,j)} = X W_V^{(i,j)}$
- $\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$

### Step 4: Result Aggregation
Two-stage concatenation process:
1. **Dimension concatenation**: Slices $j = 1,...,m$ within each head group $i$ concatenated along feature dimension
2. **Head concatenation**: Outputs from all head groups $i = 1,...,n$ concatenated along head dimension

Final output: $\text{Output} = \text{Concat}_{i=1}^n \left( \text{Concat}_{j=1}^m \text{Attention}^{(i,j)} \right)$

## Communication and Synchronization Details

### Required Communications
1. **Input distribution**: Each device receives corresponding input slice for projections
2. **Intra-group concatenation**: Partial results within each head group must be concatenated
3. **Final concatenation**: Head group outputs concatenated if not co-located

### Communication Optimization
- **Hierarchical partitioning** reduces communication overhead compared to naive full-dimension splits
- **Localized intra-head dimension partitions** minimize cross-device synchronization bandwidth
- **Load balancing** achieved through even division of both heads and dimensions

## Memory and Storage Benefits
- Each device stores only fraction of MHA parameters
- Reduced intermediate activation memory footprint per device
- Memory usage scales inversely with $m \times n$

## Implementation Considerations
- Compatible with existing model parallel frameworks
- Requires custom tensor partitioning and communication primitives
- Supports both training and inference with adapted gradient synchronization
- Parameter selection ($m$, $n$) depends on hardware topology and network bandwidth