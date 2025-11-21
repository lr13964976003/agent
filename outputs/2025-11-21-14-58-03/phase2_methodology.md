# Phase 2: Methodology Extraction

## Method Overview
Our proposed **two-level partitioning method** for Multi-Head Attention (MHA) mechanism in large transformer models. This extends beyond conventional parallelism by partitioning both attention heads and intra-head dimensions.

## Multi-Head Attention Background

### Input Specifications
- Input tensor: X ∈ ℝ^(B×L×D)
  - B: batch size = 128
  - L: sequence length = 10000
  - D: embedding dimension = h×d = 32×128 = 4096

### Standard MHA Computation
1. **Projections**: Q, K, V = XW_Q, XW_K, XW_V
   - Each W ∈ ℝ^(D×D) = ℝ^(4096×4096)
2. **Head splitting**: h = 32 heads, each with d = 128 dimensions
3. **Attention per head**: Attention_i(Q_i, K_i, V_i) = softmax(Q_i K_i^⊤/√d) V_i
4. **Concatenation**: Combine all head outputs

## Two-Level Partitioning Scheme

### Partitioning Parameters
- **n**: Number of head groups = 4 (for 16-device deployment: m×n=16)
- **m**: Number of dimension slices per head = 4
- **h_g**: Heads per group = h/n = 32/4 = 8 heads
- **d_s**: Slice dimension per partition = d/m = 128/4 = 32 dimensions
- **Total partitions**: m×n = 4×4 = 16 partitions

### Step 1: Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group (i=1,2,3,4)
- j ∈ [1,m] indexes dimension slice (j=1,2,3,4)
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g) = ℝ^(32×8 × 32×8) = ℝ^(256×256)

### Step 2: Computation per Partition
Each device handling partition (i,j) computes:
1. **Input projection**:
   - Q^(i,j) = X W_Q^(i,j)
   - K^(i,j) = X W_K^(i,j)
   - V^(i,j) = X W_V^(i,j)

2. **Attention computation**:
   - Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^⊤/√d_s) V^(i,j)
   - Where d_s = 32 (slice dimension)

### Step 3: Result Aggregation
**Two-stage concatenation**:
1. **Intra-group concatenation**: Concatenate m=4 dimension slices per head group
   - Output_i = Concat_j=1^4 Attention^(i,j)
   - Each group output: ℝ^(L×d) = ℝ^(10000×128)

2. **Inter-group concatenation**: Concatenate n=4 head groups
   - Final output = Concat_i=1^4 Output_i
   - Final dimension: ℝ^(L×D) = ℝ^(10000×4096)

## Communication Pattern

### Device Assignment
- **16 devices** arranged in 4×4 grid
- **Device (i,j)** handles partition (i,j)
- **Row communication**: Devices in same head group (i) for intra-group concatenation
- **Column communication**: Not required due to hierarchical aggregation

### Communication Steps
1. **Input broadcast**: Each device receives appropriate X slice
2. **Intra-group reduce**: 4 devices per group exchange results
3. **Final gather**: Minimal - results naturally distributed

## Implementation Details

### Memory Distribution
- **Per device storage**: 1/(m×n) = 1/16 of total MHA parameters
- **Parameter storage**: 3×ℝ^(256×256) = 196,608 parameters per device
- **Activation storage**: Reduced by factor of 16

### Precision Configuration
- **Mixed precision**: FP16 for computation
- **Batch processing**: 128 samples
- **Sequence handling**: 10000 tokens per sequence

## Advantages Summary
1. **Scalability**: m×n = 16 devices > head count (32)
2. **Load Balancing**: Equal 256×256 blocks per device
3. **Memory Reduction**: 16× reduction per device
4. **Communication Efficiency**: Hierarchical aggregation reduces global communication