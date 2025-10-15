# Phase 2: Methodology Extraction - Helix Paper

## Complete Methodology Description

### Problem Formulation
Given an input tensor X ∈ ℝ^(B×L×D) where:
- B = batch size (1024)
- L = sequence length (10000)
- D = embedding dimension (h × d = 16 × 512 = 8192)

### Multi-Head Attention Structure
- Number of heads: h = 16
- Dimension per head: d = 512
- Total embedding dimension: D = h × d = 8192

### Two-Level Partitioning Scheme

#### Level 1: Head Dimension Partitioning
- Divide h = 16 heads into n groups
- Each group contains h_g = h/n heads
- For n = 4: h_g = 16/4 = 4 heads per group

#### Level 2: Intra-Head Dimension Partitioning
- Slice each head's feature dimension d into m segments
- Each segment has size d_s = d/m
- For m = 4: d_s = 512/4 = 128 dimensions per segment

#### Total Partitions
- m × n = 16 partitions total
- Each partition corresponds to (head group, dimension slice) pair
- Maps to 16 devices for full utilization

### Weight Matrix Partitioning

#### Projection Matrices (Q, K, V)
Each weight matrix W ∈ ℝ^(D×D) where D = 8192 is partitioned as:

W_Q, W_K, W_V ∈ ℝ^(8192×8192)

Partitioning scheme:
- Output dimension split: h = 16 heads
- Each head dimension: d = 512
- Further split each head: m = 4 segments of d_s = 128

Each partition W^(i,j) ∈ ℝ^(d_s × h_g × d_s × h_g) = ℝ^(128×4 × 128×4) = ℝ^(512×512)

Where:
- i ∈ [1,n] = [1,4] indexes head group
- j ∈ [1,m] = [1,4] indexes dimension slice

### Computation Flow per Partition

#### Input Processing
Each device (i,j) receives:
- Q^(i,j) = X W_Q^(i,j) ∈ ℝ^(B×L×d_s×h_g) = ℝ^(1024×10000×128×4)
- K^(i,j) = X W_K^(i,j) ∈ ℝ^(1024×10000×128×4)
- V^(i,j) = X W_V^(i,j) ∈ ℝ^(1024×10000×128×4)

#### Attention Computation
Each device computes:
Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)

Where:
- Q^(i,j) (K^(i,j))^T ∈ ℝ^(1024×10000×10000×h_g×h_g)
- d_s = 128 (dimension per segment)
- Final output per partition: ℝ^(1024×10000×128×4)

### Aggregation Process

#### Step 1: Dimension Concatenation
Within each head group i:
Concatenate m=4 dimension slices:
Attention^(i) = Concat_{j=1}^4 Attention^(i,j) ∈ ℝ^(1024×10000×512×4)

#### Step 2: Head Concatenation
Concatenate n=4 head groups:
Output = Concat_{i=1}^4 Attention^(i) ∈ ℝ^(1024×10000×8192)

### Communication Patterns

#### Required Communications
1. **Input broadcast**: Each device receives appropriate input slice
2. **Intra-group communication**: Devices within same head group exchange dimension slices
3. **Final aggregation**: Concatenated outputs form final result

#### Communication Reduction
- Hierarchical partitioning reduces cross-device synchronization
- Localized intra-head dimension partitions minimize bandwidth usage
- No additional communication needed for final head group concatenation if placed appropriately

### Memory Efficiency
- Each device stores only 1/(m×n) = 1/16 of total parameters
- Intermediate activations similarly partitioned
- Reduced memory footprint enables larger model deployment

### Implementation Parameters
- **m**: 4 (dimension splits per head)
- **n**: 4 (head groups)
- **m×n**: 16 total partitions
- **Devices**: 16 NVIDIA H100 GPUs
- **Precision**: FP16 mixed precision
- **Batch size**: 1024
- **Sequence length**: 10000
- **Heads**: 16
- **Head dimension**: 512
- **MLP hidden size**: 32768