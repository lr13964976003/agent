# Phase 2: Methodology Extraction - Helix Two-Level Partitioning

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Methodology Details

### Two-Level Partitioning Scheme

#### Partitioning Parameters
- **h**: total number of heads = 32
- **d**: dimension per head = 128
- **n**: number of head partitions = 4
- **m**: number of dimension partitions per head = 4
- **Total partitions**: m × n = 16
- **h_g**: heads per group = h/n = 32/4 = 8 heads
- **d_s**: slice dimension per partition = d/m = 128/4 = 32
- **D**: total embedding dimension = h × d = 32 × 128 = 4096

#### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (for Q, K,V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes the head group (1 to 4)
- j ∈ [1,m] indexes the intra-head dimension slice (1 to 4)
- Each block W^(i,j) ∈ ℝ^(d_s × h_g × d_s × h_g) = ℝ^(32×8×32×8)

#### Input Tensor Processing
Given input tensor X ∈ ℝ^(B×L×D) where:
- B = batch size = 128
- L = sequence length = 10000
- D = 4096

Each device handling partition (i,j) receives corresponding slices:
- Q^(i,j) = X W_Q^(i,j)
- K^(i,j) = X W_K^(i,j)
- V^(i,j) = X W_V^(i,j)

#### Computation per Partition
Each device computes scaled dot-product attention for its slice:
```
Attention^(i,j) = softmax(Q^(i,j) × (K^(i,j))^T / sqrt(d_s)) × V^(i,j)
```
Where d_s = 32 (slice dimension)

#### Aggregation Process
1. **First-level concatenation**: Within each head group i, concatenate dimension slices j=1..m
   - Concatenate along feature dimension to reconstruct full head outputs
   - Each head group produces output: ℝ^(B×L×d×h_g) = ℝ^(128×10000×128×8)

2. **Second-level concatenation**: Across head groups i=1..n
   - Concatenate along head dimension to reconstruct full MHA output
   - Final output: ℝ^(B×L×D) = ℝ^(128×10000×4096)

### Memory and Communication Analysis

#### Memory Footprint per Device
- Parameters: Each device stores 1/(m×n) = 1/16 of total MHA parameters
- Activations: Each device processes 1/(m×n) = 1/16 of activation tensor
- Memory reduction factor: 16× compared to single device

#### Communication Patterns
1. **Input distribution**: Broadcast input tensor X to all m×n devices
2. **Intra-group communication**: Devices within same head group communicate for dimension concatenation
3. **Final aggregation**: No additional communication needed if devices are placed hierarchically

### Implementation Specifications

#### Precision Settings
- Mixed precision: FP16 for computation
- Batch size: 128 (fixed)
- Sequence length: 10000 (fixed)

#### Hardware Mapping
- Total devices: 16 NVIDIA H100 GPUs
- Device arrangement: 4×4 grid (n=4 head groups, m=4 dimension slices)
- Memory per device: ~80GB HBM3
- Interconnect: NVLink for intra-group, InfiniBand for inter-group communication