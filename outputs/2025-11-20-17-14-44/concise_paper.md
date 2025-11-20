# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures, particularly those employing multi-head attention (MHA), have become the cornerstone of state-of-the-art models in natural language processing and beyond. As model sizes continue to grow exponentially, efficiently distributing their computations across multiple hardware units becomes critical. Traditional MHA parallelization typically involves splitting the attention heads across devices; however, this approach alone can lead to suboptimal utilization and communication bottlenecks when the number of available devices exceeds the number of heads.

In this work, we introduce a novel partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions that can be mapped onto *m × n* devices. This fine-grained partitioning scheme enables more flexible scaling, better memory distribution, and reduced inter-device communication by localizing computations more effectively.

## Method

### Two-Level Partitioning Scheme

#### Core Innovation
- **Level 1**: Head-level partitioning - split *h* heads into *n* groups (h/n heads per group)
- **Level 2**: Intra-head dimension partitioning - split each head's *d* dimensions into *m* segments (d/m per segment)
- **Result**: *m × n* total partitions for *m × n* devices

#### Mathematical Formulation
- **Fixed Parameters**:
  - Batch size: 128
  - Sequence length: 10000
  - Heads: h = 32
  - Head dimension: d = 128
  - Total embedding: D = h × d = 4096
  - MLP hidden: 32768

- **Partitioning**:
  - Head groups: n = 4 → h_g = 8 heads/group
  - Dimension slices: m = 4 → d_s = 32/slice
  - Total partitions: m × n = 16

#### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (Q,K,V) partitioned into blocks W^(i,j) where:
- i ∈ [1,n] (head group index)
- j ∈ [1,m] (dimension slice index)
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g) = ℝ^(32×8 × 32×8)

#### Computation Flow
1. **Input**: X ∈ ℝ^(B×L×D) = ℝ^(128×10000×4096)
2. **Projection**: Each device computes Q^(i,j), K^(i,j), V^(i,j) for its partition
3. **Attention**: Local attention computation using scaled dot-product
4. **Aggregation**: 
   - Concatenate dimension slices within each head group
   - Concatenate head groups to produce final output

#### Memory and Communication
- **Memory per device**: 1/16 of total parameters and activations
- **Communication**: Hierarchical concatenation reduces inter-device synchronization
- **Load balancing**: Equal workload distribution across all 16 devices

## Experiments

### Setup
- **Hardware**: 16×NVIDIA H100 GPUs, 80GB HBM3 each
- **Model**: 4-layer Dense Transformer
- **Precision**: FP16 (mixed precision)
- **Fixed dimensions**: Batch=128, Seq=10000, Heads=32, Head_dim=128

### Baseline vs Proposed
| Method | Configuration | TPS (tokens/sec) | TPOT (ms) |
|--------|---------------|------------------|-----------|
| Baseline | TP=8, PP=2 | 1,200,000 | 0.35 |
| Proposed | m×n=16 | 1,580,000 | 0.22 |

### Performance Gains
- **Throughput improvement**: +31.7% (380,000 tokens/sec increase)
- **Overhead reduction**: -37.1% (0.13ms TPOT decrease)
- **Hardware utilization**: >95% across all 16 devices

### Device Mapping
4×4 grid mapping:
- Rows: head groups (0-3)
- Columns: dimension slices (0-3)
- Each device handles: 8 heads × 32 dimensions = 1024 dimensions
- Total: 16 devices fully utilized

## Conclusion
Our two-level partitioning method enables deployment of transformer models on m×n=16 devices with 31.7% throughput improvement and 37.1% overhead reduction. This approach provides a scalable solution for distributing large transformer models beyond traditional head-wise partitioning limits.