# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Deployment

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Methodology

### Two-Level Partitioning Scheme

#### Mathematical Formulation
- **Input**: X ∈ ℝ^(B×L×D) where B=128, L=10000, D=4096
- **Heads**: h=32, dimension per head d=128
- **Partitioning**: n head groups, m dimension slices per head
- **Total partitions**: m×n=16 for 16 GPU deployment

#### Partitioning Process
1. **Head-level split**: h=32 heads → n=4 groups of h_g=8 heads each
2. **Dimension-level split**: d=128 → m=4 segments of d_s=32 dimensions each
3. **Result**: 4×4=16 partitions, each handling 8 heads × 32 dimensions

#### Computation Flow
Each partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)
Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)
```

#### Hierarchical Aggregation
1. Intra-group: Concatenate m=4 dimension slices within each head group
2. Inter-group: Concatenate n=4 head groups for final output

### Implementation Details
- **Memory per device**: 1/(m×n) of total parameters (1,048,576 parameters)
- **Communication**: Localized intra-group, hierarchical aggregation
- **Compatibility**: Integrates with existing model parallel frameworks
- **Precision**: FP16 for memory efficiency

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 4-layer Dense Transformer
- **Configuration**: 32 heads × 128 dimensions, batch=128, seq_len=10000

### Results
| Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|--------|------------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 | - |
| Proposed (m×n=16) | 1,580,000 | 0.22 | +31.7% throughput, -37.1% overhead |

### Key Findings
- **31.7% throughput improvement** over traditional tensor/pipeline parallelism
- **37.1% reduction in communication overhead** (0.35ms → 0.22ms per token)
- **Full 16-GPU utilization** with balanced workload distribution
- **Scalability beyond head count limitations**

## Conclusion
The two-level partitioning method enables efficient deployment of MHA computations across m×n devices, achieving substantial improvements in inference throughput while reducing communication overhead. This approach provides a practical pathway for scaling transformer models to larger distributed systems.