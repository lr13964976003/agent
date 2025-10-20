# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures employing multi-head attention (MHA) have become central to state-of-the-art models. As model sizes grow exponentially, efficient distribution across hardware units becomes critical. Traditional MHA parallelization splits attention heads across devices, but this approach leads to suboptimal utilization when device count exceeds head count and creates communication bottlenecks.

Our two-level partitioning strategy extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. This fine-grained partitioning enables flexible scaling, better memory distribution, and reduced inter-device communication.

## Methodology

### Two-Level Partitioning Scheme

The proposed method partitions MHA along two dimensions:

1. **Head Dimension Partitioning**: Total *h* heads divided into *n* groups, each containing *h/n* heads
2. **Intra-Head Dimension Partitioning**: Each head's feature dimension *d* sliced into *m* segments of size *d/m*

This creates *m × n* partitions, where each partition corresponds to a (head group, dimension slice) pair.

### Mathematical Formulation

Given input tensor *X* ∈ ℝ^(B×L×D) where:
- *B*: batch size (1024)
- *L*: sequence length (10000)
- *D*: embedding dimension = *h* × *d* = 8192
- *h*: number of heads (16)
- *d*: dimension per head (512)

### Partitioning Parameters
- *h_g = h/n = 16/4 = 4*: heads per group
- *d_s = d/m = 512/4 = 128*: slice dimension per partition
- *m × n = 4 × 4 = 16*: total partitions

### Implementation Details

**Weight Matrix Partitioning:**
- W_Q, W_K, W_V ∈ ℝ^(8192×8192) partitioned into 16 blocks
- Each block W^(i,j) ∈ ℝ^(2048×2048)
- Each device handles one partition

**Device Assignment:**
- 16 NVIDIA H100 GPUs arranged in 4×4 grid
- Device (i,j) handles head group *i* and dimension slice *j*

**Computation Flow:**
1. Each device computes Q^(i,j), K^(i,j), V^(i,j) for its partition
2. Local attention computation: Attention^(i,j) = softmax(Q^(i,j)K^(i,j)ᵀ/√d_s)V^(i,j)
3. Hierarchical aggregation:
   - Concatenate dimension slices within each head group
   - Concatenate head groups for final MHA output

## Experiments

### Experimental Setup
- **Hardware**: 16 NVIDIA H100 GPUs, FP16 precision
- **Model**: 2-layer Dense Transformer
- **Parameters**: Fixed at batch=1024, seq_len=10000, heads=16, head_dim=512, MLP_hidden=32768

### Baseline Configuration
- **Method**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Utilization**: 16 GPUs fully utilized
- **Expected**: State-of-the-art deployment strategy

### Results

| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (4×4=16) | 1,580,000 | 0.22 |

**Performance Improvements:**
- **31.7% throughput increase** (1.2M → 1.58M tokens/sec)
- **37.1% communication reduction** (0.35 → 0.22 ms TPOT)

## Technical Contributions

1. **Scalability**: Supports deployment on *m × n* devices exceeding traditional head-wise splitting limits
2. **Load Balancing**: Even workload distribution across heads and dimensions
3. **Memory Efficiency**: Each device stores only 1/(m×n) fraction of parameters and activations
4. **Communication Optimization**: Localized intra-head partitions reduce cross-device synchronization

## Deployment Configuration

The method enables deployment across large-scale distributed infrastructures for both training and inference by adapting gradient synchronization accordingly. Choice of *m* and *n* depends on hardware topology and network bandwidth considerations.

## Conclusion

Our two-level partitioning method combines head-wise and intra-head dimension-wise slicing to effectively balance workload, reduce synchronization costs, and better leverage large-scale hardware resources. The substantial improvements in inference throughput (up to 31.7%) and communication overhead reduction (over 30%) validate this approach for efficient distributed deployment of massive transformer architectures.