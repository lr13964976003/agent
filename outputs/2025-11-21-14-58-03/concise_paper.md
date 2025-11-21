# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Deployment

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures with multi-head attention (MHA) face scaling challenges as model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this approach is limited by the fixed number of heads (typically 32-96) and leads to suboptimal utilization when available devices exceed head count. We introduce a two-level partitioning strategy that extends beyond head-wise splitting by further segmenting each attention head's internal dimension, enabling deployment on m×n devices regardless of head count.

## Method

### Two-Level Partitioning Scheme
We partition MHA along two dimensions:
1. **Head Dimension Partitioning**: Total h heads divided into n groups, each containing h/n heads
2. **Intra-Head Dimension Partitioning**: Each head's feature dimension d sliced into m segments of size d/m

### Parameters for 16-Device Deployment
- **h**: 32 heads total
- **d**: 128 dimensions per head  
- **D**: 4096 total embedding dimension (h×d)
- **n**: 4 head groups (h_g = 8 heads per group)
- **m**: 4 dimension slices (d_s = 32 dimensions per slice)
- **Total partitions**: m×n = 16 partitions → 16 devices

### Weight Matrix Partitioning
Each projection matrix W_Q, W_K, W_V ∈ ℝ^(4096×4096) partitioned into 16 blocks:
- Block dimensions: ℝ^(d_s×h_g × d_s×h_g) = ℝ^(256×256) per device
- Each device stores: 3×256×256 = 196,608 parameters total

### Computation Flow
1. **Input projection**: Each device computes Q^(i,j), K^(i,j), V^(i,j) using its 256×256 weight block
2. **Attention computation**: Device (i,j) computes softmax(Q^(i,j)(K^(i,j))^⊤/√32)V^(i,j)
3. **Aggregation**: Two-stage concatenation - first within head groups (4 devices), then across groups

### Communication Pattern
- **Hierarchical reduction**: 4 devices per head group communicate for intra-group concatenation
- **Minimal global communication**: Results naturally distributed across final head groups
- **Memory footprint**: 16× reduction per device compared to full model

## Experiments

### Setup
- **Hardware**: 16× NVIDIA H100 GPUs, FP16 precision
- **Model**: 2-layer Dense Transformer
- **Fixed parameters**: 32 heads, 128 dim/head, seq_len=10000, batch=128, MLP_hidden=16384

### Baseline vs Proposed
| Method | Configuration | TPS (tokens/sec) | TPOT (ms) | Improvement |
|--------|---------------|------------------|-----------|-------------|
| Baseline | Tensor Parallelism=8 + Pipeline Parallelism=2 | 1,200,000 | 0.35 | - |
| Proposed | m×n=16 partitions | 1,580,000 | 0.22 | +31.7% throughput, -37.1% overhead |

### Analysis
The two-level partitioning achieves 31.7% throughput improvement and 37.1% communication overhead reduction by:
- Fully utilizing all 16 GPUs with fine-grained 4×4 partitioning
- Eliminating pipeline bubbles and tensor parallelism overhead
- Achieving perfect load balancing with equal 256×256 parameter blocks per device

## Conclusion
We presented a two-level partitioning method enabling deployment of MHA on m×n devices beyond head-count limitations. With 31.7% throughput gains and 37.1% overhead reduction demonstrated on 16 GPUs, this approach provides a scalable pathway for efficient distributed transformer deployment at unprecedented scales.