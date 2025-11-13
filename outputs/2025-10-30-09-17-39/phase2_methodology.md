# HPipe Methodology - Detailed Extraction

## 3.1 Workflow Overview
HPipe operates in two phases:
1. **Prepare Phase**: Determines optimal workload distribution and sequence slicing using dynamic programming
2. **Runtime Phase**: Executes inference with token-level pipeline parallelism

## 3.2 Mathematical Formulation

### Model Structure
- LLM with n layers {l₁, l₂, ..., lₙ}
- Divided into N blocks {b₁, b₂, ..., bₙ} distributed across N devices
- Input sequence segmented into M subsequences in token dimension

### Execution Time Model
For each stage in pipeline:
```
tᵢⱼ = t_c(sᵢ, Σₖ₌₁ⁱ⁻¹ sₖ; dⱼ) + t_t(lⱼ, sᵢ, B)
```
Where:
- t_c: computation latency for subsequence sᵢ with previous subsequences s₁...sᵢ₋₁
- t_t: transmission time for intermediate activation from layer lⱼ with bandwidth B

### Optimization Objective
Minimize total latency T* with constraint:
```
T* ≤ maxᵢ∈N(Σⱼ₌₀ᴹ tᵢⱼ) + (N-1)max₀≤i<M,0≤j<N{tᵢⱼ}
```

## 3.3 Distribution Balance Algorithm

### Problem Formulation
- **NP-hard device placement problem**
- Partition layers at granularity of individual layers (not transformer blocks)
- Dynamic programming approach with optimal substructure

### Dynamic Programming Solution
```
T(a,b,m) = Σₖ₌ₐᵇ t_comp(lₖ; dₘ) + t_comm(lⱼ, m)
A[b][m] = min₁≤k<j{max{A[k][m-1], T(k+1,b,m)}}
```
Where A[b][m] is minimum execution time for layers 1..b on first m devices

### Algorithm Details
- **Input**: Computation and communication time per layer for each device
- **Output**: Minimal slowest execution time A[N][M] and workload distribution schema
- **Complexity**: Polynomial time via dynamic programming

## 3.4 Sequence Schedule Algorithm

### Key Insight
- Execution time per token increases linearly with position index
- Longer slices at beginning, shorter slices toward end
- Balance between granularity and parallelism

### Optimization Formulation
```
T* ≤ min_{tₘ}{maxᵢ∈N{min_{S*∈S}{Σⱼ₌₀ᴹ tᵢⱼ | tᵢⱼ ≤ tₘ}} + (N-1)tₘ}
```
Where tₘ = max{tᵢⱼ} restricts each slice to similar execution time

### Dynamic Programming Sequence Slicing
- **Input**: Maximum execution time tₘₐₓ, execution times for different length slices
- **Process**: Enumerate possible tₘ values, find optimal slicing scheme
- **Output**: Optimal sequence segmentation {s₀, s₁, ..., sₘ}

## Token-Level Pipeline Details

### Architecture
- **Embedding phase**: Initial token embeddings computed
- **Transformer blocks**: L layers processed in pipeline
- **Attention mechanism**: K,V values cached for subsequent tokens
- **Sequential processing**: Subsequences processed in order while maintaining causal attention

### Communication Pattern
- **Between devices**: Intermediate activations transferred
- **Within pipeline stages**: No communication until stage completion
- **Memory optimization**: K,V cache reused across subsequence processing

### Device Mapping Strategy
1. **Compute capability assessment**: Measure FLOPS per device
2. **Communication bandwidth**: Measure PCIe vs network bandwidth
3. **Layer assignment**: More layers to faster devices, considering communication overhead
4. **Dynamic adjustment**: Rebalance based on actual performance measurements