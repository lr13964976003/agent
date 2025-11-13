# HPipe: Detailed Methodology

## 3.1 Workflow Design

### Two-Phase Architecture
**Prepare Phase**: 
- Input: Device specifications and network conditions
- Process: Determine optimal workload distribution and sequence slicing using dynamic programming
- Output: Preprocessed optimal slicing schemes for all supported sequence lengths

**Runtime Phase**:
- Input: Sequence S of length L
- Process: Divide S into subsequences s₀,...,sₘ and execute sequentially across devices
- Pipeline: Device dᵢ handles computation for sᵢ while dᵢ₋₁ and dᵢ₊₁ process adjacent subsequences

## 3.2 Mathematical Formulation

### System Model
- **LLM Structure**: n layers {l₁,...,lₙ} divided into N blocks {b₁,...,bₙ} across N devices
- **Sequence Segmentation**: Input sequence divided into M subsequences in token dimension
- **Execution Time**: tᵢⱼ = execution time of stage i on device j

### Execution Time Components
```
tᵢⱼ = tᶜ(∑ₘ₌₁ⁱ⁻¹ sₘ; dⱼ) + tᵗ(lⱼ, sᵢ, B)
```
Where:
- tᶜ: computation latency for given subsequence and previous subsequences
- tᵗ: transmission time based on intermediate activation size, subsequence length, and bandwidth
- lⱼ: last layer on device j
- B: bandwidth between devices

### Optimization Objective
Minimize total latency T* subject to:
```
T* ≤ maxᵢ∈N(∑ⱼ₌₀ᴹ tᵢⱼ) + (N-1)max₀≤i<M,₀≤j<N {tᵢⱼ}
```

## 3.3 Distribution Balance Algorithm

### Problem Formulation
- **NP-hard**: Device placement problem for heterogeneous environment
- **Assumption**: Constant device sequence for simplification
- **Granularity**: Layer-level partitioning (finer than transformer block)

### Dynamic Programming Approach
```
T(a,b,m) = ∑ₖ₌ₐᵇ t_comp(lₖ; dₘ) + t_comm(lⱼ, m)
A[b][m] = min₁≤k<j {max{A[k][m-1], T(k+1,b,m)}}
```

Where:
- A[b][m]: minimum execution time for layers 1 to b on m devices
- T(k+1,b,m): execution time for layers k+1 to b on device m
- Cut-off points recorded for deriving workload distribution

## 3.4 Sequence Schedule Algorithm

### Execution Time Model
- Linear relationship: execution time increases with token position index
- **tₘ** = max{tᵢⱼ}: key parameter to minimize overall latency

### Optimization Strategy
```
T* ≤ min_{tₘ} {max_{i∈N} {min_{S*∈S} {∑ⱼ₌₀ᴹ tᵢⱼ | tᵢⱼ ≤ tₘ}} + (N-1)tₘ}
```

### Dynamic Programming for Sequence Slicing
- **tₘ**: restricts each slice to similar execution time
- **S***: optimal slicing scheme from all possible tₘ values
- **Algorithm**: Derives optimal slicing using S-Sₙ recurrence

## Algorithm Details

### Algorithm 1: Workload Distribution
1. Initialize A[i][1] for single device scenarios
2. Dynamic programming to find A[N][M] for balanced distribution
3. Derive workload distribution schema from cut points
4. Returns: minimal slowest execution time and corresponding distribution

### Algorithm 2: Sequence Slicing
1. Enumerate possible tₘ values
2. For each tₘ, use dynamic programming to find optimal slicing
3. Store least latency in L[s_cur] and slice lengths in S[s_cur]
4. Select optimal schema among different tₘ values
5. Returns: optimal slicing {s₀,...,sₙ}

## Technical Specifications

### Device Heterogeneity Handling
- **Computation**: Different FLOPs capabilities across devices
- **Communication**: PCIe intra-host (high bandwidth) vs network inter-host (lower bandwidth)
- **Memory**: Varying memory capacities across P100 and RTX3090 GPUs

### Sequence Granularity Considerations
- **Fine-grained**: Small slices → GPU underutilization
- **Coarse-grained**: Large slices → fewer pipeline stages → reduced parallelism
- **Optimal**: Balanced based on computational volume and device capabilities

### Memory Optimization
- **KV Cache**: Cached across subsequence computations
- **Activation**: Intermediate activations transferred between devices
- **Parameter Distribution**: Based on device memory capacity and computational power