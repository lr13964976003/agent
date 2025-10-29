# HPipe Methodology

## Workflow Overview

HPipe operates in two distinct phases:
1. **Prepare Phase**: Determines optimal workload distribution and sequence slicing through dynamic programming
2. **Runtime Phase**: Executes inference with token-level pipeline parallelism

## Mathematical Formulation

### Model Partitioning
- **Model Structure**: LLM with n layers {l₁, l₂, ..., lₙ}
- **Partitioning**: Divided into N blocks {b₁, b₂, ..., bₙ} across N devices
- **Sequence Handling**: Input segmented into M subsequences in token dimension

### Execution Time Calculation

For stage i on device j:
```
tᵢⱼ = tᶜ(sᵢ, Σₖ₌₁ⁱ⁻¹ sₖ; dⱼ) + tᵗ(lⱼ, sᵢ, B)
```

Where:
- tᶜ: Computation time for subsequence sᵢ with previous subsequences on device dⱼ
- tᵗ: Transmission time for intermediate activations through bandwidth B

### Optimization Objective
```
T* ≤ maxᵢ∈N(Σⱼ₌₀ᴹ tᵢⱼ) + (N-1) max₀≤i<M,0≤j<N{tᵢⱼ}
```

## Distribution Balance Algorithm

### Problem Formulation
- **Input**: Computation and communication time per layer for each device
- **Output**: Minimal slowest execution time and corresponding workload distribution

### Dynamic Programming Approach
```
T(a,b,m) = Σₖ₌ₐᵇ t_comp(lₖ; dₘ) + t_comm(lⱼ, m)
A[b][m] = min₁≤k<j{max{A[k][m-1], T(k+1,b,m)}}
```

### Algorithm Steps
1. Initialize execution time for different layer counts on first device
2. Apply dynamic programming for balanced distribution
3. Record cut-off points for optimal partitioning
4. Derive workload distribution schema from cut points

## Sequence Schedule Algorithm

### Key Insight
Token processing time increases linearly with position index due to attention mechanism requiring all previous tokens.

### Dynamic Programming Formulation
```
For maximum execution time t_max:
- Enumerate possible t_max values
- For each sequence length, find optimal slicing
- Store minimal latency for each sequence length
- Select optimal schema among all t_max enumerations
```

### Algorithm Details
1. Initialize arrays to store latency and slicing information
2. Iterate through possible segment lengths
3. Apply dynamic programming to find optimal slicing under each t_max constraint
4. Select the configuration with minimal total latency

## Implementation Considerations

### Memory Management
- Cache K,V values for subsequent token calculations
- Balance memory usage across devices based on capacity
- Account for parameter storage and activation memory

### Communication Patterns
- Intra-host: PCIe communication
- Inter-host: Network communication (1000 Mbps in evaluation)
- Minimize synchronization frequency through token-level parallelism

### Heterogeneity Handling
- Layer assignment based on device computational capability
- Communication cost consideration in partitioning decisions
- Runtime adaptation to varying network conditions