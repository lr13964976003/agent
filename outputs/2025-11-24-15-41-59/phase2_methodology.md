# Phase 2: Methodology Extraction

## Problem Setup and Notation

### Input Space
- **Input sequence**: X ∈ ℝ^(B×L×d_model)
  - B: batch size
  - L: sequence length
  - d_model: model's hidden size
- **Multi-Head Attention**: H attention heads, each with dimension d_h = d_model/H
- **Projections**: Q = XW_Q, K = XW_K, V = XW_V where W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

### Distributed Setting
- **Devices**: P distributed devices {D₀, D₁, ..., Dₚ₋₁}
- **Objective**: Compute MHA with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation

### Data Partitioning
- **Sequence dimension split**: X = [X⁽⁰⁾, X⁽¹⁾, ..., X⁽ᴾ⁻¹⁾]
- **Per device data**: X⁽ᵖ⁾ ∈ ℝ^(B×L/P×d_model) resides on device D_p
- **Memory reduction**: Activation memory per device reduced by factor P
- **Challenge**: Self-attention requires all keys K and values V across entire sequence

## Ring Attention Algorithm

### Ring Topology Structure
- **Logical ring**: Devices arranged in fixed ring order
- **Communication pattern**: Sequential peer-to-peer exchanges
- **Stages**: P stages for P devices

### Algorithm Steps

#### 1. Initialization Phase
```
Each device computes:
- Q⁽ᵖ⁾, K⁽ᵖ⁾, V⁽ᵖ⁾ from local X⁽ᵖ⁾
- Initial KV_block = (K⁽ᵖ⁾, V⁽ᵖ⁾)
```

#### 2. Ring Communication Loop
```
For t = 0 to P-1 stages:
    src_idx = (p - t) mod P
    
    Each device D_p:
    - Computes partial attention: partial = Attention(Q⁽ᵖ⁾, KV_block)
    - Accumulates: output_p += partial
    - Sends KV_block to next device in ring
    - Receives new KV_block from previous device
```

#### 3. Final Aggregation
After P stages, each device has computed attention outputs for its local queries using all keys and values across the sequence.

### Pseudocode Implementation
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        
        # Ring communication
        send KV_block to next device
        receive KV_block from previous device
```

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
1. **Data placement**: Sequence parallelism defines how data is distributed
2. **Communication**: Ring Attention defines communication order instead of all-gather
3. **Memory efficiency**: Combines benefits of both approaches

### Implementation Details
- **Topology**: NCCL's send/recv primitives or MPI point-to-point operations
- **Overlap**: Computation overlaps with asynchronous communication
- **Precision**: Mixed-precision (fp16/bf16) for Q, K, V tensors
- **Fused kernels**: Projection and softmax fused with communication hooks
- **Scalability**: Benefits scale with L and P, particularly for L > 16k tokens

## Communication Complexity Analysis

### Comparison
- **Naive all-gather**: Each device exchanges O(L·d_model) per step
- **Ring Attention**: Each device exchanges O(L/P·d_model) per stage
- **Total volume**: Same O(L·d_model) but lower peak bandwidth
- **Memory cost**: Activation memory drops from O(L·d_model) to O(L/P·d_model)