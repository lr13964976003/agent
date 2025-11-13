# Phase 2: Detailed Methodology Extraction

## Problem Setup and Notation

### Input Definition
- **Input tensor**: X ∈ ℝ^(B×L×d_model)
  - B: batch size
  - L: sequence length
  - d_model: model's hidden dimension
- **Attention heads**: H heads, each with dimension d_h = d_model/H
- **Transform matrices**: W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)
- **Devices**: P distributed devices {D_0, D_1, ..., D_{P-1}}

### MHA Computation
For single head:
```
Attn(Q, K, V) = softmax(QK^T/√d_h) V
where: Q = XW_Q, K = XW_K, V = XW_V
```

## Method 1: Sequence Parallelism

### Data Partitioning
Sequence dimension L is split across P devices:
```
X = [X^(0), X^(1), ..., X^(P-1)]
where X^(p) ∈ ℝ^(B×(L/P)×d_model) on device D_p
```

### Memory Impact
- **Before**: Each device stores O(L·d_model)
- **After**: Each device stores O(L/P·d_model)
- **Reduction factor**: P (number of devices)

### Communication Challenge
- **Requirement**: Each device needs all K and V tensors for full sequence
- **Naïve approach**: All-gather operation for K and V
- **Cost**: O(L·d_model) per device exchange

## Method 2: Ring Attention

### Ring Topology Structure
- **Logical arrangement**: Devices form a ring D_0 → D_1 → ... → D_{P-1} → D_0
- **Communication pattern**: Peer-to-peer exchanges between neighbors
- **Stages**: P sequential stages (t = 0 to P-1)

### Algorithm Steps

#### Initialization Phase
Each device computes local projections:
```
Q^(p), K^(p), V^(p) = ComputeProjections(X^(p))
```

#### Ring Communication Phase
For each stage t (0 ≤ t < P):
1. **Source calculation**: src = (p - t) mod P
2. **Partial attention**: Compute attention between Q^(p) and current KV_block
3. **Accumulation**: Add partial results to output
4. **Communication**: 
   - Send KV_block to next device in ring
   - Receive KV_block from previous device

#### Mathematical Formulation
```
output_p = 0
KV_block = (K_p, V_p)
for t in 0..P-1:
    src_idx = (p - t) mod P
    partial = Attention(Q_p, KV_block)
    output_p += partial
    send KV_block to next device
    receive KV_block from previous device
```

## Combined Method: Ring Attention + Sequence Parallelism

### Integration Strategy
- **Data placement**: Sequence parallelism defines X^(p) placement
- **Communication order**: Ring Attention defines KV exchange pattern
- **Memory efficiency**: No global all-gather required

### Pseudocode Implementation
```
# On each device p in parallel
Q_p, K_p, V_p = Project(X_p)
output_p = 0
KV_block = (K_p, V_p)

for t in range(P):
    src_idx = (p - t) % P
    partial = Attention(Q_p, KV_block)
    output_p += partial
    
    # Async communication
    send(KV_block, to=(p+1)%P)
    KV_block = recv(from=(p-1)%P)
```

## Communication Complexity Analysis

### Bandwidth Requirements
- **Peak bandwidth**: O(L/P·d_model) per stage
- **Total volume**: Same as naïve approach but distributed over P stages
- **Overlap benefit**: Computation overlaps with communication

### Memory Footprint
- **Activation memory**: Reduced from O(L·d_model) to O(L/P·d_model)
- **KV cache**: Each device only holds current KV_block
- **Intermediate tensors**: Partial attention results accumulated locally

## Implementation Details

### Communication Primitives
- **Backend**: NCCL send/recv primitives or MPI point-to-point
- **Synchronization**: Asynchronous communication with computation
- **Buffer management**: Double buffering for overlap

### Precision and Optimization
- **Mixed precision**: fp16 or bf16 for Q, K, V tensors
- **Fused kernels**: Projection + softmax + communication hooks
- **Kernel fusion**: Reduces launch overhead

### Scalability Factors
- **Sequence length**: Benefits increase with L > 16k tokens
- **Device count**: Performance improves with P
- **Memory pressure**: Reduces activation memory by P factor