# Phase 2: Methodology Extraction

## Methodology Overview

### 1. Problem Setup and Notation

**Input Representation**:
- Input sequence: X ∈ ℝ^(B×L×d_model)
- B: batch size
- L: sequence length (100,000 tokens in experiments)
- d_model: model's hidden size

**Multi-Head Attention Structure**:
- H attention heads
- Each head dimension: d_h = d_model/H
- Weight matrices: W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)
- Single head attention: Attn(Q,K,V) = softmax(QK^T/√d_h)V

**Distributed Setup**:
- P distributed devices: {D_0, D_1, ..., D_{P-1}}
- Objective: Compute MHA with minimal communication overhead and reduced memory footprint

### 2. Sequence Parallelism Implementation

**Data Partitioning**:
- Sequence dimension L split across P devices
- Each device D_p stores: X^(p) ∈ ℝ^(B×(L/P)×d_model)
- Memory reduction: Activation memory per device drops from O(L·d_model) to O((L/P)·d_model)

**Challenge Identified**:
- Self-attention requires all keys K and values V across entire sequence
- Naive approach would require all-gather operation of O(L·d_model) tensors to every device

### 3. Ring Attention Algorithm

**Topology Structure**:
- Devices arranged in logical ring topology
- Sequential peer-to-peer exchanges replace global communication

**Algorithm Stages (P stages total)**:

**Stage 0 - Initialization**:
- Each device computes local projections:
  - Q^(p) = X^(p)W_Q
  - K^(p) = X^(p)W_K  
  - V^(p) = X^(p)W_V

**Stages 0 to P-1 - Ring Communication**:
- At stage t (0 ≤ t < P):
  1. Each device computes partial attention:
     - Attention(Q^(p), K^(src), V^(src)) where src = (p-t) mod P
  2. Accumulate partial attention results
  3. Pass KV_block = (K^(src), V^(src)) to next device in ring
  4. Receive new KV_block from previous device

**Final Aggregation**:
- After P stages, each device has computed attention outputs for its local queries using all keys and values across the sequence

### 4. Combined Ring Attention + Sequence Parallelism

**Integration Strategy**:
- Sequence parallelism defines data placement (how sequence is split)
- Ring attention defines communication pattern (how KV blocks are shared)

**Pseudocode Implementation**:
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)  # Local projections
    output_p = 0                  # Initialize output accumulator
    KV_block = (K_p, V_p)         # Start with local KV
    
    for t in 0..P-1:
        src_idx = (p - t) mod P   # Determine source device
        partial = Attention(Q_p, KV_block)  # Compute partial attention
        output_p += partial       # Accumulate results
        send KV_block to next device in ring
        receive KV_block from previous device

    return output_p
```

### 5. Communication Complexity Analysis

**Naive All-Gather Approach**:
- Each device exchanges O(L·d_model) per step
- High peak bandwidth requirement
- Synchronization bottleneck

**Ring Attention Approach**:
- Each device exchanges O((L/P)·d_model) per stage
- P stages total, same total volume but lower peak bandwidth
- Better overlap between communication and computation
- Scales efficiently with P

### 6. Implementation Details

**Hardware Interface**:
- Uses NCCL's send/recv primitives or MPI point-to-point operations
- Takes advantage of high-bandwidth interconnects (NVLink/NVSwitch)

**Performance Optimizations**:
- **Overlap**: Computation for one block overlaps with async communication of next KV block
- **Precision**: Mixed-precision (FP16/BF16) for Q,K,V to reduce bandwidth requirements
- **Fused Kernels**: Projection and softmax operations fused with communication hooks
- **Scalability**: Benefits grow with sequence length L and device count P

**Memory Management**:
- Activation memory reduced by factor of P through sequence parallelism
- KV blocks streamed through ring, avoiding full-sequence storage on each device
- Intermediate results accumulated incrementally to avoid large temporary buffers