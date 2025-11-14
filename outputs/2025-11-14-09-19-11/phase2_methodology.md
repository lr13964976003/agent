# Phase 2: Methodology Extraction

## Problem Setup and Notation

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
  - B: batch size
  - L: sequence length
  - d_model: model's hidden size

### MHA Structure
- H attention heads, each of dimension: d_h = d_model / H
- Attention computation per head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Projections: $Q = X W_Q, K = X W_K, V = X W_V$
- Weight matrices: $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setup
- P distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Method

### Data Partitioning
- Sequence dimension L split across P devices
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- Memory reduction: Activation memory drops by factor of P
- Each device processes L/P tokens instead of full sequence

### Communication Challenge
- Self-attention requires all keys K and values V across entire sequence
- Naïve approach requires all-gather operation: gathering all K and V tensors to every device
- Costly when L is large

## Ring Attention Method

### Ring Topology Structure
- Devices arranged in logical ring
- Partial K and V blocks passed in fixed order
- Sequential, peer-to-peer exchanges instead of all-to-all communication

### Algorithm Stages (P stages total)

#### Stage 1: Initialization
- Each device computes local projections:
  - $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

#### Stage 2: Ring Communication (for t = 0 to P-1)
- Each device computes partial attention between:
  - Its local $Q^{(p)}$
  - Current $K^{(\text{src})}, V^{(\text{src})}$ it holds
- KV tensors passed to next device in ring
- Accumulate partial attention results over stages

#### Stage 3: Final Aggregation
- After P stages, each device has computed attention outputs for its local queries using all keys and values across the sequence

### Ring Attention Pseudocode
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device in ring
        receive KV_block from previous device
```

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- Sequence parallelism: Defines data placement (each device stores L/P sequence slice)
- Ring attention: Defines communication order (sequential peer-to-peer instead of all-gather)

### Communication Pattern
- Each device sends/receives one block per stage
- Total communication volume same as all-gather but with lower peak bandwidth
- Better overlap between communication and computation

## Communication Complexity Analysis

### Naïve All-Gather
- Each device exchanges: $\mathcal{O}(L d_{\text{model}})$ per step
- Peak bandwidth: High due to simultaneous all-to-all communication

### Ring Attention
- Each device exchanges: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- P stages total, same total volume but lower peak bandwidth
- Sequential nature allows better overlap with computation

### Memory Cost Comparison
- Without sequence parallelism: $\mathcal{O}(L d_{\text{model}})$ per device
- With sequence parallelism: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per device

## Implementation Details

### Technical Specifications
- **Communication Primitives**: NCCL send/recv or MPI point-to-point operations
- **Precision**: Mixed-precision (fp16 or bf16) for Q, K, V to reduce bandwidth
- **Kernel Optimization**: Fused kernels for projection and softmax with communication hooks
- **Overlap Strategy**: Computation of attention for one block overlaps with asynchronous communication of next KV block

### Scalability Characteristics
- Performance benefits increase with:
  - Sequence length L (especially L > 16k tokens)
  - Number of devices P
- Particularly effective for memory-constrained and bandwidth-limited environments

### Hardware Requirements
- GPU clusters with high-speed interconnects (NVLink, NVSwitch)
- 16×H100 GPUs tested in inference-only setting
- Sufficient network bandwidth for peer-to-peer communication