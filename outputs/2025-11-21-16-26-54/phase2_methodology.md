# Phase Two: Methodology Extraction

## Mathematical Notation and Problem Setup

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- B: batch size
- L: sequence length
- d_model: model's hidden size

### MHA Structure
- H attention heads
- Each head dimension: $d_h = d_{\text{model}} / H$
- Single head attention: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Projections: $Q = X W_Q, K = X W_K, V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### System Setup
- P distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Details

### Data Partitioning
- Sequence dimension L split across devices: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- **Memory benefit**: Activation memory reduced by factor of P

### Communication Challenge
- Self-attention requires all keys K and values V across entire sequence
- Naïve approach would require all-gather operation: costly when L is large

## Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Sequential peer-to-peer exchanges instead of all-to-all
- Reduces peak communication bandwidth requirements

### Algorithm Stages (P stages total)

#### Stage 1: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

#### Stage 2: Ring Communication (for t = 0 to P-1)
Each stage t:
1. Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
2. Pass K,V tensors to next device in ring
3. Source index calculation: $\text{src} \leftarrow (p - t) \bmod P$
4. Accumulate partial attention results over stages

#### Stage 3: Aggregation
After P stages, each device has computed attention outputs for local queries using all keys and values across sequence

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence parallelism**: Defines data placement (each device stores sequence slice)
- **Ring Attention**: Defines communication order (sequential block transfers instead of all-gather)

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
        send KV_block to next device in ring
        receive KV_block from previous device
```

## Communication Complexity Analysis

### Naïve All-Gather Approach
- Each device exchanges: $\mathcal{O}(L d_{\text{model}})$ per step
- Peak bandwidth requirement: High

### Ring Attention Approach
- Each device exchanges: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- P stages total → same total volume but **lower peak bandwidth**
- Better overlap between communication and computation

### Memory Cost Comparison
- Without sequence parallelism: $\mathcal{O}(L d_{\text{model}})$ per device
- With sequence parallelism: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per device

## Implementation Details

### Technical Specifications
- **Communication**: NCCL's `send/recv` primitives or MPI point-to-point operations
- **Overlap**: Computation overlaps with asynchronous communication of next K,V block
- **Precision**: Mixed-precision (fp16 or bf16) for Q,K,V to reduce bandwidth
- **Fused Kernels**: Projection and softmax fused with communication hooks
- **Scalability**: Benefits grow with L and P, especially for L > 16k tokens

### Performance Characteristics
- **Best suited**: Large-scale, memory-constrained, bandwidth-limited environments
- **Scaling factors**: Sequence length L and number of devices P
- **Threshold**: Particularly effective for sequences >16k tokens