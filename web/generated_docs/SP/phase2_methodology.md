# Phase 2: Methodology Extraction

## Notation and Problem Setup
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
  - B = batch size
  - L = sequence length
  - d_model = model's hidden size
- H = attention heads, each of dimension $d_h = d_{\text{model}} / H$
- P = number of distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$

## Core Methodology: Combined Ring Attention + Sequence Parallelism

###: Sequence Parallelism Implementation
- Sequence dimension L is split across P devices: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device $D_p$ stores only: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$  
- **Memory reduction**: Activation memory drops from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### Ring Attention Algorithm
**Topology**: Logical ring connecting P devices
**Stages**: P sequential stages (0 ≤ t < P)

**Process Flow**:
1. **Initialization**: Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$

2. **Ring Communication** (per stage t):
   - Device computes partial attention between local $Q^{(p)}$ and current $K^{(src)}, V^{(src)}$ block
   - Source index calculation: $\text{src} \leftarrow (p - t) \bmod P$
   - Accumulate partial attention results over stages
   - Pass $K, V$ tensors to next device in ring

3. **Aggregation**: After P stages, each device has computed attention outputs for local queries using all keys/values

### Combined Implementation Details

**Pseudocode**:
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

### Communication Complexity Analysis
- **Naïve All-Gather**: $\mathcal{O}(L d_{\text{model}})$ per step
- **Ring Attention**: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage, P stages total
- **Peak bandwidth benefit**: Lower peak bandwidth due to sequential communication pattern
- **Overlap benefit**: Computation overlaps with asynchronous communication

### Implementation Parameters
- **Topology**: NCCL `send/recv` primitives or MPI point-to-point
- **Overlap**: Computation of one block overlaps with communication of next KV block
- **Precision**: Mixed-precision (fp16/bf16) for Q, K, V tensors
- **Fused kernels**: Projection and softmax fused with communication hooks
- **Scalability**: Benefits grow with L and P, especially for L > 16k tokens