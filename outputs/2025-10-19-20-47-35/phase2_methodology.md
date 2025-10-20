# Phase 2: Methodology Extraction

## Notation and Problem Setup
- Input: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where B=batch size, L=sequence length, d_model=hidden size
- MHA has H attention heads, each of dimension $d_h = d_{\text{model}} / H$
- P distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation
- Sequence dimension L split across P devices: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device stores $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- Reduces activation memory by factor of P from $\mathcal{O}(L \cdot d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$

## Ring Attention Algorithm
Structured as P stages:

### Stage 1: Initialization
- Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

### Stage 2: Ring Communication (for t = 0 to P-1)
- Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
- Source device calculation: $\text{src} \leftarrow (p - t) \bmod P$
- Pass $K, V$ tensors to next device in ring after computation
- Accumulate partial attention results over stages

### Stage 3: Aggregation
- After P stages, each device has computed attention outputs for local queries using all keys and values across sequence

## Combined Ring Attention + Sequence Parallelism
### Data Placement
- Sequence parallelism defines data placement: each device stores slice of sequence
- Ring Attention defines communication order: send/receive one block per stage instead of all-gather

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
- Naïve All-Gather: Each device exchanges $\mathcal{O}(L \cdot d_{\text{model}})$ per step
- Ring Attention: Each device exchanges $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per stage, P stages total
- Same total communication volume but lower peak bandwidth and better overlap between communication/computation

## Implementation Details
- Topology: NCCL's `send/recv` primitives or MPI point-to-point operations
- Overlap: Attention computation for one block overlaps with asynchronous communication of next $K, V$ block
- Precision: Mixed-precision (fp16 or bf16) for $Q, K, V$ tensors to reduce bandwidth
- Fused kernels: Projection and softmax fused with communication hooks to reduce kernel launch overhead
- Scalability: Performance benefits grow with L and P, especially for L > 16k tokens

## Attention Computation Details
For single head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
Where: $Q = X W_Q, K = X W_K, V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$