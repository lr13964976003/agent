# Phase 2: Methodology Extraction

## Ring Attention Algorithm

### Mathematical Formulation

For P distributed devices {D₀, D₁, ..., Dₚ₋₁}, input sequence split as:
```
X = [X⁽⁰⁾, X⁽¹⁾, ..., X⁽ᴾ⁻¹⁾]
X⁽ᵖ⁾ ∈ ℝ^(B × (L/P) × d_model)
```

### Ring Communication Stages
```
// Stage t (0 ≤ t < P)
src_idx = (p - t) mod P
partial_attention = Attention(Q_p, K_src, V_src)
KV_block = send_to_next(K, V)
KV_block = receive_from_prev(K, V)
```

### Pseudocode Implementation
```python
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    for t in range(P):
        src_idx = (p - t) % P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device in ring
        receive KV_block from previous device
```

## Communication Complexity Analysis

### Naïve All-Gather Approach
- Communication volume: Θ(L × d_model)
- Peak bandwidth requirement: High
- Memory footprint: O(L × d_model) per device

### Ring Attention Approach  
- Communication volume: Θ((L/P) × d_model) per stage
- Total stages: P
- Peak bandwidth: Reduced by factor P
- Memory footprint: O((L/P) × d_model) per device

## Implementation Details

### Hardware Specifications
- **GPUs**: 16×H100 with NVLink and NVSwitch
- **Network**: High-bandwidth interconnect
- **Memory**: HBM3 with 80GB per GPU
- **Precision**: BF16 (16-bit floating point)

### NCCL/MPI Primitives
- **send/recv**: NCCL point-to-point communication
- **asynchronous overlap**: Computation overlaps with communication
- **CUDA synchronization**: After each ring stage for consistency

### Fused Kernel Optimizations
- **Projection fusion**: Q, K, V projections combined into single kernel
- **Softmax fusion**: Attention computation fused with softmax
- **Communication hooks**: Integrated with NCCL for minimal overhead

### Mixed Precision Strategy
- **Computation**: BF16 for all matrix operations
- **Communication**: BF16 for K/V tensors
- **Accumulation**: FP32 for attention output accumulation
- **Master weights**: FP32 for parameter storage

## Model Architecture Parameters

### Dense Transformer Architecture
- **Layers**: 16 transformer layers
- **Embeddings**: d_model = 4096 (32 heads × 128 dim)
- **MLP**: ffn_hidden_size = 16,384
- **Sequence length**: L = 100,000 tokens
- **Batch size**: B = 128
- **Attention heads**: H = 32
- **Head dimension**: d_h = 128

### Baseline Configuration
- **Tensor Parallelism**: TP = 8 (splits model across 8 GPUs)
- **Pipeline Parallelism**: PP = 2 (splits layers across 2 stages)
- **Total devices**: 16 GPUs (8×2 arrangement)
- **No sequence parallelism**: Full sequence stored on each device

### RA+SP Configuration
- **Ring size**: 16 (all 16 GPUs in ring topology)
- **Sequence parallelism**: Sequence split across 16 devices
- **Memory reduction**: 16× reduction in activation memory
- **Communication pattern**: Ring-based K/V exchange