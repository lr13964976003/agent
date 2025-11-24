# Stage 2: Methodology Extraction - Ring Attention with Sequence Parallelism

## Mathematical Notation and Setup

### Input Representation
- **Input Sequence**: X ∈ ℝ^(B×L×d_model)
  - B: batch size (fixed at 128 in experiments)
  - L: sequence length (fixed at 100,000 tokens)
  - d_model: model's hidden size

### Multi-Head Attention Structure
- **Attention Heads**: H = 32 heads (fixed in experiments)
- **Head Dimension**: d_h = d_model/H = 128 (fixed in experiments)
- **Attention Computation**: Attn(Q,K,V) = softmax(QK^T/√d_h)V
- **Projection Matrices**: W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

### Distributed Setup
- **Devices**: P distributed devices {D_0, D_1, ..., D_{P-1}}
- **Objective**: Compute MHA with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Method

### Data Partitioning
- **Sequence Split**: X = [X^(0), X^(1), ..., X^(P-1)]
- **Per Device Storage**: X^(p) ∈ ℝ^(B×L/P×d_model)
- **Memory Reduction**: Activation memory reduced by factor P

### Communication Challenge
- **Required Access**: Each device needs all keys K and values V across entire sequence
- **Naive Approach**: All-gather operation requiring O(L·d_model) communication per device
- **Problem**: Costly when L is large (100k tokens)

## Ring Attention Method

### Ring Topology
- **Device Arrangement**: Logical ring D_0 → D_1 → ... → D_{P-1} → D_0
- **Communication Pattern**: Sequential peer-to-peer exchanges

### Algorithm Stages (P stages for P devices)

#### Stage 1: Initialization
- Each device computes local projections:
  - Q^(p), K^(p), V^(p) = Project(X^(p))
  - All computed from local X^(p) ∈ ℝ^(B×L/P×d_model)

#### Stage 2: Ring Communication
For each stage t (0 ≤ t < P):
- **Source Index**: src = (p - t) mod P
- **Computation**: Compute partial attention between local Q^(p) and current K^(src), V^(src)
- **Communication**: Pass K,V tensors to next device in ring
- **Accumulation**: Accumulate partial attention results over stages

#### Stage 3: Aggregation
- After P stages: Each device has complete attention output for local queries
- **Output**: Complete attention computed using all keys/values across sequence

### Communication Complexity Analysis
- **Naive All-Gather**: O(L·d_model) per step per device
- **Ring Attention**: O(L/P·d_model) per stage × P stages = same total volume
- **Peak Bandwidth**: Reduced due to sequential communication pattern
- **Overlap**: Computation and communication overlap possible

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
1. **Data Placement**: Sequence parallelism defines how input is split
2. **Communication Order**: Ring attention defines communication pattern
3. **Memory Efficiency**: Both techniques complement each other

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
        
        # Async communication
        send KV_block to next device
        receive KV_block from previous device
```

## Implementation Details

### Hardware and Framework
- **Communication**: NCCL's send/recv primitives or MPI point-to-point
- **Precision**: Mixed-precision (fp16 or bf16) for Q,K,V tensors
- **Overlapping**: Async communication during computation
- **Kernel Fusion**: Fused projection and softmax operations

### Memory Specifications
- **Activation Memory**: Reduced from O(L·d_model) to O(L/P·d_model)
- **Model Parameters**: Each device stores full parameter set (data parallelism separate)
- **Intermediate Buffers**: Ring buffer for K,V blocks between stages

### Scalability Parameters
- **Benefits**: Grow with L (sequence length) and P (number of devices)
- **Threshold**: Particularly effective for L > 16k tokens
- **Hardware**: NVLink and NVSwitch interconnects assumed

## Key Parameters Summary
- **Batch Size**: B = 128 (fixed)
- **Sequence Length**: L = 100,000 tokens (fixed)
- **Attention Heads**: H = 32 (fixed)
- **Head Dimension**: d_h = 128 (fixed)
- **MLP Hidden Size**: 16,384 (fixed)
- **Model Layers**: 16 layers (dense transformer)
- **Precision**: BF16
- **Devices**: 16×H100 GPUs in experiments