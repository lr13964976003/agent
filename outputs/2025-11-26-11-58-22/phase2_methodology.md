# Phase 2: Methodology Extraction

## Complete Model Architecture

### Layer Structure
- **Total layers**: 61
- **Dense layers**: 3 (layers 0-2)
- **MoE layers**: 58 (layers 3-60)
- **Experts per MoE layer**: 64
- **Total experts**: 3,712 (58 × 64)

### Dimensional Specifications
```
Token dimension: 7168
MHA heads: 128
Head dimension: 128
MLP hidden size: 2048
Precision: BF16
Model type: Transformer with MoE layers
```

## Expert Parallelism Methodology

### 1. Expert Placement Algorithm

#### Single-Expert-Per-GPU Principle
```
For each MoE layer l in [3, 60]:
    For each expert e in [0, 63]:
        Assign expert(l, e) to GPU g where g = (l-3) × 64 + e
        Ensure GPU g is on node n where n = ⌊g/8⌋
```

#### GPU-to-Expert Mapping (Complete)
- **Total GPUs**: 3,904 (488 nodes × 8 GPUs/node)
- **GPUs used**: 3,715 (3 dense layers + 3,712 experts)
- **Unused GPUs**: 189 (can be used for redundancy or expansion)

### 2. Cross-Node Distribution Formula

#### Node Assignment Calculation
```
layer_index: l (3 ≤ l ≤ 60)
expert_index: e (0 ≤ e ≤ 63)
global_gpu_id: g = (l-3) × 64 + e
node_id: n = ⌊g/8⌋
gpu_within_node: local_g = g mod 8
```

#### Complete Device Mapping Example
```
Layer 3: GPUs 0-63 (Nodes 0-7)
    Expert 0 → GPU 0 (Node 0, GPU 0)
    Expert 1 → GPU 1 (Node 0, GPU 1)
    ...
    Expert 63 → GPU 63 (Node 7, GPU 7)

Layer 4: GPUs 64-127 (Nodes 8-15)
    Expert 0 → GPU 64 (Node 8, GPU 0)
    Expert 1 → GPU 65 (Node 8, GPU 1)
    ...
    Expert 63 → GPU 127 (Node 15, GPU 7)

... continues for all 58 MoE layers ...

Layer 60: GPUs 3648-3711 (Nodes 456-463)
    Expert 0 → GPU 3648 (Node 456, GPU 0)
    Expert 1 → GPU 3649 (Node 456, GPU 1)
    ...
    Expert 63 → GPU 3711 (Node 463, GPU 7)
```

### 3. Dense Layer Allocation

#### Dense Layers 0-2
- **Placement**: GPUs 3712-3714 respectively
- **Nodes**: Node 464 (GPUs 0,1,2)
- **Replication**: Not needed for inference-only setting

### 4. Communication Strategy

#### Token Routing Algorithm
```
For each token t in batch:
    1. Compute gating scores for all 64 experts in target layer
    2. Select top-k experts (k=2 in standard MoE)
    3. Route token to GPU hosting each selected expert
    4. Asynchronous send/recv operations
    5. Expert computation on destination GPU
    6. Return results to source GPU
```

#### Bandwidth Optimization
- **Token batching**: Group tokens by destination GPU
- **Message size**: Batch size × 7168 × 2 bytes (BF16)
- **Peak bandwidth**: 1.8TBps × 80% utilization = 1.44TBps
- **Network topology**: NVLink within node, InfiniBand between nodes

### 5. Memory Requirements

#### Per-GPU Memory Usage
```
Expert parameters: 7168 × 2048 × 2 (weights) + 2048 × 7168 × 2 (weights)
                 = 29.36 MB per expert
Activation memory: batch_size × 7168 × precision (BF16)
Total per GPU: ~30MB + activations
```

#### Memory Sufficiency Check
- **GPU memory**: 64GB per H100
- **Expert memory**: 30MB per expert
- **Available for activations**: ~63.97GB
- **Maximum batch**: Limited by activation memory, not expert parameters

### 6. Compute Utilization

#### FLOPS Calculation
- **Single GPU FLOPS**: 400TFlops
- **MFU**: 60% → 240TFlops effective
- **Expert computation**: 2×7168×2048×batch_size FLOPs
- **Peak throughput**: 240T / (2×7168×2048×10^-12) ≈ 8.2M tokens/sec per expert

#### Pipeline Efficiency
- **Communication overlap**: 95%+ compute utilization
- **Load balancing**: Dynamic gating prevents stragglers
- **Synchronization**: CUDA streams for async operations

## Integration with Other Parallel Strategies

### Tensor Parallelism (Optional)
- **Application**: For experts exceeding single-GPU memory
- **Split**: Column-row parallel for MLP layers
- **Granularity**: Within-expert parallelism (not used in this setup)

### Data Parallelism
- **Scope**: Entire model replicas
- **Usage**: Not needed for inference
- **Future**: Training scenario extension

### Pipeline Parallelism
- **Scope**: Layer-wise pipeline
- **Usage**: Not used (one forward pass per token)
- **Alternative**: Expert-level pipeline within layer