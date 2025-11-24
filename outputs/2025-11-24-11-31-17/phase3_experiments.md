# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Model Configuration
- **Architecture**: 16-layer Mixture-of-Experts (MoE) transformer
- **Experts per Layer**: 16 experts (MLP-based)
- **Precision**: BF16 (BFloat16)
- **Token Dimension**: 4096
- **Hidden Dimension**: 16384 (MLP hidden size)
- **Multi-Head Attention**: 
  - Number of heads: 32
  - Head dimension: 128
  - Total MHA dimension: 4096

### Runtime Configuration
- **Batch Size**: 128 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 1,280,000 tokens
- **Setting**: Inference-only (no training)
- **Hardware**: H100 GPUs

### Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token in milliseconds

## 2. Baseline Deployment (TP=8, PP=2)

### Configuration Details
- **GPUs Used**: 16 H100 GPUs
- **Parallel Strategy**: 
  - Tensor Parallelism (TP) = 8
  - Pipeline Parallelism (PP) = 2
  - Expert Parallelism (EP) = Not explicitly large (experts colocated)

### GPU Allocation Breakdown
```
Pipeline Stage 1 (8 GPUs):
- GPUs: 1-8
- Each GPU: 1/8 tensor shard + 8 experts per layer
- Layers: 1-8

Pipeline Stage 2 (8 GPUs):
- GPUs: 9-16  
- Each GPU: 1/8 tensor shard + 8 experts per layer
- Layers: 9-16

Total:
- 16 experts per layer distributed as 2 experts per GPU
- Pipeline stages process sequentially
```

### Performance Results - Baseline
- **TPS**: 120,000 tokens/second
- **TPOT**: 8.3 milliseconds

### Limitations Identified
- **Intra-GPU Contention**: Multiple experts share GPU compute resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Underutilization**: Experts compete for GPU memory and compute

## 3. Proposed Cross-Node Expert Parallelism

### Configuration Details
- **GPUs Used**: 16 H100 GPUs
- **Parallel Strategy**: 
  - Expert Parallelism (EP) = 16 (Large EP)
  - Tensor Parallelism (TP) = 1 (within expert)
  - Pipeline Parallelism (PP) = 1 (across layers)

### GPU Allocation Breakdown
```
Per-Layer Configuration:
- Expert 1 → GPU 1
- Expert 2 → GPU 2
- ...
- Expert 16 → GPU 16

Layer-wise Scaling:
- Layer 1: 16 experts on GPUs 1-16 (parallel)
- Layer 2: 16 experts on GPUs 1-16 (parallel)
- ...
- Layer 16: 16 experts on GPUs 1-16 (parallel)

Key: Each GPU hosts one expert per layer, rotating across layers
```

### Routing Mechanism
- **Token Distribution**: Dynamic routing based on gating scores
- **Cross-Node Transfer**: Asynchronous token batches
- **Load Balancing**: Real-time expert utilization monitoring

### Performance Results - Proposed
- **TPS**: 450,000 tokens/second
- **TPOT**: 2.2 milliseconds

## 4. Comparative Analysis

### Performance Improvements
| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| TPS | 120,000 | 450,000 | 3.75× |
| TPOT | 8.3 ms | 2.2 ms | 3.77× |

### Resource Utilization
- **Baseline**: 16 GPUs shared among multiple experts
- **Proposed**: 16 GPUs dedicated, one expert per GPU
- **Efficiency Gain**: Elimination of intra-GPU contention

### Scalability Analysis
- **Linear Scaling**: Near-linear scaling demonstrated for EP ≥ 16
- **Network Limitation**: Bandwidth becomes primary bottleneck
- **Break-even Point**: Achieved with 16+ experts and adequate network bandwidth

## 5. Experimental Validation

### Verification of Core Claims
1. **Expert Independence**: Verified with one expert per GPU
2. **Communication Overlap**: Demonstrated via asynchronous token routing
3. **Load Balancing**: Dynamic adjustment prevents expert overload
4. **Scalability**: EP=16 shows linear scaling characteristics

### Limitations Noted
- **Inference Only**: Results apply to inference, training not tested
- **Fixed Topology**: Experiments on homogeneous H100 cluster
- **Network Assumption**: High-bandwidth interconnects (NVLink/NVSwitch)

## 6. Experimental Conclusion

The experiments validate that large-scale cross-node expert parallelism with EP ≥ 16 achieves:
- Significant throughput improvements (3.75×)
- Substantial latency reduction (3.77×)
- Effective resource utilization through expert independence
- Practical scalability for large MoE deployments

The results support the paper's core thesis that maximizing expert parallelism through one-expert-per-GPU deployment is superior to traditional colocation approaches when sufficient network bandwidth is available.