# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision floating point)
- **Batch configuration**: 
  - 1024 sequences per batch
  - 10,000 tokens per sequence
  - Total: 10,240,000 tokens per batch
- **Token dimension**: 8192
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 8192
- **MLP hidden size**: 32768

### Hardware Environment
- **GPU**: H100 (inference-only setting)
- **Total GPUs**: 16 H100 GPUs
- **Network**: High-performance computing (HPC) cluster environment

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## Parallel Deployment Details

### Baseline Configuration (TP=8, PP=2)
- **Parallelism strategy**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **GPUs used**: 16 H100 GPUs
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Pipeline stages: 2 stages total, each spanning 8 GPUs
  - Expert placement: Multiple experts colocated on each GPU (typically 8 experts per layer per GPU)
  - Resource sharing: GPUs shared among multiple experts
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Contention**: Intra-GPU contention due to multiple experts sharing compute resources

### Proposed Cross-Node Expert Parallelism
- **Parallelism strategy**: Expert Parallelism (EP=16)
- **GPUs used**: 16 H100 GPUs
- **Per-GPU allocation**:
  - Each GPU hosts exactly one expert per layer
  - Total: 16 experts per layer × 4 layers = 64 expert instances
  - Expert distribution: One expert per GPU across all layers
- **Routing mechanism**:
  - Dynamic routing of input tokens to GPU holding corresponding expert
  - Asynchronous token batch transfer
  - Minimal idle time through overlapping communication and computation
- **Resource utilization**: All 16 experts per layer compute in parallel

## Performance Results

### Quantitative Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

### Performance Analysis
- **Throughput improvement**: 3.75× increase (450,000 vs 120,000 TPS)
- **Latency reduction**: 3.8× decrease (2.2 vs 8.3 ms TPOT)
- **Scaling efficiency**: Near-linear scaling achieved in large EP regime (EP=16)
- **Resource utilization**: Full GPU utilization through one-expert-per-GPU principle

### Bottleneck Analysis
#### Baseline Limitations
- **Intra-GPU contention**: Multiple experts sharing GPU compute resources
- **Pipeline stalls**: Sequential processing through pipeline stages
- **Resource underutilization**: GPUs not fully utilized due to sharing

#### Proposed Method Advantages
- **Expert isolation**: Each expert runs independently on dedicated GPU
- **Parallel processing**: All 16 experts per layer compute simultaneously
- **Communication overlap**: Asynchronous token routing minimizes waiting time
- **Load balancing**: Dynamic routing prevents expert overloading

## Experimental Validation

### Test Conditions
- **Inference-only setting**: No training overhead considered
- **Fixed batch size**: 1024 sequences × 10,000 tokens = 10.24M tokens
- **Stable workload**: Consistent token routing patterns
- **Network environment**: High-bandwidth, low-latency HPC cluster

### Reproducibility Factors
- **Hardware specification**: H100 GPU cluster
- **Model parameters**: Fixed dimensions and precision
- **Network topology**: Consistent cross-node communication patterns
- **Measurement methodology**: Standardized TPS and TPOT calculations

## Discussion Points

### Scalability Implications
- **Large EP regime**: EP ≥ 16 enables near-linear scaling
- **Resource requirements**: Requires sufficient GPUs (≥16 for tested configuration)
- **Network dependency**: Relies on high-performance interconnects
- **Memory efficiency**: Each GPU memory dedicated to single expert

### Practical Considerations
- **Deployment complexity**: Requires careful expert placement and routing
- **Network optimization**: Topology-aware placement critical for performance
- **Load monitoring**: Dynamic load balancing prevents performance degradation
- **Future extensibility**: Method applicable to training scenarios and larger models