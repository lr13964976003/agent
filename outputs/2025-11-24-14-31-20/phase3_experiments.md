# Phase 3: Experiments Extraction

## Experimental Setup

### 1. Model Configuration
- **Architecture**: 16-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: BF16 (reduced precision for efficiency)
- **Batch size**: 128 sequences
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 4096
- **Multi-head attention**:
  - Number of heads: 32
  - Dimension per head: 128
  - Total MHA dimension: 4096 (32 × 128)
- **MLP hidden size**: 16,384

### 2. Experimental Environment
- **GPU**: H100 (inference-only setting)
- **Total GPUs**: 16 H100 GPUs for both configurations
- **Task**: Inference-only (no training)
- **Environment**: High-performance computing cluster

### 3. Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## Parallel Deployment Configurations

### 3.1 Baseline Deployment (TP=8, PP=2)
- **Configuration**: Tensor Parallelism 8, Pipeline Parallelism 2
- **GPU allocation**: 16 H100 GPUs total
- **Per-GPU allocation**:
  - 1/8 tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - **Experts per GPU**: 8 experts per layer per GPU (colocated)
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource contention**: Multiple experts per GPU share compute resources

### 3.2 Proposed Cross-Node Expert Parallelism
- **Configuration**: Large expert parallelism (EP=16)
- **GPU allocation**: 16 H100 GPUs (perfect match for 16 experts per layer)
- **Per-GPU allocation**: Exactly one expert per layer per GPU
- **Expert placement**: Each GPU hosts exactly one expert for each of the 16 layers
- **Routing mechanism**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch sending
  - Minimal GPU idle time through overlap

## Experimental Results

### 4.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|---------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts per layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert per layer per GPU | 450,000 | 2.2 |

### 4.2 Performance Analysis
- **Throughput improvement**: 450,000 / 120,000 = 3.75× higher
- **Latency reduction**: 8.3 / 2.2 = 3.77× lower latency
- **Rounding**: Paper states "~3.75× higher throughput" and "~3.8× lower latency"
- **Efficiency**: Full GPU utilization through single-expert-per-GPU deployment

### 4.3 Bottleneck Analysis
- **Baseline limitations**:
  - Intra-GPU contention from multiple experts
  - Pipeline stalls due to sequential processing
  - Shared compute resources limiting parallel execution

- **Proposed method advantages**:
  - All 16 experts per layer compute in parallel
  - No intra-GPU expert contention
  - Maximal expert-level parallelism achieved
  - Communication effectively overlapped with computation

## Experimental Validation

### 5.1 Scalability Confirmation
- **Large EP regime**: EP ≥ 16 successfully validated
- **Network bandwidth**: Sufficient to sustain communication overhead
- **Near-linear scaling**: Achieved with 16 GPUs
- **HPC environment**: Effective in high-performance computing clusters

### 5.2 Deployment Validation
- **Topology-aware placement**: Successfully minimized network hotspots
- **Asynchronous routing**: Effectively overlapped communication and computation
- **Load balancing**: Maintained balanced expert utilization
- **Memory efficiency**: No memory constraints reported with BF16 precision

## Key Findings

### 6.1 Primary Results
1. **3.75× throughput improvement** achieved through single-expert-per-GPU deployment
2. **3.8× latency reduction** via maximized expert parallelism
3. **Full GPU utilization** without expert contention
4. **Scalable communication overlap** in large EP regime

### 6.2 Critical Success Factors
1. **Sufficient GPU count**: 16 GPUs for 16 experts per layer
2. **High-bandwidth network**: Modern interconnects (NVLink, InfiniBand)
3. **Asynchronous communication**: NCCL/MPI implementation
4. **Topology-aware placement**: Minimizing network congestion

### 6.3 Practical Implications
- **HPC clusters**: Optimal for environments with abundant GPU resources
- **Inference workloads**: Particularly effective for inference-only scenarios
- **Model scaling**: Blueprint for future large-scale MoE deployments
- **Future work**: Extensibility to training scenarios and dynamic routing