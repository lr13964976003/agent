# Phase 3: Experiments Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Experimental Setup Details

### 1. Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision floating point)

### 2. Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8192 dimensions per token
- **Multi-head attention**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32,768 dimensions

### 3. Hardware Configuration
- **GPU**: 16 × H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)

### 4. Evaluation Metrics
- **TPS (Tokens per Second)**: Primary throughput metric
- **TPOT (Time per Output Token)**: Primary latency metric

## Parallel Deployment Configurations

### 4.1 Baseline Configuration (TP=8, PP=2)
- **Parallelism strategy**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **GPU allocation**: 16 GPUs total
- **Per-GPU deployment**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts are colocated: 8 experts per layer per GPU
- **Processing flow**: Sequential token flow through pipeline stages
- **Resource sharing**: Multiple experts per GPU share compute resources

### 4.2 Proposed Cross-Node Expert Parallelism
- **Parallelism strategy**: Expert Parallelism (EP=16)
- **GPU allocation**: 16 GPUs total
- **Per-GPU deployment**: Each GPU hosts exactly one expert per layer
- **Expert distribution**: 16 experts per layer → 16 GPUs (one-to-one mapping)
- **Routing mechanism**: Dynamic token routing to GPU holding corresponding expert
- **Communication**: Asynchronous token batch transfer with minimal idle time

## Experimental Results

### 5.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts/layer/GPU + TP shard | 120,000 | 8.3 |
| Proposed Cross-Node EP | 16 | 1 expert/layer/GPU | 450,000 | 2.2 |

### 5.2 Performance Improvements
- **Throughput improvement**: 3.75× (450,000 vs 120,000 TPS)
- **Latency reduction**: 3.8× (2.2ms vs 8.3ms TPOT)
- **GPU utilization**: Full utilization with dedicated expert per GPU
- **Expert concurrency**: All 16 experts compute in parallel per layer

### 5.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU contention from multiple experts
  - Pipeline stalls due to sequential processing
  - Shared compute resources causing delays

- **Proposed solution advantages**:
  - Dedicated GPU resources per expert
  - Parallel expert computation
  - Asynchronous communication overlapping computation

## Scalability Validation

### 6.1 Large EP Regime Performance
- **EP scale**: 16 (meets large EP definition)
- **Scaling behavior**: Near-linear scaling demonstrated
- **Network efficiency**: High-bandwidth interconnects effectively utilized
- **Load balancing**: Dynamic routing prevents expert overloading

### 6.2 Resource Utilization
- **GPU compute**: 100% utilization per expert
- **Memory usage**: Balanced across all 16 GPUs
- **Network bandwidth**: Optimized through token batching
- **Communication overhead**: Amortized across large batch sizes

## Experimental Validity

### 7.1 Controlled Variables
- **Identical model**: Same 4-layer, 16-expert MoE architecture
- **Identical hardware**: 16 H100 GPUs for both configurations
- **Identical input**: 1024 sequences × 10,000 tokens
- **Identical precision**: FP16 throughout

### 7.2 Isolated Testing
- **Inference-only**: Focused on inference performance
- **No training variables**: Eliminates training-related confounders
- **Pure parallelism comparison**: TP/PP vs EP strategies

### 7.3 Reproducibility Factors
- **Clear configuration**: Explicit TP=8, PP=2 vs EP=16 settings
- **Quantitative metrics**: TPS and TPOT measurements
- **Hardware specification**: H100 GPU cluster details provided