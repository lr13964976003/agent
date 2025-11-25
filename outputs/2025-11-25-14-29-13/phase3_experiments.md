# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 16-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 MLP-based experts
- **Precision**: BF16 (BFloat16)
- **Token dimension**: 4096
- **MLP hidden size**: 16384
- **Multi-head attention**: 32 heads, 128 dimensions per head

### Dataset Configuration
- **Batch size**: 128 sequences
- **Sequence length**: 10000 tokens per sequence
- **Total tokens per batch**: 1,280,000 tokens

### Environment
- **GPUs**: Adequate H100 GPUs
- **Setting**: Inference-only evaluation
- **Network**: High-performance computing cluster with cross-node connectivity

## Parallel Deployment Configurations

### Baseline Configuration (TP=8, PP=2)
- **Parallelism strategy**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **GPU allocation**: 
  - Each GPU holds tensor-parallel shard for all layers
  - Multiple experts colocated on same GPU
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Expert sharing**: Multiple experts per GPU share compute resources

### Proposed Cross-Node Expert Parallelism
- **Parallelism strategy**: Expert Parallelism (EP=16) + Topology-aware placement
- **GPU allocation**: 
  - Each GPU hosts exactly one expert per layer
  - 16 GPUs total (1 expert/GPU/layer)
- **Routing mechanism**: 
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously
- **Communication**: Cross-node transfers with computation overlap

## Performance Results

### Throughput Comparison
| Configuration | TPS (Tokens/second) | Relative Improvement |
|---------------|-------------------|---------------------|
| Baseline (TP=8, PP=2) | 120,000 | 1.0× |
| Proposed (EP=16) | 450,000 | 3.75× |

### Latency Comparison
| Configuration | TPOT (ms/token) | Relative Improvement |
|---------------|----------------|---------------------|
| Baseline (TP=8, PP=2) | 8.3 | 1.0× |
| Proposed (EP=16) | 2.2 | 3.77× |

### Resource Utilization
- **Baseline**: Intra-GPU contention due to shared expert resources
- **Proposed**: Full GPU utilization with dedicated expert per device
- **Network efficiency**: Asynchronous routing minimizes idle time

## Key Performance Factors

### Throughput Drivers
1. **Expert-level parallelism**: All 16 experts compute simultaneously per layer
2. **No GPU contention**: One expert per GPU eliminates resource sharing
3. **Asynchronous communication**: Token routing overlaps with computation
4. **Pipeline efficiency**: Fine-grained scheduling reduces idle time

### Latency Reduction Factors
1. **Parallel expert processing**: 16× reduction potential vs sequential processing
2. **Communication overlap**: Network transfers hidden by computation
3. **Load balancing**: Preventing expert hotspots reduces stragglers
4. **Topology-aware placement**: Minimizes cross-node communication distance

## Scalability Analysis

### Linear Scaling Evidence
- With 16 GPUs achieving EP=16, system demonstrates near-linear scaling
- Network bandwidth becomes primary limiting factor (not compute)
- Communication costs effectively amortized across large token batches

### Bottleneck Analysis
- **Primary bottleneck**: Network bandwidth in large EP regime
- **Mitigation strategy**: Topology-aware routing and token batching
- **Secondary consideration**: Load balancing to prevent expert overloading

## Experiment Validation

### Hypothesis Testing
- **H1**: One expert per GPU maximizes compute concurrency → **Confirmed** (3.75× throughput)
- **H2**: Communication overlap mitigates cross-node latency → **Confirmed** (3.77× latency reduction)
- **H3**: Large EP (≥16) enables better scalability → **Confirmed** (linear scaling observed)

### Reproducibility Factors
- Fixed model architecture (16 layers, 16 experts/layer)
- Consistent batch configuration (128×10000 tokens)
- Standardized BF16 precision
- Clear baseline comparison (TP=8, PP=2 vs EP=16)

## Performance Implications

### Real-world Impact
- **3.75× throughput improvement** enables larger model deployment
- **3.77× latency reduction** supports real-time applications
- **Scalable architecture** adapts to available GPU resources
- **HPC optimization** leverages modern cluster infrastructure

### Future Scalability
- Method proven effective for EP ≥ 16
- Architecture supports extension to thousands of experts
- Framework integrates with existing TP/DP parallelisms
- Foundation for training scenario extensions