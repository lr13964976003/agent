# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts**: 16 experts per layer
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: BF16 (BFloat16)

### 1.2 Input Specifications
- **Batch Size**: 128 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 4,096 dimensions per token
- **MHA Configuration**: 
  - Number of heads: 32
  - Dimension per head: 128
- **MLP Hidden Size**: 32,768 dimensions

### 1.3 Hardware Setup
- **GPUs**: 16 × H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Setting**: Inference-only evaluation

### 1.4 Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (Traditional Approach)

#### Configuration Parameters
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs**: 16 H100

#### Per-GPU Allocation
- **Expert Distribution**: 8 experts per layer per GPU
- **Tensor Sharding**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Pipeline Stages**: 2 stages total, each spanning 8 GPUs
- **Expert Colocation**: Multiple experts share GPU resources

#### Processing Flow
- Tokens flow sequentially through pipeline stages
- Multiple experts per GPU share compute resources
- Intra-GPU contention between experts

### 2.2 Proposed Cross-Node Expert Parallelism

#### Configuration Parameters
- **Expert Parallelism (EP)**: 16 (large EP regime)
- **Tensor Parallelism (TP)**: 1 (per expert)
- **Pipeline Parallelism (PP)**: 1 (per layer)
- **Total GPUs**: 16 H100

#### Per-GPU Allocation
- **Expert Distribution**: Exactly 1 expert per layer per GPU
- **Memory Layout**: Each GPU hosts complete expert weights
- **No Colocation**: No expert sharing on GPUs
- **Full Parallelism**: All 16 experts per layer compute simultaneously

#### Routing Strategy
- **Dynamic Routing**: Input tokens routed to GPU holding target expert
- **Async Communication**: Token batches sent asynchronously
- **Network Optimization**: Topology-aware placement

## 3. Performance Results

### 3.1 Comparative Performance Table

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis

#### Throughput Improvement
- **Baseline TPS**: 120,000 tokens/second
- **Proposed TPS**: 450,000 tokens/second
- **Speedup**: 3.75× improvement

#### Latency Reduction
- **Baseline TPOT**: 8.3 ms per token
- **Proposed TPOT**: 2.2 ms per token
- **Reduction**: 3.8× lower latency

#### Resource Utilization
- **Baseline**: GPU sharing causes contention
- **Proposed**: Full GPU utilization per expert
- **Scaling**: Near-linear scaling in EP ≥ 16 regime

## 4. Experimental Validation

### 4.1 Scalability Confirmation
- **EP Regime**: Successfully demonstrated EP ≥ 16
- **Network Impact**: Communication overhead mitigated by overlap
- **Load Balance**: Dynamic routing prevents expert stragglers

### 4.2 Bottleneck Analysis
- **Baseline**: Compute contention due to expert colocation
- **Proposed**: Network communication as primary consideration
- **Trade-off**: Compute concurrency vs. communication overhead

### 4.3 Reproducibility Conditions
- **Hardware**: Identical H100 GPU configurations
- **Software**: Same BF16 precision throughout
- **Workload**: Fixed batch size (128×10,000 tokens)
- **Environment**: Controlled HPC cluster environment

## 5. Experimental Insights

### 5.1 Key Findings
- **Expert Isolation**: One expert per GPU eliminates intra-device contention
- **Communication Overlap**: Async routing effectively hides network latency
- **Parallel Scaling**: Large EP regime enables near-linear performance scaling

### 5.2 Practical Implications
- **Deployment**: Requires sufficient GPU resources (≥16 GPUs)
- **Network**: Demands high-bandwidth interconnects
- **Memory**: Benefits from large GPU memory per device
- **Future Scaling**: Extends to models with thousands of experts