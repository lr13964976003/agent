# Phase 3: Experiments Extraction

## Experimental Setup and Results

### 1. Experimental Configuration

#### 1.1 Model Architecture
- **Type**: 16-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: Multi-Layer Perceptron (MLP)
- **Precision**: BF16 (BFloat16)
- **Total Layers**: 16 MoE layers

#### 1.2 Input Specifications
- **Batch Size**: 128 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 4,096 (embedding dimension)
- **Multi-Head Attention**: 
  - Number of heads: 32
  - Dimension per head: 128
  - Total MHA dimension: 4,096 (32 × 128)
- **MLP Hidden Dimension**: 16,384

#### 1.3 Performance Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Per-token latency measurement

### 2. Parallel Deployment Comparison

#### 2.1 Baseline Configuration (TP=8, PP=2)
- **Parallelism Strategy**: 
  - Tensor Parallelism: 8-way split
  - Pipeline Parallelism: 2 stages
- **GPU Allocation**: adequate H100 GPUs (shared among experts)
- **Expert Placement**: 
  - Multiple experts colocated on each GPU
  - TP shards for all layers on each GPU
- **Processing Flow**: 
  - Sequential token flow through pipeline stages
  - Shared compute resources among colocated experts

#### 2.2 Proposed Method (Cross-Node Expert Parallelism)
- **Parallelism Strategy**: 
  - Expert Parallelism: 16 (large EP)
  - One expert per GPU per layer
- **GPU Allocation**: adequate H100 GPUs (one GPU per expert per layer)
- **Expert Placement**: 
  - Each GPU hosts exactly one expert per layer
  - No expert sharing across layers
- **Processing Flow**: 
  - Dynamic token routing to destination expert
  - Asynchronous token batch transfer
  - Parallel computation across all 16 experts

### 3. Performance Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | adequate | TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node EP | adequate | 1 expert each layer per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× TPOT |

### 4. Performance Analysis

#### 4.1 Throughput Improvements
- **3.75× Increase**: From 120,000 to 450,000 TPS
- **Root Cause**: Elimination of intra-GPU contention
- **Parallelism Gain**: All 16 experts compute simultaneously

#### 4.2 Latency Reduction
- **3.8× Decrease**: From 8.3ms to 2.2ms TPOT
- **Contributing Factors**:
  - No resource sharing between experts
  - Asynchronous communication overlap
  - Elimination of pipeline stalls

#### 4.3 Scalability Characteristics
- **Near-linear Scaling**: With 16 GPUs per expert layer
- **Large EP Regime**: EP ≥ 16 shows optimal performance
- **Resource Utilization**: Full GPU compute utilization per expert

### 5. Discussion Points

#### 5.1 Infrastructure Requirements
- **High-Bandwidth Network**: Essential for cross-node communication
- **Sufficient GPU Count**: Must support one-expert-per-GPU constraint
- **Memory Per GPU**: Adequate for single expert storage

#### 5.2 Trade-offs
- **Communication vs Compute**: Optimized for compute-heavy scenarios
- **Resource Allocation**: Higher absolute GPU count requirement
- **Network Overhead**: Mitigated by asynchronous communication

#### 5.3 Applicability
- **Inference-Only**: Current evaluation limited to inference
- **HPC Environments**: Optimal for high-performance clusters
- **Large-Scale Deployments**: Best suited for abundant GPU resources