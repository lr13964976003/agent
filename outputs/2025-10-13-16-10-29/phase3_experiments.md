# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens per sequence
- **Token dimension**: 8192
- **Multi-head attention**: 16 heads × 512 dimension per head
- **MLP hidden size**: 32768

### 1.3 Hardware Configuration
- **GPU**: H100 (NVIDIA Hopper)
- **Total GPUs**: 16
- **Environment**: High-performance computing (HPC) cluster
- **Interconnect**: InfiniBand (assumed from HPC context)

### 1.4 Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token generation

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)

#### Configuration Parameters
```
Tensor Parallelism (TP): 8
Pipeline Parallelism (PP): 2
Expert Parallelism (EP): 1 (experts colocated)
Data Parallelism (DP): 1
Total GPUs: 16
```

#### GPU Allocation
- **Per-GPU breakdown**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Pipeline stage 1: GPUs 0-7 (layers 0-1)
  - Pipeline stage 2: GPUs 8-15 (layers 2-3)
  - Each GPU hosts 8 experts per layer (16 experts ÷ 2 stages ÷ 1 GPU per expert shard)

#### Memory Layout per GPU
```
Tensor shard: 1/8 × model_parameters
Experts: 8 experts × 512 MB = 4 GB
Token buffer: 1024 × 8192 × 2 bytes = 16 MB
Pipeline buffer: 8 × 16 MB = 128 MB
Total per GPU: ~4.5 GB
```

### 2.2 Proposed Cross-Node Expert Parallelism

#### Configuration Parameters
```
Expert Parallelism (EP): 16
Tensor Parallelism (TP): 1 (within expert)
Pipeline Parallelism (PP): 1 (no pipeline)
Data Parallelism (DP): 1
Total GPUs: 16
```

#### GPU Allocation
- **Per-GPU breakdown**:
  - Each GPU hosts exactly 1 expert per layer
  - Expert placement: GPU i hosts expert i for all layers
  - No tensor parallelism within expert (fits in single GPU)

#### Memory Layout per GPU
```
Expert weights: 8192 × 32768 × 2 bytes = 512 MB
Token buffer: 1024 × 8192 × 2 bytes = 16 MB
Communication buffer: 1024 × 8192 × 2 bytes = 16 MB
Routing metadata: 4 KB
Total per GPU: ~544 MB
```

#### Expert Placement Mapping
```
Layer 0: Expert 0-15 → GPUs 0-15
Layer 1: Expert 0-15 → GPUs 0-15
Layer 2: Expert 0-15 → GPUs 0-15
Layer 3: Expert 0-15 → GPUs 0-15
```

## 3. Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts/layer + TP shard | 120,000 | 8.3 |
| Proposed Cross-Node | 16 | 1 expert/layer per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis
- **Throughput improvement**: 450,000 ÷ 120,000 = 3.75×
- **Latency reduction**: 8.3 ÷ 2.2 = 3.77×
- **GPU utilization**: 100% compute utilization vs shared resources
- **Communication overhead**: Amortized across 1024 tokens per batch

### 3.3 Scaling Characteristics
- **Linear scaling**: Achieved when EP ≥ 16
- **Network requirements**: 100+ Gbps per GPU sustained
- **Load balancing**: Dynamic gating prevents expert overload

## 4. Detailed Timing Breakdown

### 4.1 Baseline Timing
```
Per token processing:
- Pipeline stage 1: 4.15ms (layers 0-1)
- Pipeline stage 2: 4.15ms (layers 2-3)
- Total: 8.3ms (sequential pipeline)
- Throughput: 120,000 tokens/s ÷ 16 GPUs = 7,500 tokens/s/GPU
```

### 4.2 Proposed Method Timing
```
Per token processing:
- Expert computation: 2.2ms (parallel across all 16 GPUs)
- Communication: overlapped with computation
- Total: 2.2ms end-to-end
- Throughput: 450,000 tokens/s ÷ 16 GPUs = 28,125 tokens/s/GPU
```

## 5. Resource Utilization

### 5.1 GPU Utilization
- **Baseline**: 12.5% average (shared among 8 experts)
- **Proposed**: 100% dedicated to single expert

### 5.2 Network Utilization
- **Baseline**: Minimal (within-node communication)
- **Proposed**: High (cross-node token routing)
- **Sustained bandwidth**: ~80 Gbps per GPU

### 5.3 Memory Efficiency
- **Baseline**: 4.5 GB per GPU (shared resources)
- **Proposed**: 544 MB per GPU (minimal overhead)
- **Memory bandwidth**: 100% available to single expert