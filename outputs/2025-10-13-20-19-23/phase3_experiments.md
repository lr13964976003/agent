# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Token Dimension**: 8192 dimensions per token

### 1.3 Attention Configuration
- **Multi-Head Attention (MHA)**:
  - Number of Heads: 16
  - Dimension per Head: 512
  - Total Attention Dimension: 16 × 512 = 8192 (matches token dimension)

### 1.4 MLP Configuration
- **Hidden Size**: 32768 neurons in MLP hidden layer
- **Expert Architecture**: Standard transformer FFN with GELU activation

### 1.5 Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement in milliseconds

### 1.6 Hardware Environment
- **GPU Type**: H100 GPUs
- **Total GPUs**: 16 for both baseline and proposed methods
- **Setting**: Inference-only evaluation

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Parallelism Configuration**:
  - Tensor Parallelism (TP): 8-way split
  - Pipeline Parallelism (PP): 2 stages
  - Expert Parallelism (EP): Not explicitly used (experts colocated)

- **GPU Allocation**:
  - Total GPUs: 16 H100s
  - Per-GPU Allocation:
    - Each GPU holds 1/8 of tensor-parallel shard for all layers
    - Each pipeline stage spans 8 GPUs (2 stages × 8 GPUs = 16 GPUs)
    - Experts colocated: 8 experts per layer per GPU (16 experts/layer ÷ 2 stages = 8 experts/GPU)

- **Processing Flow**:
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU expert contention occurs

### 2.2 Proposed Cross-Node Expert Parallelism
- **Parallelism Configuration**:
  - Expert Parallelism (EP): 16 (one expert per GPU per layer)
  - Tensor Parallelism (TP): Not used within experts
  - Pipeline Parallelism (PP): Not used (layers processed in parallel)

- **GPU Allocation**:
  - Total GPUs: 16 H100s
  - Per-GPU Allocation:
    - Each GPU hosts exactly **one expert per layer**
    - 16 experts per layer × 4 layers = 64 expert instances total
    - 64 experts ÷ 16 GPUs = 4 experts per GPU (one per layer)

- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding target expert
  - Token batches sent asynchronously
  - Minimal idle time through overlapping computation and communication

## 3. Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | **450,000** | **2.2** |

### 3.2 Performance Improvements
- **Throughput Gain**: 3.75× improvement (450,000 vs 120,000 tokens/second)
- **Latency Reduction**: 3.8× improvement (2.2ms vs 8.3ms TPOT)
- **Expert Utilization**: 100% expert-level parallelism achieved
- **Resource Efficiency**: Eliminated intra-GPU contention

### 3.3 Scalability Analysis
- **Linear Scaling**: Near-linear scaling demonstrated with 16 GPUs
- **Communication Overhead**: Successfully mitigated through async routing
- **Memory Usage**: Optimal memory utilization per GPU

## 4. Discussion

### 4.1 Key Insights
- **Expert Contention**: Baseline suffers from multiple experts sharing GPU resources
- **Parallelism Efficiency**: Proposed method achieves true expert-level parallelism
- **Communication vs Compute**: Trade-off successfully shifted from compute contention to manageable communication overhead

### 4.2 Deployment Advantages
- **Simplified Architecture**: One expert per GPU eliminates complex resource sharing
- **Predictable Performance**: Consistent latency due to dedicated GPU per expert
- **Scalability**: Method scales with available GPUs and network bandwidth

### 4.3 Limitations and Considerations
- **Network Requirements**: Requires high-bandwidth interconnects (NVLink/InfiniBand)
- **GPU Count**: Requires at least as many GPUs as experts per layer for optimal deployment
- **Memory Constraints**: Each expert must fit within single GPU memory

### 4.4 Future Extensions
- **Training Scenarios**: Method can be extended to training with gradient synchronization
- **Dynamic Load Balancing**: Real-time expert load monitoring and routing adjustment
- **Larger Models**: Scales to thousands of experts with sufficient GPU resources