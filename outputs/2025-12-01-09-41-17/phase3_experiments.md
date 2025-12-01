# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 16-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 64 experts
- **Precision**: FP8
- **Batch Configuration**: 128 sequences per batch
- **Sequence Length**: 128 tokens per sequence
- **Token Dimension**: 1024
- **Multi-Head Attention**: 16 heads, 64 dimensions per head
- **MOE Hidden Size**: 2048

### 1.2 Environment
- **Hardware**: H100 GPUs (adequate availability)
- **Setting**: Inference-only evaluation
- **Metrics**:
  - TPS (Tokens per Second): Throughput measurement
  - TPOT (Time per Output Token): Latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU holds 8 tensor-parallel shards for all layers
  - Experts are colocated on 16 GPUs
- **Processing Flow**: 
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources
- **Limitations**: 
  - Intra-GPU contention among experts
  - Pipeline stalls during sequential processing

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert per layer**
  - Total: 16 experts computing in parallel per layer
- **Routing Mechanism**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time
- **Advantages**:
  - All 16 experts per layer compute in parallel
  - No intra-GPU expert contention
  - Maximized throughput and minimized token latency

## 3. Performance Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | adequate | TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | adequate | 1 expert each layer per GPU | 450,000 | 2.2 |

### 3.1 Performance Improvements
- **Throughput Gain**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Efficiency**: Near-linear scaling achieved with EP ≥ 16

### 3.2 Key Performance Factors
- **Expert Isolation**: One expert per GPU eliminates resource contention
- **Parallel Execution**: All experts compute simultaneously
- **Asynchronous Routing**: Minimal waiting time across nodes
- **Load Balancing**: Prevents expert overloading

## 4. Scalability Analysis

### 4.1 Large EP Regime Benefits
- **EP ≥ 16**: Qualifies as large expert parallelism
- **Network Efficiency**: Advanced HPC networking sustains high bandwidth
- **Compute Saturation**: Full GPU utilization with minimal idle time
- **Communication Overlap**: Asynchronous transfers hide latency

### 4.2 Resource Utilization
- **GPU Compute**: Maximized through expert isolation
- **Memory Bandwidth**: Efficient usage without expert contention
- **Network Bandwidth**: Topology-aware placement minimizes hotspots
- **Overall Efficiency**: Balanced trade-off between communication and computation

## 5. Discussion Points

### 5.1 Deployment Advantages
- **Full GPU Utilization**: Each GPU dedicated to single expert
- **Reduced Contention**: No intra-GPU expert competition
- **Improved Throughput**: Parallel expert execution
- **Lower Latency**: Asynchronous token routing

### 5.2 Implementation Considerations
- **Network Requirements**: High-bandwidth, low-latency interconnects essential
- **Memory Management**: Careful expert placement for memory balance
- **Load Distribution**: Dynamic gating adjustments prevent overload
- **Topology Awareness**: Node placement optimized for communication patterns

## 6. Conclusion from Experiments
The experimental results validate that large-scale cross-node expert parallelism with one expert per GPU significantly outperforms traditional colocation approaches. The 3.75× throughput improvement and 3.8× latency reduction demonstrate the effectiveness of maximizing expert-level parallelism in HPC environments with adequate GPU resources.