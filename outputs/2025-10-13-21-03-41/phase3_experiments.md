# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Specifications
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
- **MLP Hidden Size**: 32,768

### 1.3 Hardware Configuration
- **GPUs**: 16 H100 GPUs
- **Setting**: Inference-only (no training evaluation)
- **Network**: High-performance computing (HPC) environment with high-bandwidth interconnects

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (Comparison Method)
- **Parallelism Strategy**: TP=8, PP=2
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Pipeline consists of 2 stages, each spanning 8 GPUs
  - Experts are colocated: 8 experts per layer per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Parallelism Strategy**: Large EP (Expert Parallelism) with EP=16
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Routing Strategy**: Input tokens dynamically routed to GPU holding corresponding expert
- **Communication**: Token batches sent asynchronously to ensure minimal idle time

## 3. Experimental Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput Improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Expert Parallelism**: Full 16-way parallelism achieved vs shared GPU resources

## 4. Results Analysis

### 4.1 Baseline Limitations
- **Intra-GPU Contention**: Multiple experts sharing same GPU cause resource contention
- **Pipeline Stalls**: Sequential processing through pipeline stages creates bottlenecks
- **Reduced Parallelism**: Limited expert-level concurrency due to colocation

### 4.2 Proposed Method Advantages
- **Maximized Expert Parallelism**: All 16 experts per layer compute in parallel
- **Dedicated Resources**: Each expert has exclusive access to GPU compute and memory
- **Asynchronous Operations**: Non-blocking token routing and communication overlap
- **Load Balancing**: Dynamic routing prevents expert overloading

### 4.3 Scalability Characteristics
- **Large EP Regime**: Benefits most pronounced when EP ≥ 16
- **Network Efficiency**: Topology-aware placement minimizes cross-node communication
- **Compute Utilization**: Near 100% GPU utilization through expert isolation

## 5. Key Findings

### 5.1 Throughput Gains
- Primary driver: Elimination of intra-GPU expert contention
- Secondary benefit: Overlapped communication and computation
- Network overhead successfully mitigated through careful scheduling

### 5.2 Latency Reduction
- Direct result of parallel expert processing
- No pipeline stalls from sequential layer processing
- Immediate token routing to available experts

### 5.3 Resource Utilization
- **GPUs**: All 16 GPUs fully utilized for expert computation
- **Memory**: Balanced distribution prevents memory hotspots
- **Network**: Efficient token batching minimizes communication overhead

## 6. Experimental Validations

### 6.1 Large EP Hypothesis
- Confirmed that EP ≥ 16 provides significant performance benefits
- Validated one-expert-per-GPU deployment strategy
- Demonstrated scalability in high-performance computing environments

### 6.2 Communication Overhead Management
- Successfully overlapped communication with computation
- Topology-aware placement reduced network bottlenecks
- Asynchronous routing prevented idle time

### 6.3 Load Balancing Effectiveness
- Dynamic gating prevented expert overloading
- Balanced token distribution across all experts
- No significant straggler effects observed

## 7. Reproducibility Details

### 7.1 Model Architecture
- Standard transformer architecture with MoE layers
- No custom modifications to basic MoE structure
- FP16 precision for consistent memory usage

### 7.2 Hardware Consistency
- Same 16 H100 GPUs used for both methods
- Identical network infrastructure
- Controlled environment eliminates hardware variations

### 7.3 Input Standardization
- Fixed batch size and sequence length
- Consistent token dimensions across experiments
- Same routing algorithms for fair comparison