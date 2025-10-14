# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8192 dimensions per token
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192
- **MLP hidden size**: 32,768

### 1.2 Hardware Configuration
- **GPU type**: NVIDIA H100
- **Total GPUs**: 16 H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Network**: High-bandwidth interconnect (InfiniBand/NVLink)

### 1.3 Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token in milliseconds

## 2. Baseline Configuration (TP=8, PP=2)

### 2.1 Parallelism Strategy
- **Tensor Parallelism (TP)**: 8-way tensor parallelism
- **Pipeline Parallelism (PP)**: 2 pipeline stages
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)
- **Total GPUs**: 16 (8 × 2)

### 2.2 GPU Allocation Details
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - Experts are colocated on GPUs: 8 experts per layer per GPU
  - Total experts per layer: 16, distributed as 8 experts on each of 2 GPUs

### 2.3 Processing Flow
- **Sequential processing**: Tokens flow sequentially through pipeline stages
- **Resource sharing**: Multiple experts share compute resources on each GPU
- **Contention**: Intra-GPU contention between experts

### 2.4 Performance Results
- **TPS**: 120,000 tokens/second
- **TPOT**: 8.3 milliseconds per token

## 3. Proposed Cross-Node Expert Parallelism

### 3.1 Parallelism Strategy
- **Expert Parallelism (EP)**: 16-way expert parallelism
- **Tensor Parallelism (TP)**: 1 (no tensor parallelism within experts)
- **Pipeline Parallelism (PP)**: 1 (no pipeline stages)
- **Total GPUs**: 16 (one GPU per expert per layer)

### 3.2 GPU Allocation Details
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert per layer
  - Expert 0 of layer 0 on GPU 0
  - Expert 1 of layer 0 on GPU 1
  - ...
  - Expert 15 of layer 3 on GPU 63 (but only 16 GPUs used, so experts are reused across layers)
  - **Correction**: With 4 layers × 16 experts = 64 expert instances total, but only 16 GPUs
  - **Actual deployment**: Each GPU hosts one expert, but this expert serves all 4 layers (expert parameters shared across layers)
  - **Clarification**: Each GPU hosts expert i for all 4 layers (expert weights shared across layers)

### 3.3 Processing Flow
- **Parallel processing**: All 16 experts per layer compute in parallel
- **Dynamic routing**: Input tokens dynamically routed to GPU holding corresponding expert
- **Asynchronous communication**: Token batches sent asynchronously to minimize idle time
- **No contention**: Each expert runs in isolation on dedicated GPU

### 3.4 Performance Results
- **TPS**: 450,000 tokens/second
- **TPOT**: 2.2 milliseconds per token

## 4. Performance Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Performance Gain |
|--------|-----------|-------------------|----------------|-----------|------------------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 | 1.0× (baseline) |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× lower latency |

## 5. Detailed Analysis

### 5.1 Throughput Improvement
- **Absolute improvement**: 450,000 - 120,000 = 330,000 tokens/second
- **Relative improvement**: (450,000/120,000) = 3.75× higher throughput
- **Root cause**: Elimination of intra-GPU contention and maximized expert parallelism

### 5.2 Latency Reduction
- **Absolute reduction**: 8.3 - 2.2 = 6.1 milliseconds per token
- **Relative reduction**: (8.3/2.2) = 3.8× lower latency
- **Root cause**: Parallel expert processing and overlapped communication

### 5.3 Resource Utilization
- **Baseline**: GPU resources shared among 8 experts per GPU
- **Proposed**: Each GPU dedicated to single expert, full compute utilization
- **Memory efficiency**: No parameter replication within GPU

### 5.4 Communication Overhead
- **Baseline**: Minimal inter-node communication due to colocation
- **Proposed**: Increased inter-node communication offset by:
  - Token batching
  - Asynchronous routing
  - Compute-communication overlap
  - High-bandwidth interconnect

## 6. Scalability Analysis

### 6.1 Linear Scaling Validation
- **Tested scale**: 16 GPUs (EP=16)
- **Large EP regime**: EP ≥ 16 as defined in paper
- **Scaling behavior**: Near-linear scaling observed
- **Limiting factor**: Network bandwidth at scale

### 6.2 Bottleneck Analysis
- **Baseline bottleneck**: Intra-GPU expert contention
- **Proposed bottleneck**: Network communication (mitigated by overlap)
- **Next scaling challenge**: Memory capacity for larger models

## 7. Experimental Validity

### 7.1 Controlled Variables
- **Model architecture**: Identical between baseline and proposed
- **Hardware**: Same 16 H100 GPUs
- **Batch size**: Fixed at 1024 sequences
- **Sequence length**: Fixed at 10,000 tokens
- **Precision**: FP16 for both methods

### 7.2 Fair Comparison
- **Resource equivalence**: Both methods use exactly 16 GPUs
- **Model equivalence**: Same number of parameters and computation
- **Workload equivalence**: Same input batch for both methods
- **Measurement**: Same metrics (TPS, TPOT) for both methods