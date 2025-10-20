# Phase 3: Complete Experiments of Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half-precision floating point)

### 1.2 Input Specifications
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8192 dimensions per token
- **Multi-Head Attention Details**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192

### 1.3 Expert Architecture
- **MLP Hidden Size**: 32,768 (4× token dimension)
- **Expert Type**: Standard MLP within each MoE layer

### 1.4 Hardware Configuration
- **GPU Type**: NVIDIA H100
- **Total GPUs**: 16
- **Use Case**: Inference-only (no training)

### 1.5 Performance Metrics
- **TPS (Tokens per Second)**: Primary throughput metric
- **TPOT (Time per Output Token)**: Primary latency metric (milliseconds)

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
**Configuration Details**:
- **Total GPUs**: 16 H100
- **Tensor Parallelism (TP)**: 8 (splits tensor across 8 GPUs)
- **Pipeline Parallelism (PP)**: 2 (splits model into 2 pipeline stages)
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)

**Per-GPU Allocation**:
- Each GPU holds 1/8 of tensor-parallel shard for all 4 layers
- Each pipeline stage spans 8 GPUs (2 stages × 8 GPUs = 16 total)
- **Expert Placement**: 8 experts per layer colocated on each GPU (shared resources)
- **Processing**: Sequential pipeline stages with shared expert computation

**Resource Sharing Characteristics**:
- GPUs shared among multiple experts
- Intra-GPU expert contention occurs
- Pipeline stalls due to sequential processing

### 2.2 Proposed Cross-Node Expert Parallelism
**Configuration Details**:
- **Total GPUs**: 16 H100 (one GPU per expert per layer)
- **Expert Parallelism (EP)**: 16 (maximum possible with 16 GPUs)
- **Tensor Parallelism (TP)**: 1 (no tensor splitting within expert)
- **Pipeline Parallelism (PP)**: 1 (no pipeline splitting)

**Per-GPU Allocation**:
- **Exact Expert Placement**: Each GPU hosts exactly one expert per layer
- **Layer Distribution**: All 4 layers have their 16 experts distributed across 16 GPUs
- **Dedicated Resources**: No expert sharing - complete GPU dedicated to single expert

**Routing Mechanism**:
- **Dynamic Token Routing**: Input tokens routed to GPU holding corresponding expert
- **Asynchronous Communication**: Token batches sent asynchronously to minimize idle time
- **Load Balancing**: Automatic distribution prevents expert overloading

**Processing Characteristics**:
- **Full Parallelism**: All 16 experts per layer compute simultaneously
- **No Contention**: Each expert has dedicated GPU resources
- **Optimal Utilization**: Maximum GPU compute utilization achieved

## 3. Experimental Results

### 3.1 Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment Strategy | TPS (Tokens/s) | TPOT (ms) | Improvement Factor |
|--------|-----------|----------------------------|----------------|-----------|-------------------|
| **Baseline (TP=8, PP=2)** | 16 | 8 experts each layer + TP shard per GPU (shared) | 120,000 | 8.3 | 1.0× (reference) |
| **Proposed Cross-Node Expert Parallelism** | 16 | 1 expert each layer per GPU (dedicated) | 450,000 | 2.2 | 3.75× throughput, 3.8× latency |

### 3.2 Detailed Analysis

**Throughput Improvement**:
- Absolute increase: 450,000 - 120,000 = 330,000 TPS
- Relative improvement: 450,000 ÷ 120,000 = 3.75× higher throughput

**Latency Reduction**:
- Absolute reduction: 8.3 - 2.2 = 6.1 ms per token
- Relative improvement: 8.3 ÷ 2.2 = 3.77× lower latency
(rounded to 3.8× in paper)

### 3.3 Bottleneck Analysis

**Baseline Bottlenecks**:
- **Compute Contention**: Multiple experts sharing GPU compute units
- **Memory Bandwidth**: Shared memory access patterns
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Load Imbalance**: Uneven expert activation causing idle time

**Proposed Solution Bottlenecks**:
- **Network Communication**: Cross-node token transfers
- **Load Balancing**: Dynamic routing to prevent stragglers
- **Synchronization**: Coordination across 16 parallel experts

### 3.4 Scalability Characteristics

**Linear Scaling Region**:
- With 16 GPUs achieving EP=16, system operates in optimal regime
- Network bandwidth sufficient to sustain communication overhead
- Compute fully utilized without contention

**Resource Utilization**:
- **Baseline**: GPU utilization limited by expert sharing
- **Proposed**: 100% GPU utilization per expert (compute-bound)

### 3.5 Validation of Design Goals

**Goal 1**: Maximize expert-level parallelism → Achieved through EP=16
**Goal 2**: Eliminate intra-GPU contention → Achieved with one-expert-per-GPU
**Goal 3**: Overlap communication with computation → Achieved via asynchronous routing
**Goal 4**: Scale to large clusters → Validated with 16-GPU configuration

## 4. Experimental Validity

**Controlled Variables**:
- Same model architecture (4-layer MoE, 16 experts/layer)
- Same hardware (16 H100 GPUs)
- Same input specifications (1024×10,000 tokens)
- Same precision (FP16)

**Variable**: Parallelism strategy (TP/PP vs EP)

**Conclusion**: Results demonstrate clear superiority of large EP approach under identical resource constraints.