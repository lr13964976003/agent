# Phase 2: Methodology Extraction

## Methodology Overview

Our approach maximizes expert-level parallelism through three key components:
1. **Expert Placement Strategy** - Assigning experts across GPUs and nodes
2. **Routing and Load Balancing** - Ensuring balanced input distribution
3. **Communication Overlap and Scheduling** - Minimizing cross-node transfer impact

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Mathematical Formulation**:
  - Given: E experts, G GPUs
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs maximizing concurrency
- **Benefit**: Minimal resource contention, full GPU compute utilization

### 1.2 Cross-Node Distribution
- **Topology-aware placement considering**:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Optimization Goal**: Minimize max tokens sent across any single link
- **Placement Algorithm**: Ensures one-expert-per-GPU principle maintained

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE approach**: Top-K gating scores determine expert activation
- **K value**: Not explicitly stated, but implied K=1 for maximum specialization
- **Dynamic adjustment**: Monitor per-expert load, adjust gating probabilities

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches async to overlap with computation
- **Load Balancing**: Dynamic adjustment to prevent expert overloading

### 2.3 Routing Process Flow
1. Input tokens processed by gating network
2. Tokens grouped by destination expert
3. Batched tokens asynchronously sent to expert locations
4. Expert computation begins as soon as partial batch arrives

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Technique**: Interleave expert computation with communication
- **CUDA Streams**: Utilized for asynchronous communication
- **Libraries**: NCCL or MPI for high-performance communication
- **Mechanism**: While current batch processes, next batch transfers in parallel

### 3.2 Pipeline Scheduling
- **Multi-layer MoE scheduling**:
  - Token outputs from previous layer immediately routed to next layer's experts
  - Subsequent layer experts start processing partial batches
  - Fine-grained pipeline increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Primary limiting factor**: Network bandwidth
- **Mitigation**: Topology-aware routing and token batching
- **Benefit**: One-expert-per-GPU ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within GPU if expert exceeds memory
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Integration**: Maintains high expert-level parallelism while handling memory constraints

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPU**: H100 (based on experimental setup)
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch fabrics
- **Cluster Size**: Minimum 16 GPUs for large EP regime

### 5.2 Software Stack
- **Precision**: BF16 for reduced memory footprint
- **Communication**: NCCL/MPI for cross-node transfers
- **Scheduling**: CUDA streams for asynchronous operations

### 5.3 Model Architecture Details
- **Layers**: 16 MoE layers
- **Experts per layer**: 16
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Token dimension**: 4096
- **MLP hidden size**: 16384
- **Multi-head attention**: 32 heads × 128 dimensions = 4096 total
- **Sequence length**: 10,000 tokens
- **Batch size**: 128 sequences

## 6. Comparative Analysis

### 6.1 Baseline Method
- **TP=8, PP=2** configuration
- **GPU usage**: 16 H100 GPUs
- **Per-GPU allocation**: 
  - 1/8 tensor-parallel shard for all layers
  - 8 experts per layer per GPU
- **Limitations**: Intra-GPU contention, pipeline stalls

### 6.2 Proposed Method
- **Single-expert-per-GPU** deployment
- **GPU usage**: 16 H100 GPUs (exact match for 16 experts per layer)
- **Per-GPU allocation**: Exactly one expert per layer per GPU
- **Advantages**: Maximal expert-level parallelism, minimal contention