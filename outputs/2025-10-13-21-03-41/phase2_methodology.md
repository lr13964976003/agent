# Phase 2: Methodology Extraction

## Method Overview
The proposed method focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, with three key components:

1. Expert Placement Strategy
2. Routing and Load Balancing
3. Communication Overlap and Scheduling

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Implementation**:
  - For MoE layer with E experts and cluster of G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K gating scores per input token
- Subset of experts activated based on highest scores

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While one batch processes on GPU, next batch transfers simultaneously from other nodes
  - CUDA streams or asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing as soon as partial batch arrives (no waiting for full batch)
- **Benefit**: Fine-grained pipeline increases throughput and reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - Mitigated by topology-aware routing and token batching
  - One-expert-per-GPU ensures all GPUs fully utilized for compute while communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **For models exceeding single-GPU memory**:
  - Each expert can be partitioned using tensor model parallelism (TP) within its GPU
  - Data parallelism (DP) applied across MoE network replicas for synchronized weight updates while maintaining high expert-level parallelism

## 5. Implementation Details

### 5.1 Model Architecture
- **Layers**: 4-layer MoE
- **Experts per layer**: 16
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16

### 5.2 Input Specifications
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens per sequence
- **Token dimension**: 8192
- **Multi-head attention**: 16 heads × 512 dimensions per head
- **MLP hidden size**: 32768

### 5.3 Hardware Requirements
- **GPUs**: 16 H100 GPUs
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand, H100-class NVSwitch)
- **Memory**: Sufficient per-GPU memory for single expert storage

## 6. Deployment Strategy

### 6.1 Baseline Comparison
- **Baseline**: TP=8, PP=2 using 16 H100 GPUs
  - Each GPU holds 1/8 tensor-parallel shard for all layers
  - 2 pipeline stages spanning 8 GPUs each
  - 8 experts per layer per GPU (colocated)

### 6.2 Proposed Method
- **Configuration**: 16 H100 GPUs (one GPU per expert per layer)
- **Per-GPU allocation**: Each GPU hosts exactly one expert per layer
- **Routing**: Dynamic token routing to GPU holding corresponding expert
- **Communication**: Asynchronous token batch transfer with minimal idle time

## 7. Performance Optimization Techniques

### 7.1 Asynchronous Operations
- Non-blocking communication primitives
- CUDA stream management for concurrent compute and communication
- Event-based synchronization for pipeline stages

### 7.2 Memory Management
- Expert weight storage optimization
- Token buffer management for cross-node transfers
- Dynamic memory allocation based on routing patterns

### 7.3 Load Monitoring
- Real-time expert utilization tracking
- Dynamic gating probability adjustment
- Hotspot detection and mitigation