# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert per layer
- **Allocation Rule**: 
  - If E ≤ G (experts ≤ GPUs): Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize independent expert concurrency while balancing memory usage
- **Benefit**: Eliminates intra-GPU expert contention, fully utilizes GPU compute units

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard top-K gating scores determine expert activation per token
- K typically = 1 or 2 for sparse activation

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: 
  - Monitor per-expert load
  - Dynamically adjust gating probabilities to prevent expert overloading
  - Prevent stragglers that degrade throughput

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While current batch processes on GPU, next batch transfers simultaneously from other nodes
  - CUDA streams or NCCL/MPI for asynchronous communication
  - Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE Networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches without waiting for full batch
  - Fine-grained pipeline increases throughput and reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth**: Primary limiting factor in large EP setups
- **Mitigation**: Topology-aware routing and token batching
- **One-Expert-Per-GPU**: Ensures all GPUs fully utilized for compute while communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within expert if single expert exceeds GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamless integration with existing parallelism strategies

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPUs**: H100-class or equivalent with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or NVSwitch fabrics
- **Memory**: Sufficient per-GPU memory for single expert + activations

### 5.2 Software Stack
- **Communication Libraries**: NCCL, MPI, or custom async communication
- **Deep Learning Framework**: PyTorch or TensorFlow with custom MoE implementation
- **Scheduling**: CUDA streams for async operations

### 5.3 Memory Layout
- **Expert Parameters**: Stored contiguously on assigned GPU
- **Token Buffers**: Double-buffering for async token transfer
- **Activation Storage**: Temporary buffers for intermediate computations