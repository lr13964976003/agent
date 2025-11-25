# Phase 2: Methodology Extraction

## Methodology Overview

### Core Principle
Maximize expert-level parallelism by deploying at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Detailed Method Components

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Implementation**: 
  - For E experts and G GPUs: Each expert assigned to distinct GPU if E ≤ G
  - If E > G: Experts replicated across GPUs to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

#### 1.2 Cross-Node Distribution
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- Standard MoE routing: Top-K gating scores determine expert activation per token
- K=2 typically used for load balancing

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While batch N processed on GPU, batch N+1 transferred from other nodes
  - CUDA streams or NCCL/MPI for asynchronous communication

#### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs immediately routed to next layer's experts
  - Subsequent layer experts start processing partial batches instead of waiting for full batch
- **Benefit**: Fine-grained pipeline increases throughput, reduces idle time

### 4. Memory and Model Parallelism Integration

#### 4.1 Handling Large Models
- **Tensor Model Parallelism (TP)**: Applied within each expert's GPU if needed
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Synchronization**: Weight updates synchronized while maintaining high expert-level parallelism

### 5. Scalability Specifications

#### 5.1 Large EP Regime (EP ≥ 16)
- **Network optimization**: Topology-aware routing and token batching
- **Compute optimization**: One-expert-per-GPU ensures full GPU utilization
- **Trade-off**: Communication costs amortized across many tokens

#### 5.2 Model Architecture Parameters
- **Layers**: 16 MoE layers
- **Experts per layer**: 16 (matching EP=16)
- **Expert type**: MLP-based experts
- **Token dimension**: 4096
- **MLP hidden size**: 16384
- **Attention**: Multi-head attention with 32 heads, 128 dimensions per head
- **Precision**: BF16
- **Batch configuration**: 128 sequences × 10000 tokens per sequence

## Method Flow Summary

1. **Initialization**: Topology-aware expert placement across GPUs (1 expert/GPU/layer)
2. **Token Input**: Batch of 128×10000 tokens enters system
3. **Routing**: Gating network determines top-2 experts per token
4. **Token Distribution**: Tokens batched by destination expert, asynchronously routed
5. **Expert Computation**: Each expert processes assigned tokens independently
6. **Communication Overlap**: Next token batch transferred while current processed
7. **Pipeline Flow**: Processed tokens immediately routed to next layer's experts
8. **Load Balancing**: Dynamic adjustment of routing probabilities based on load
9. **Output Collection**: Final tokens collected across all experts and layers