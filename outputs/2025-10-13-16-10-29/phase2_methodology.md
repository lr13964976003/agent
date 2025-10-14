# Phase 2: Methodology Extraction

## Methods Overview

### Core Principle
Maximize expert-level parallelism by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources.

### Mathematical Formulation
- For a MoE layer with E experts and cluster of G GPUs
- Ensure each expert assigned to distinct GPU if E ≤ G
- If E > G, replicate experts to maximize concurrency while balancing memory

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
```
Constraint: max_experts_per_gpu = 1
Placement: expert_i → GPU_j where j = i mod G
```

### 1.2 Cross-Node Distribution Algorithm
**Inputs:**
- E: number of experts
- G: number of GPUs
- T: cluster topology (bandwidth matrix, latency matrix)
- M: GPU memory capacity per node
- P: expected token routing patterns

**Objective:**
Minimize: max_tokens_per_link
Subject to:
- ∑experts_on_gpu ≤ 1 for all GPUs
- ∑memory_usage ≤ M for all nodes
- Balance token routing across links

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Top-K gating scores determine expert activation
- K typically = 2 for MoE models
- Gating network: softmax(W_gate · x) → expert probabilities

### 2.2 Token Sharding Algorithm
```
For each token t:
1. Compute gating scores g = softmax(W_gate · t)
2. Select top-K experts: E_top = top_k_indices(g)
3. Route token to GPU hosting expert e ∈ E_top
4. Batch tokens by destination GPU
5. Send asynchronously using NCCL/MPI
```

### 2.3 Load Balancing
- Monitor per-expert load: L_e = tokens_processed_e / total_tokens
- Adjust gating probabilities: g'_e = g_e · (1 + α(1 - L_e))
- α: balancing factor (typically 0.1-0.3)

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap
```
CUDA Stream Configuration:
- Stream 0: Expert computation
- Stream 1: Token communication
- Stream 2: Gradient synchronization (if training)
```

### 3.2 Pipeline Scheduling
**Multi-layer MoE processing:**
```
For layer l = 1 to L:
    Async send tokens to experts in layer l+1
    Process tokens in layer l experts
    Sync: ensure layer l+1 receives all tokens
```

### 3.3 Token Batching Strategy
- Batch size: 1024 tokens per communication
- Batching window: 100μs-1ms (tunable)
- Priority: high-load experts first

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
**Network requirements:**
- Bandwidth: ≥ 100 Gbps per GPU (InfiniBand)
- Latency: ≤ 10μs node-to-node
- Topology: Fat-tree or dragonfly preferred

### 4.2 Memory and Model Parallelism Integration
**Tensor Parallelism within expert:**
- If expert_size > GPU_memory:
  - Apply TP within expert: expert_tensor → TP_size shards
  - TP_size = 2, 4, or 8 (power of 2)

**Data Parallelism across replicas:**
- DP_size = total_GPUs / (EP_size × TP_size × PP_size)
- Synchronized updates via NCCL AllReduce

## 5. Implementation Details

### 5.1 GPU Memory Layout
```
Per GPU allocation:
- Expert weights: 8192 × 32768 × 2 bytes (FP16) = 512 MB
- Token buffer: 1024 × 8192 × 2 bytes = 16 MB
- Communication buffer: 1024 × 8192 × 2 bytes = 16 MB
- Total per GPU: ~544 MB
```

### 5.2 Communication Protocol
- **Library**: NCCL 2.18+
- **Primitives**: ncclSend, ncclRecv, ncclAllToAllv
- **Optimization**: CUDA Graphs for fixed communication patterns

### 5.3 Routing Table
```
RoutingTable[i][j] = GPU_ID hosting expert j for layer i
Dimensions: [num_layers][num_experts]
Updated every N iterations (N=1000 for inference)
```