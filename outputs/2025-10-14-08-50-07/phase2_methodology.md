# Phase 2: Methodology Extraction

## Expert Placement Strategy

### 1. Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Assignment Rule**: 
  - If E ≤ G (experts ≤ GPUs): Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs while maximizing concurrency
- **Benefit**: Eliminates intra-GPU expert contention

### 2. Cross-Node Distribution Algorithm
- **Inputs**: 
  - E = number of experts per layer (16)
  - G = total GPUs available (16)
  - Network topology (bandwidth, latency matrix)
  - GPU memory capacity per node
- **Output**: Expert-to-GPU mapping matrix
- **Constraints**:
  - One expert per GPU maximum
  - Minimize max tokens per network link
  - Balance memory usage across nodes

## Routing and Load Balancing

### 1. Gating Mechanism
- **Top-K Selection**: For each token, select top-K experts based on gating scores
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities

### 2. Token Sharding Process
```
Input: Batch of tokens B = [t1, t2, ..., tN]
Process:
1. Compute gating scores for all tokens
2. Group tokens by destination expert
3. Create token batches per expert
4. Asynchronously send batches to target GPUs
5. Process tokens on destination GPUs
6. Return results to originating GPUs
```

### 3. Load Balancing Algorithm
- **Metrics**: Tokens processed per expert, queue lengths, processing times
- **Adjustment**: Modify gating probabilities to prevent expert overload
- **Frequency**: Continuous monitoring with periodic rebalancing

## Communication Overlap and Scheduling

### 1. Overlapping Compute and Communication
- **Mechanism**: Use CUDA streams or NCCL/MPI for async communication
- **Timeline**:
  - Time t0: Send batch n to GPU i
  - Time t1: Process batch n-1 on GPU i
  - Time t2: Receive results from GPU i
- **Benefit**: Hide communication latency behind computation

### 2. Pipeline Scheduling for Multi-Layer MoE
- **Layer-wise Processing**:
  - Layer L: Process tokens, generate outputs
  - Immediately route outputs to Layer L+1 experts
  - Layer L+1: Begin processing as soon as partial batch arrives
- **Optimization**: Reduce idle time through fine-grained pipelining

## Scalability Framework

### 1. Large EP Regime (EP ≥ 16)
- **Network Optimization**: 
  - Topology-aware routing
  - Token batching to reduce message count
  - Bandwidth utilization monitoring
- **Compute Optimization**:
  - One expert per GPU ensures full utilization
  - Communication costs amortized across many tokens

### 2. Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Apply within expert if single expert exceeds GPU memory
- **Data Parallelism (DP)**: Replicate entire MoE network for synchronized updates
- **Hierarchy**: EP (across experts) → TP (within expert) → DP (across replicas)

## Technical Parameters
- **Expert Count**: 16 experts per layer
- **GPU Allocation**: 1 expert per GPU
- **Communication**: NCCL/MPI with CUDA streams
- **Batching**: Token-level batching by destination expert
- **Precision**: FP16 for computation and communication
- **Memory**: Each expert requires 8192 × 32768 × 2 bytes = 512 MB for weights alone