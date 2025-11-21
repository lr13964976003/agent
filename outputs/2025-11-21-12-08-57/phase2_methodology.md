# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
```
Given:
- E = number of experts per layer (16 in paper)
- G = number of available GPUs (16 in paper)
- L = number of MoE layers (16 in paper)

Constraint: Each GPU hosts at most one expert per layer

If E ≤ G:
  - Assign each expert to a distinct GPU
  - GPU ID = expert_index mod G
  
If E > G:
  - Replicate experts across GPUs
  - Ensure maximum independence of expert computation
  - Balance memory usage across replicas
```

### 1.2 Cross-Node Distribution Algorithm
- **Input**: Cluster topology (node-to-node bandwidth, latency, GPU memory capacity)
- **Goal**: Minimize maximum tokens sent across any single link
- **Output**: Expert-to-GPU mapping matrix

```
Variables:
- B[i,j] = bandwidth between node i and node j
- L[i,j] = latency between node i and node j
- M[k] = memory capacity of GPU k
- R[e] = expected routing load for expert e

Objective:
Minimize max_{i,j} (Tokens_{i→j} / B[i,j])
Subject to:
- Each GPU hosts ≤ 1 expert per layer
- Memory constraints: Σ weights_on_GPU_k ≤ M[k]
```

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
```
Input: Token embedding t ∈ ℝ^4096
Gating network: G(t) = softmax(W_g t + b_g) ∈ ℝ^E
Top-K selection: Select top 2 experts based on gating scores (standard MoE practice)

Output: Expert indices {e1, e2} and weights {w1, w2}
```

### 2.2 Token Sharding Process
```
For each MoE layer:
1. Batch tokens by destination expert
2. Create token batches B_e for each expert e
3. Asynchronously send B_e to GPU hosting expert e
4. Overlap with current layer computation
```

### 2.3 Load Balancing Algorithm
```
Dynamic adjustment:
- Monitor expert load L_e = number of tokens processed by expert e
- Update gating probabilities: P_e = P_e * (1 - α) + α * (L_avg / L_e)
- α = 0.01 (learning rate for load balancing)
```

## 3. Communication Overlap and Scheduling

### 3.1 CUDA Stream Architecture
```
Stream 1: Compute - Expert computation
Stream 2: Communication - Token transfers
Stream 3: Synchronization - All-reduce operations
```

### 3.2 Pipeline Scheduling
```
Layer-wise pipeline:
For layer l from 1 to L:
  1. Receive tokens from layer l-1 (async)
  2. Begin expert computation as soon as partial batch arrives
  3. Send completed tokens to layer l+1 (async)
  4. Continue processing remaining tokens
```

### 3.3 Memory Layout
```
Per-GPU memory allocation:
- Expert weights: 16384 × 4096 × 2 bytes (BF16) = 128 MB
- Token buffer: 4096 × batch_size × sequence_length × 2 bytes = 10.48 GB (for 128×10000 tokens)
- Communication buffer: 4096 × max_tokens_per_expert × 2 bytes
```

## 4. Parallelism Integration

### 4.1 Tensor Model Parallelism (If needed)
```
When expert exceeds single-GPU memory:
- Apply column-parallel to first linear layer of MLP
- Apply row-parallel to second linear layer of MLP
- Expert hidden dimension splits: 16384 → 8192 per device
```

### 4.2 Data Parallelism
```
- Applied across MoE network replicas
- Synchronized weight updates
- Maintains expert-level parallelism within each replica
```

## 5. Optimization Parameters

### 5.1 Communication Parameters
```
- NCCL algorithm: Ring-based all-reduce for small messages, Tree-based for large messages
- Buffer size: 256 MB per communication stream
- Async threshold: 1 MB (messages > 1MB use async communication)
```

### 5.2 Load Balancing Parameters
```
- Monitoring interval: 100 iterations
- Rebalancing threshold: 10% load imbalance
- Expert capacity factor: 1.2 (allow 20% overflow)
```

### 5.3 Scheduling Parameters
```
- Pipeline depth: 4 stages
- Micro-batch size: 32 sequences per micro-batch
- Overlap granularity: Token-level (not sequence-level)
```