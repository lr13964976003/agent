# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Mathematical Formulation
Given:
- E: Total number of experts in MoE layer
- G: Total number of available GPUs

#### Case 1: E ≤ G (Expert count ≤ GPU count)
**Placement Formula:**
```
GPU_assignment[i] = GPU_i for expert_i where i ∈ [0, E-1]
```
Each expert e_i is assigned to GPU_i, ensuring one expert per GPU with no replication.

#### Case 2: E > G (Expert count > GPU count)
**Replication Strategy:**
```
Replication_factor = ceil(E/G)
GPU_assignment[i] = GPU_(i mod G) for expert_i where i ∈ [0, E-1]
```
Each GPU hosts multiple experts in a round-robin fashion, maintaining memory balance.

### 1.2 Topology-Aware Placement Algorithm
**Input Parameters:**
- B_ij: Bandwidth matrix between node i and node j
- L_ij: Latency matrix between node i and node j
- M_k: Memory capacity of GPU k
- P_t: Expected token routing probability distribution

**Optimization Objective:**
Minimize maximum link load:
```
min max_k Σ_i Σ_j (tokens_sent_ij × routing_probability_ij × link_utilization_k)
```
Subject to:
- 1-expert-per-GPU constraint
- Memory capacity constraints: Σ(expert_size) ≤ M_k for each GPU k

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism Details
For each token t:
**Gating scores:**
```
g_i(t) = softmax(W_gate × h_t)[i]
```
where h_t is the token representation and W_gate is the gating weight matrix.

**Top-K selection:**
Select top-k experts with highest gating scores:
```
TopK = {e_i | g_i(t) ≥ threshold, sorted by g_i(t) desc}
```

### 2.2 Token Batching Strategy
**Batch Formation:**
```
Batch_size = min(available_tokens, GPU_memory_limit / token_size)
```
**Token grouping by destination:**
```
For each GPU g:
    tokens_for_g = {t | destination_expert(t) ∈ GPU_g}
    batch_tokens_g = concatenate(tokens_for_g)
```

### 2.3 Dynamic Load Balancing
**Load monitoring:**
```
Load_e = tokens_processed_e / total_tokens_sent
```
**Adaptive gating adjustment:**
```
g'_i(t) = g_i(t) × (1 + α × (1 - Load_e/avg_load))
```
where α is the balancing factor (0.1-0.3 typical range).

## 3. Communication Overlap and Scheduling

### 3.1 CUDA Stream Management
**Stream Architecture:**
- Compute_stream: Primary computation stream
- Send_stream: Token transmission to remote GPUs
- Receive_stream: Token reception from remote GPUs

**Overlap Pattern:**
```
Time step t:   [Receive t+1] [Compute t] [Send t-1]
Time step t+1: [Receive t+2] [Compute t+1] [Send t]
```

### 3.2 Pipeline Scheduling Algorithm
**Multi-layer scheduling:**
```
For layer l in [1, L]:
    tokens_in_l = route(tokens_out_{l-1})
    compute_async(experts_layer_l, tokens_in_l)
    tokens_out_l = gather_results_async()
```

## 4. Memory and Model Parallelism Integration

### 4.1 Tensor Parallelism for Large Experts
**When expert_size > GPU_memory:**
```
Expert_parallelism = ceil(expert_size / GPU_memory_available)
Tensor_split_dimensions = [hidden_dim, ffn_hidden_dim]
```

### 4.2 Data Parallelism Integration
**DP group formation:**
```
DP_size = total_GPUs / (TP_size × EP_size)
```
**Synchronization:**
- All-reduce gradients across DP replicas
- Expert weights synchronized within EP group

## 5. Scalability Considerations

### 5.1 Large EP Regime Characteristics
**Threshold Definition:**
Large EP: EP_size ≥ 16
**Performance implications:**
- Communication overhead becomes O(EP_size²)
- Compute efficiency scales as O(EP_size)
- Cross-node bandwidth becomes bottleneck

### 5.2 Network Bandwidth Utilization
**Optimal bandwidth calculation:**
```
Required_bandwidth = tokens_per_second × token_size × EP_size × routing_factor
Utilization = actual_bandwidth / required_bandwidth
```
Target: ≥80% bandwidth utilization

## 6. Hardware Specifications for Deployment

### 6.1 GPU Specifications
- **Model**: H100 GPUs
- **Compute power**: 400 TFLOPS per GPU
- **Memory**: 64GB HBM3 per GPU
- **Memory bandwidth**: 1.8 TB/s per GPU
- **NVLink bandwidth**: 900 GB/s between GPUs
- **InfiniBand**: 400 Gbps between nodes

### 6.2 Performance Metrics
- **MFU (Model FLOPS Utilization)**: 60% achieved
- **Bandwidth utilization**: 80% achieved
- **Token processing rate**: Variable based on batch size and sequence length