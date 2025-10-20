# Phase 2: Detailed Methodology - Large-Scale Cross-Node Expert Parallelism

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
**Mathematical Formulation:**
- Let E = number of experts per layer (E = 16 in experiments)
- Let G = number of available GPUs (G = 16 in experiments)
- Placement constraint: ∀g ∈ [1,G], expert_count(g) ≤ 1

**Placement Algorithm:**
```
function place_experts(E, G, topology):
    # topology: dict with node_bandwidth[node_i][node_j] and gpu_memory[node]
    placement = {}
    
    # Step 1: Create expert-GPU mapping
    for expert_id in range(E):
        gpu_id = expert_id % G
        placement[expert_id] = gpu_id
    
    # Step 2: Topology-aware optimization
    # Objective: minimize max(tokens sent across any single link)
    link_loads = calculate_link_loads(placement, expected_token_routing)
    
    while max(link_loads) > threshold:
        # Swap experts between GPUs to balance link loads
        expert_to_swap = find_expert_for_swap(link_loads)
        new_placement = swap_experts(placement, expert_to_swap)
        if max(calculate_link_loads(new_placement)) < max(link_loads):
            placement = new_placement
            link_loads = calculate_link_loads(placement)
    
    return placement
```

### 1.2 Cross-Node Distribution
**Network Topology Parameters:**
- NVLink bandwidth: 400 GB/s bidirectional
- InfiniBand bandwidth: 200 GB/s per link
- Intra-node latency: 5-10 μs
- Inter-node latency: 1-5 μs

**Memory Requirements per GPU:**
- Expert parameters: 32,768 (hidden) × 8,192 (input) × 2 bytes (FP16) = 537 MB
- Expert output: 8,192 × 8,192 × 2 bytes = 131 MB
- Activation buffers: 1024 × 10,000 × 8,192 × 2 bytes = 160 GB (shared across all layers)
- Total per GPU: ~537 MB + ~131 MB + shared buffers = ~668 MB per expert

## 2. Routing and Load Balancing

### 2.1 Token Sharding Algorithm
**Input:** Batch of tokens X ∈ ℝ^(batch_size × seq_len × hidden_dim)
**Output:** Expert-specific token batches

**Mathematical Formulation:**
```
for each token t in batch:
    # Top-K gating scores
    g_i(t) = softmax(W_gate · t)[i] for i ∈ {1,...,E}
    
    # Dynamic load balancing
    load_e = (tokens routed to expert e) / expert_capacity
    adjustment_factor = max(0.1, 1.0 - load_e * 0.5)
    
    adjusted_g_i(t) = g_i(t) * adjustment_factor
    
    # Select top-K experts
    selected_experts = top_k(adjusted_g_i(t), k=2)
    
    # Route tokens
    for expert_id in selected_experts:
        add_to_expert_batch(expert_id, token_data)
```

### 2.2 Asynchronous Communication
**CUDA Stream Configuration:**
- Stream 0: Compute (expert computation)
- Stream 1: Communication (token sending)
- Stream 2: Communication (token receiving)

**NCCL Operations:**
```
# Send tokens to expert on GPU i
ncclSend(token_batch, count, ncclFloat16, dest_gpu, ncclComm, compute_stream)

# Receive tokens from other GPUs
ncclRecv(token_batch, count, ncclFloat16, src_gpu, ncclComm, compute_stream)
```

## 3. Communication Overlap and Scheduling

### 3.1 Pipeline Scheduling Details
**Layer-wise Processing:**
```
for layer in [1, 2, 3, 4]:
    # Step 1: Route tokens to layer's experts
    route_tokens_to_layer(layer)
    
    # Step 2: Start receiving next layer's tokens while computing current
    if layer < 4:
        start_async_recv(layer + 1)
    
    # Step 3: Process current layer
    process_experts_layer(layer)
    
    # Step 4: Send results to next layer
    if layer < 4:
        start_async_send(layer, layer + 1)
```

### 3.2 Token Batching Optimization
**Batch Size Calculations:**
- Optimal batch size per expert: 1024 sequences × 10,000 tokens ÷ 16 experts = 640,000 tokens per expert
- Network message size: 640,000 tokens × 8,192 dimensions × 2 bytes = 10.48 GB per expert

### 3.3 Load Balancing Parameters
- **Update frequency**: Every 100 tokens
- **Threshold for rebalancing**: When load_e > 1.2 × average_load
- **Gating probability adjustment**: ±10% per update cycle
- **Expert capacity**: 1.2 × average_load

## 4. Memory and Model Parallelism Integration

### 4.1 Tensor Parallelism within Experts
For experts exceeding single-GPU memory:
- **Column-parallel first linear**: Split hidden dimension (32,768) across 2 GPUs → 16,384 each
- **Row-parallel second linear**: Split hidden dimension (32,768) across 2 GPUs → 16,384 each

### 4.2 Data Parallelism Configuration
- **DP degree**: 1 (inference-only setting)
- **Synchronization**: All-reduce across DP replicas every step (not used in inference)

## 5. Scalability Framework Implementation

### 5.1 Large EP Regime Optimization (EP ≥ 16)
**Network Bandwidth Calculations:**
- Total communication per layer: 1024 × 10,000 × 8,192 × 2 bytes = 167.8 GB
- With 16 experts: 167.8 GB ÷ 16 = 10.49 GB per expert per layer
- Communication time: 10.49 GB ÷ 400 GB/s = 26.2 ms (NVLink)

### 5.2 Compute vs Communication Balance
**Compute Time per Expert:**
- MLP forward pass: ~5ms (H100 with 8192×32768×8192 matrix multiplication)
- Communication overlap efficiency: 26.2ms ÷ 5ms = 5.24× (communication dominates)

**Optimization:** Overlap communication across multiple layers to hide latency.