# Phase 2: Methodology Extraction

## Core Methodology: Large-Scale Cross-Node Expert Parallelism

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Principle
- **Constraint**: Each GPU hosts at most one expert per layer
- **Mathematical Formulation**:
  - Given E experts per layer and G GPUs
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU contention between experts

#### 1.2 Cross-Node Distribution Algorithm
- **Input Parameters**:
  - Cluster topology: Node-to-node bandwidth matrix B[i,j]
  - Node-to-node latency matrix L[i,j]
  - GPU memory capacity per node M[n]
  - Expected token routing probability distribution P[e]
  - Number of experts E, number of GPUs G

- **Optimization Objective**:
  Minimize max_load = max_{i,j} (tokens_sent[i,j] / B[i,j])
  Subject to: Each GPU has ≤ 1 expert per layer

- **Placement Process**:
  1. Calculate communication cost for each expert pair
  2. Use topology-aware bin-packing to distribute experts
  3. Balance memory usage across nodes
  4. Minimize expected cross-node traffic

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE gating**: Top-K gating scores determine expert activation
- **Input**: Token representation x ∈ ℝ^8192
- **Gate computation**: g = softmax(W_g · x) where W_g ∈ ℝ^(E×8192)
- **Top-K selection**: Select top 2 experts per token based on gating scores

#### 2.2 Token Sharding Across Nodes
- **Token Batching Process**:
  1. Group tokens by destination expert ID
  2. Create batches of size B_e for each expert e
  3. Pack batches into network messages
  4. Send asynchronously using non-blocking communication

- **Batch Size Optimization**:
  - Optimal batch size B_e = min(T_e, B_max)
  - Where T_e = tokens routed to expert e, B_max = network bandwidth constraint

#### 2.3 Dynamic Load Balancing
- **Monitoring**: Track per-expert load λ_e = tokens_processed[e] / time_window
- **Adjustment**: Modify gating probabilities p_e = softmax(logits_e - α·λ_e)
- **Parameter**: α = load balancing strength (typically 0.1-0.3)

### 3. Communication Overlap and Scheduling

#### 3.1 Asynchronous Communication Pattern
- **CUDA Streams Configuration**:
  - Stream 0: Expert computation
  - Stream 1: Token sending
  - Stream 2: Token receiving

- **Overlap Timeline**:
  ```
  Time t: GPU i processes batch k
  Time t+ε: GPU i sends results of batch k-1 to GPU j
  Time t+2ε: GPU i receives batch k+1 from GPU m
  ```

#### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Layer-wise Pipeline**:
  - Layer l processes tokens as soon as partial batch arrives
  - No waiting for full batch completion from layer l-1
  - Overlap factor = number of pipeline stages

- **Token Routing Pipeline**:
  - Output tokens from layer l immediately routed to layer l+1
  - Routing decision made based on gating network at layer l+1
  - Continuous flow without synchronization barriers

### 4. Memory and Model Parallelism Integration

#### 4.1 Tensor Parallelism within Expert
- **When needed**: If expert parameters exceed single GPU memory
- **Partitioning**: Apply column-row tensor parallelism to expert MLP
  - First linear: column parallel (hidden_size → ffn_hidden_size/TP)
  - Second linear: row parallel (ffn_hidden_size/TP → hidden_size)
- **Communication**: All-reduce after each linear operation

#### 4.2 Data Parallelism Integration
- **DP replicas**: Multiple copies of entire MoE network
- **Synchronization**: All-reduce gradients across DP replicas
- **Expert consistency**: Ensure expert parameters remain synchronized

### 5. Scalability Considerations for Large EP (≥16)

#### 5.1 Network Bandwidth Requirements
- **Minimum bandwidth**: BW_min = (tokens_per_second × token_size) / num_links
- **Token size**: 8192 dimensions × 2 bytes (FP16) = 16,384 bytes per token
- **Required bandwidth**: ~7.5 GB/s per link for 450K TPS with 16 GPUs

#### 5.2 Load Distribution Metrics
- **Expert utilization**: U_e = actual_tokens_e / expected_tokens_e
- **Network balance**: σ = std_dev(tokens_per_link) / mean(tokens_per_link)
- **Target**: U_e ≈ 1.0 for all experts, σ < 0.2

### 6. Implementation Details

#### 6.1 Communication Libraries
- **Primary**: NCCL for GPU-to-GPU communication
- **Fallback**: MPI for cross-node communication
- **Optimization**: Use NCCL send/recv operations for point-to-point transfers

#### 6.2 Memory Management
- **Expert parameters**: 32768 (hidden) × 8192 (input) × 2 bytes = 512 MB per expert
- **Activation memory**: 1024 (batch) × 10000 (seq) × 8192 (dim) × 2 bytes = 160 GB total
- **Buffer allocation**: Pre-allocate communication buffers for each expert pair

#### 6.3 Synchronization Primitives
- **Barriers**: NCCL all-reduce for gradient synchronization
- **Events**: CUDA events for compute-communication synchronization
- **Streams**: Separate streams for computation, send, and receive operations