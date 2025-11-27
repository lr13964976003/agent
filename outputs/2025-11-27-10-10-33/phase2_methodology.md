# Phase 2: Methodology Extraction

## Method Overview

The proposed method is a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models that maximizes computational parallelism by deploying at most one expert per GPU. The methodology consists of three key components: expert placement strategy, routing and load balancing, and communication overlap and scheduling.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Allocation Rule**: Each GPU hosts exactly one expert per MoE layer
- **Mathematical Formulation**: For E experts in a layer and G available GPUs:
  - If E ≤ G: Each expert assigned to a unique GPU
  - If E > G: Experts are replicated across GPUs while maintaining one expert per GPU
- **Resource Utilization**: Ensures each expert operates without intra-GPU contention

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement**: Considers three parameters:
  - Node-to-node bandwidth matrix B[i][j] ∈ ℝ^(N×N) where N is number of nodes
  - GPU memory capacity per node M[n] ∈ ℝ^N
  - Expected token routing probability matrix P[i][j] ∈ ℝ^(E×E)
- **Optimization Objective**: Minimize Σ(i,j) (tokens_sent[i][j] / B[i][j])
- **Placement Algorithm**: Greedy assignment minimizing maximum link utilization

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K Routing**: For each token, select top K experts by gating score
- **Gating Function**: g(x) = softmax(W_gate · x + b_gate) ∈ ℝ^E
- **Expert Selection**: Select experts e_1, e_2, ..., e_K with highest gating scores

### 2.2 Token Sharding Across Nodes
- **Token Batching Strategy**:
  - Group tokens by destination GPU based on expert assignment
  - Batch size per GPU: min(batch_size_tokens, max_batch_per_expert)
- **Asynchronous Communication Protocol**:
  - Sender node: Pre-batch tokens by destination
  - Receiver node: Post buffers for incoming tokens
  - Overlap: Send batch i while computing batch i-1

### 2.3 Load Balancing Algorithm
- **Dynamic Gating Adjustment**:
  - Monitor load L_e for each expert e over time window T
  - Adjust gating probabilities: g'_e = g_e · (L_avg / L_e)^α
  - Where α = 0.1 (balance factor), L_avg = (Σ_e L_e) / E
- **Rebalancing Frequency**: Every 1000 tokens processed

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **CUDA Stream Architecture**:
  - Stream 0: Expert computation
  - Stream 1: Token data transfer (NCCL)
  - Stream 2: Gradient synchronization (if training)
- **Synchronization Points**:
  - Event A: Computation of batch i starts
  - Event B: Transfer of batch i+1 completes
  - Wait condition: Event B must complete before Event A finishes

### 3.2 Pipeline Scheduling for Multi-Layer Networks
- **Layer-wise Pipeline**:
  - Layer n processes tokens while layer n+1 receives tokens
  - Pipeline depth = number of MoE layers = 58
- **Token Flow Model**:
  - Token t enters layer 1 at time τ
  - Token t enters layer k at time τ + (k-1) · t_layer
  - Where t_layer = t_compute + t_communication - t_overlap

## 4. Scalability Considerations

### 4.1 Large EP Regime Definition
- **Minimum EP**: EP ≥ 16
- **Network Requirements**: 
  - Inter-node bandwidth ≥ 100 Gbps
  - Latency ≤ 10 μs for optimal overlap
- **Compute Saturation**: Achieved when t_compute ≥ 5 × t_communication

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism within Expert**:
  - Applied when expert size > GPU memory
  - Partition MLP weights using column-row parallel strategy
  - Dimensions: [18432, 7168] → [9216, 7168] on two devices
- **Data Parallelism**:
  - Replicas R = total_GPUs / (experts_per_layer × layers)
  - All-reduce gradients across replicas every micro-batch

## 5. Implementation Parameters

### 5.1 Hardware Configuration
- **GPU**: H100 (64GB HBM3)
- **Compute**: 400 TFLOPS (FP16/BF16)
- **Memory Bandwidth**: 1.8 TB/s
- **Network**: NVLink/NVSwitch 4.0 (900 GB/s), InfiniBand HDR (200 Gbps)

### 5.2 Model Parameters
- **Model**: 61-layer transformer
- **MoE Layers**: 58 (layers 4-61)
- **Expert Count**: Configurable (E=16, 32, 64, 128)
- **Hidden Size**: 7168
- **MLP Hidden Size**: 18432
- **Attention Heads**: 128 (56 dimensions each)
- **Precision**: BF16 throughout

### 5.3 Deployment Constraints
- **Memory Budget per GPU**:
  - Expert weights: 18432 × 7168 × 2 bytes = 264 MB
  - Activation buffer: batch_size × seq_len × 7168 × 2 bytes
  - KV cache: batch_size × seq_len × 128 × 56 × 2 bytes
  - Total: ≤ 60GB (leaving 4GB for overhead)

### 5.4 Optimization Parameters
- **Batch Sizes**: 
  - Micro-batch: 1024 tokens
  - Pipeline batch: 8192 tokens
- **Sequence Length**: Up to 8192 tokens
- **Communication**:
  - NCCL algorithm: Ring for small messages, Tree for large
  - Buffer size: 64MB per peer connection
  - Compression: None (BF16 already efficient)

## 6. Performance Model

### 6.1 Throughput Calculation
- **Compute Time**: t_compute = (ops_per_token) / (400 TF × 0.6 MFU)
- **Communication Time**: t_comm = (bytes_per_token) / (1.8 TB/s × 0.8 utilization)
- **Total Time**: t_total = max(t_compute, t_communication - t_overlap)

### 6.2 Scaling Laws
- **Strong Scaling**: Throughput ∝ min(GPUs, experts) up to network limits
- **Weak Scaling**: Throughput/GPU remains constant with proper load balance
- **Communication Lower Bound**: t_comm ≥ (tokens × hidden_dim × 2) / network_bandwidth