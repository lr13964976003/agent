# Phase 2: Methodology Extraction - MA Separation

## 3. MA Separation Methodology

### 3.1 Problem Formulation
- **Temporal Mismatch**: T_attention > T_moe when experts distributed across multiple GPUs
- **Complexity**: Attention O(n²d) vs MoE parallel expert execution
- **GPU allocation ratio**: 3:1 for Attention and MoE most appropriate

### 3.2 MA Separation Architecture

#### 3.2.1 Attention Parallelization Strategy
**Stage 1: Query-Key-Value Projection Parallelization**
- Input hidden states replicated across k attention GPUs
- Each GPU computes Q, K, V for subset of attention heads
- Formula: head_start = i * (num_heads / k), head_end = (i+1) * (num_heads / k)

**Stage 2: Attention Score Computation and Distribution**
- Each attention GPU computes scores for assigned heads
- All-reduce operations for information exchange
- Attention scores computation: attention_scores_i = compute_attention(Q_i, K_all, V_all)

**Stage 3: Output Aggregation and Distribution**
- Attention outputs aggregated from all GPUs
- Distributed to MoE GPUs for next phase
- Final aggregation: final_output = all_reduce(output_1, output_2, ..., output_k)

#### 3.2.2 MoE Parallelization Strategy
**Expert Distribution**
- 16 experts distributed across available GPUs
- Formula: experts_per_gpu = total_experts / num_moe_gpus
- Each GPU hosts: experts[j*experts_per_gpu : (j+1)*experts_per_gpu]

**Routing and Load Balancing**
- Gating network determines expert selection
- Top-K routing: K=2
- Synchronized attention output used for routing

**Expert Computation**
- Selected experts process assigned tokens in parallel
- Expert output collection and aggregation

### 3.3 Synchronization Mechanism
**Time Prediction Model**
- Predicts execution times for attention and MoE computations
- Based on: sequence length, hidden dimension, active experts, GPU specs, current load

**Dynamic Load Balancing**
- Adjusts attention heads and expert assignments
- Conditional logic: 
  - if predicted_T_attention > predicted_T_moe: increase_attention_parallelism()
  - elif predicted_T_moe > predicted_T_attention: adjust_expert_distribution()

**Barrier Synchronization**
- CUDA streams and events for precise synchronization
- Events: attention_complete_event, moe_complete_event
- Stream synchronization: cudaStreamWaitEvent operations

### 3.4 Communication Optimization
**Gradient Compression**
- Top-K sparsification for gradient tensors
- Quantization to reduced precision formats
- Asynchronous gradient accumulation

**Overlapping Communication and Computation**
- Communication operations overlapped with computation
- Async communication patterns with continuation of computation

**Hierarchical All-Reduce**
- Intra-node reduction first
- Inter-node reduction second
- Optimized attention output aggregation

## Model Architecture Details

### Layer Configuration
- **Number of layers**: 4
- **Hidden dimension**: 4096
- **Attention heads**: 32
- **MoE experts per layer**: 16
- **Expert hidden dimension**: 16384 (4× hidden dimension)
- **Top-K routing**: K=2
- **Sequence length**: 2048 tokens
- **Activation function**: GELU
- **Expert type**: Feed-forward network with SwiGLU activation

### MoE Configuration
- **Expert capacity factor**: 1.0
- **Load balancing loss coefficient**: 0.01
- **Router z-loss coefficient**: 0.001
- **Expert dropout**: 0.1

### Hardware Mapping
- **Total GPUs**: 16 × NVIDIA A100 80GB
- **GPU allocation**: 12 GPUs for Attention, 4 GPUs for MoE (3:1 ratio)
- **Node configuration**: 4 nodes × 4 GPUs per node
- **Interconnect**: NVLink 3.0 (600 GB/s) intra-node, InfiniBand HDR (200 Gb/s) inter-node