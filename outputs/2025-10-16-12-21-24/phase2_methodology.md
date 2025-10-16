# Phase 2: Methodology Extraction - MA Separation

## 3. MA Separation Methodology

### 3.1 Problem Formulation
- **Temporal mismatch**: T_attention > T_moe due to attention sequential nature vs MoE parallel execution
- **Goal**: Achieve T_attention ≈ T_moe through attention parallelization

### 3.2 MA Separation Architecture

#### 3.2.1 Attention Parallelization Strategy (3-Stage)

**Stage 1: Query-Key-Value Projection Parallelization**
- **Replication**: Input hidden states replicated across k attention GPUs
- **Partitioning**: Each GPU computes Q,K,V for subset of attention heads
- **Formula**: For GPU i in attention GPUs:
  - head_start = i × (num_heads / k)
  - head_end = (i+1) × (num_heads / k)
  - Q_i, K_i, V_i = projection_layers[head_start:head_end](input)

**Stage 2: Attention Score Computation and Distribution**
- **Cross-GPU communication**: All-reduce operations for K,V across all attention GPUs
- **Computation**: Each GPU computes attention for assigned heads
- **Output**: Distributed attention scores per head subset

**Stage 3: Output Aggregation and Distribution**
- **Aggregation**: All-reduce across all attention GPUs
- **Distribution**: Broadcast final attention output to MoE GPUs
- **Formula**: final_output = all_reduce(output_1, output_2, ..., output_k)

#### 3.2.2 MoE Parallelization Strategy

**Expert Distribution:**
- **Total experts**: 16 distributed across available GPUs
- **Experts per GPU**: total_experts / num_moe_gpus
- **Unique assignment**: Each expert hosted on exactly one GPU
- **Formula**: For GPU j in moe GPUs:
  - hosted_experts = experts[j×experts_per_gpu : (j+1)×experts_per_gpu]

**Routing and Load Balancing:**
- **Gating network**: Determines expert selection based on attention output
- **Top-K routing**: K=2 experts per token
- **Dynamic assignment**: Real-time expert utilization monitoring
- **Route tokens**: to_experts(tokens, top_experts)

**Expert Computation:**
- **Parallel execution**: Selected experts process tokens simultaneously
- **Formula**: For expert in active_experts:
  - expert_output[expert] = expert_computation(tokens_for_expert[expert])

### 3.3 Synchronization Mechanism

**Time Prediction Model:**
- **Inputs**: Sequence length, hidden dimension, active experts, GPU specs
- **Output**: Predicted execution times for attention and MoE
- **Model**: Neural network with 3 hidden layers

**Dynamic Load Balancing:**
- **Trigger**: When predicted_T_attention > predicted_T_moe + 5% threshold
- **Actions**: 
  - Increase attention parallelism (more GPUs for attention)
  - Adjust expert distribution (rebalance across MoE GPUs)
- **Algorithm**: Real-time monitoring every 100 iterations

**Barrier Synchronization:**
- **CUDA streams**: Separate streams for attention and MoE
- **Events**: cudaEventRecord for completion notification
- **Synchronization**: cudaStreamWaitEvent ensures both complete before next layer

### 3.4 Communication Optimization

**Gradient Compression:**
- **Techniques**: Top-K sparsification, quantization to 8-bit, async accumulation
- **Target**: Reduce communication overhead for gradients

**Overlapping Communication and Computation:**
- **Strategy**: Issue async communication during computation
- **Implementation**: Non-blocking all-reduce operations with computation overlap

**Hierarchical All-Reduce:**
- **Intra-node**: Reduce within each node first (NVLink)
- **Inter-node**: Reduce across nodes (InfiniBand)
- **Benefit**: Minimizes expensive inter-node communication

## Critical Parameters for Deployment

### Model Architecture Parameters
- **Layers**: 4
- **Hidden dimension**: 4096
- **Attention heads**: 32
- **Experts per layer**: 16
- **Expert hidden dimension**: 16384
- **Top-K routing**: K=2
- **Sequence length**: 2048 tokens

### Hardware Configuration Parameters
- **Total GPUs**: 16
- **GPU type**: A100 80GB
- **Memory per GPU**: 80GB HBM2e
- **Interconnect**: NVLink 3.0 (600 GB/s) + InfiniBand HDR (200 Gb/s)
- **Node topology**: 4 nodes × 4 GPUs per node

### MA Separation Configuration Parameters
- **Attention GPUs**: 8 (out of 16)
- **Attention heads per GPU**: 4
- **MoE GPUs**: 8 (out of 16)
- **Experts per GPU**: 2
- **Synchronization interval**: 100 iterations
- **Load balancing threshold**: 5%
- **Communication**: 8-bit quantization for gradients

### Communication Patterns
- **Attention All-Reduce**: Hierarchical pattern (intra-node then inter-node)
- **MoE All-to-All**: Direct expert-to-expert communication
- **Gradient Synchronization**: Compressed all-reduce across all GPUs