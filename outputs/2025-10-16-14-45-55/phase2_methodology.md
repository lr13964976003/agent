# MA Separation: Detailed Methodology

## 3. MA Separation Methodology

### 3.1 Problem Formulation
Given:
- T_attention = O(n²d) where n=sequence length, d=hidden dimension
- T_moe = time for parallel expert execution across GPUs
- Goal: T_attention ≈ T_moe for synchronized execution

### 3.2 Architecture Design

#### 3.2.1 Attention Parallelization Strategy
**Stage 1: QKV Projection Parallelization**
```
Input: Hidden states H (batch, seq_len, hidden_dim=4096)
Distributed across k=8 attention GPUs

For GPU i in [0,1,...,7]:
    head_start = i * 4  # 32 heads / 8 GPUs = 4 heads per GPU
    head_end = (i+1) * 4
    Q_i, K_i, V_i = projection_layers[head_start:head_end](H)
    # Each GPU computes 4 attention heads
    # Q_i, K_i, V_i shapes: (batch, seq_len, 4*64) = (batch, seq_len, 256)
```

**Stage 2: Attention Score Computation**
```
For GPU i in attention GPUs:
    # Compute attention for assigned heads
    attention_scores_i = compute_attention(Q_i, K_all, V_all)
    # K_all, V_all gathered via all-gather from all GPUs
    output_i = attention_scores_i @ V_all
    # output_i shape: (batch, seq_len, 256)
```

**Stage 3: Output Aggregation**
```
# Gather outputs from all attention GPUs
all_outputs = [output_0, output_1, ..., output_7]
final_output = all_reduce(all_outputs)  # Sum across GPUs
final_output shape: (batch, seq_len, 2048)  # 8*256

# Broadcast to MoE GPUs for next computation phase
broadcast_to_moe_gpus(final_output)
```

#### 3.2.2 MoE Parallelization Strategy
**Expert Distribution:**
```
Total experts: 16
MoE GPUs: 8
experts_per_gpu = 16/8 = 2 experts per GPU

Expert mapping:
- GPU 0: experts [0,1]
- GPU 1: experts [2,3]
- GPU 2: experts [4,5]
- GPU 3: experts [6,7]
- GPU 4: experts [8,9]
- GPU 5: experts [10,11]
- GPU 6: experts [12,13]
- GPU 7: experts [14,15]
```

**Token Routing:**
```
# After attention layer
attention_output shape: (batch, seq_len, 4096)

# Gating computation
gate_scores = gating_network(attention_output)  # (batch, seq_len, 16)
top_experts = top_k(gate_scores, k=2)  # Select top 2 experts per token

# Route tokens based on expert GPU mapping
For GPU j in moe_gpus:
    tokens_for_expert = extract_tokens_for_gpu_experts(top_experts, GPU_j_experts)
    expert_output = expert_computation(tokens_for_expert)
```

### 3.3 Synchronization Mechanism

#### 3.3.1 Time Prediction Model
**Input Features:**
- Sequence length: 2048 (fixed)
- Hidden dimension: 4096
- Active experts count: varies by batch
- Current GPU load metrics

**Neural Network Architecture:**
- 3 hidden layers: [64, 32, 16] neurons
- Activation: ReLU
- Output: Predicted T_attention, T_moe

#### 3.3.2 Dynamic Load Balancing
```python
def dynamic_load_balancing():
    predicted_attention_time = time_predictor.predict_attention_time()
    predicted_moe_time = time_predictor.predict_moe_time()
    
    time_diff = abs(predicted_attention_time - predicted_moe_time)
    
    if time_diff > 0.05 * max(predicted_attention_time, predicted_moe_time):
        if predicted_attention_time > predicted_moe_time:
            # Increase attention parallelism
            redistribute_attention_heads()
        else:
            # Redistribute experts
            adjust_expert_distribution()
```

#### 3.3.3 Barrier Synchronization
```cuda
// CUDA synchronization primitives
cudaEvent_t attention_complete, moe_complete;
cudaStream_t attention_stream, moe_stream, next_layer_stream;

// Record completion events
cudaEventRecord(attention_complete, attention_stream);
cudaEventRecord(moe_complete, moe_stream);

// Synchronize before next layer
cudaStreamWaitEvent(next_layer_stream, attention_complete);
cudaStreamWaitEvent(next_layer_stream, moe_complete);
```

### 3.4 Communication Optimization

#### 3.4.1 Hierarchical All-Reduce
**Three-level hierarchy:**
1. **Intra-node reduction** (4 GPUs per node)
2. **Inter-node reduction** (4 nodes)
3. **Final aggregation**

#### 3.4.2 Communication Patterns
**Attention Output Aggregation:**
```python
# Hierarchical all-reduce for 8 attention GPUs
# Step 1: Intra-node reduce (2 nodes × 4 GPUs each)
intra_node_reduce_result = all_reduce_intra_node(attention_outputs)

# Step 2: Inter-node reduce between 2 nodes
final_attention_output = all_reduce_inter_node(intra_node_reduce_result)
```

**MoE All-to-All:**
```python
# Token routing across 8 MoE GPUs
# Each GPU sends tokens to appropriate expert GPUs
all_to_all_communication(token_assignments, expert_locations)
```

### 3.5 Implementation Details

#### 3.5.1 Memory Layout
**Attention GPU Memory (8 GPUs total):**
- Model parameters: 23.1 GB
  - Attention weights: 12.8 GB
  - QKV projection weights: 8.4 GB
  - Output projection: 1.9 GB
- Activations: 18.7 GB
  - Input activations: 8.4 GB
  - Attention scores: 6.2 GB
  - Output activations: 4.1 GB

**MoE GPU Memory (8 GPUs total):**
- Expert parameters: 46.2 GB
  - 2 experts × 23.1 GB each
- Routing parameters: 0.8 GB
- Activations: Varies by token routing

#### 3.5.2 Custom CUDA Kernels
**Fused Attention Operations:**
```cuda
// Optimized attention kernel
__global__ void fused_attention_kernel(
    float* Q, float* K, float* V,
    float* output, int batch, int seq_len, int num_heads, int head_dim) {
    // Fused QK^T computation + softmax + QV multiplication
    // Uses shared memory for intermediate results
    // Optimized for A100 tensor cores
}
```

**Hierarchical All-Reduce:**
```cuda
__global__ void hierarchical_all_reduce(
    float* data, int size, int local_rank, int local_size) {
    // Two-level reduction algorithm
    // NCCL optimized for NVLink topology
}
```

#### 3.5.3 Gradient Computation
**Attention Gradients:**
```python
# Distributed gradient computation
# Each attention GPU computes gradients for its assigned heads
dQ_i, dK_i, dV_i = attention_backward(loss, attention_outputs[i])

# All-reduce gradient across GPUs
all_reduce(dQ_all, dK_all, dV_all)
```

**MoE Gradients:**
```python
# Expert gradients computed locally on each MoE GPU
# Only active experts compute gradients
for expert in active_experts:
    expert_gradients[expert] = expert_backward(loss, expert_outputs[expert])

# Gradient synchronization for shared parameters
all_reduce(shared_parameter_gradients)
```

### 3.6 Model Configuration Parameters

**Model Dimensions:**
- Hidden dimension: 4096
- Attention heads: 32 (4 heads per GPU × 8 GPUs)
- Head dimension: 64 (4096/64)
- Sequence length: 2048
- Batch size: 1024 sequences
- Expert hidden dimension: 16384 (4× hidden dim)

**Parallel Configuration:**
- Attention GPUs: 8 (GPUs 0-7)
- MoE GPUs: 8 (GPUs 8-15)
- Experts per GPU: 2
- Attention replication: 2× (for fault tolerance)
- Synchronization interval: 100 iterations

**Communication Parameters:**
- NCCL version: 2.15
- All-reduce algorithm: Ring + Tree hybrid
- Compression: 8-bit quantization for gradients
- Overlap factor: 0.75 (75% overlap between compute and comm)