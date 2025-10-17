# Phase 2: Methodology Extraction - MA Separation

## 3. MA Separation Methodology

### 3.1 Problem Formulation
- **Time Variables**: T_attention vs T_moe
- **Complexity**: O(n²d) for attention (n=sequence length, d=hidden dimension)
- **Mismatch**: T_attention > T_moe when experts distributed
- **Solution Goal**: T_attention ≈ T_moe through attention parallelization

### 3.2 MA Separation Architecture

#### 3.2.1 Attention Parallelization Strategy - Three Stages

**Stage 1: Query-Key-Value Projection Parallelization**
- **GPU Count**: k attention GPUs (12 in experiments)
- **Head Distribution**: `head_start = i * (num_heads / k)`, `head_end = (i+1) * (num_heads / k)`
- **Input**: Hidden states replicated across k GPUs
- **Computation**: Q_i, K_i, V_i = projection_layers[head_start:head_end](input)

**Stage 2: Attention Score Computation and Distribution**
- **Operation**: Each GPU computes for assigned heads
- **Communication**: All-reduce operations for information exchange
- **Formula**: attention_scores_i = compute_attention(Q_i, K_all, V_all)
- **Output**: output_i = attention_scores_i @ V_all

**Stage 3: Output Aggregation and Distribution**
- **Aggregation**: final_output = all_reduce(output_1, output_2, ..., output_k)
- **Distribution**: broadcast_to_moe_gpus(final_output)

#### 3.2.2 MoE Parallelization Strategy

**Expert Distribution Parameters**
- **Total Experts**: 16 experts per layer
- **MoE GPUs**: 4 GPUs dedicated to MoE (from 16 total)
- **Experts per GPU**: 16/4 = 4 experts per GPU
- **Distribution**: `experts[j*4 : (j+1)*4]` for GPU j

**Routing and Load Balancing**
- **Gating Network**: Determines expert selection from attention output
- **Top-K Routing**: K=2 (selects top 2 experts per token)
- **Formula**: gate_scores = gating_network(attention_output)
- **Token Routing**: route_tokens_to_experts(tokens, top_experts)

**Expert Computation**
- **Parallel Processing**: All selected experts process tokens simultaneously
- **Distribution**: expert_output[expert] = expert_computation(tokens_for_expert[expert])

### 3.3 Synchronization Mechanism

**Time Prediction Model Parameters**
- **Inputs**: Sequence length, hidden dimension, active experts, GPU specs, current load
- **Prediction**: Execution times for attention and MoE computations

**Dynamic Load Balancing Algorithm**
```
if predicted_T_attention > predicted_T_moe:
    increase_attention_parallelism()
elif predicted_T_moe > predicted_T_attention:
    adjust_expert_distribution()
```

**Barrier Synchronization Implementation**
- **Mechanism**: CUDA streams and events
- **Key Operations**:
  - cudaEventRecord(attention_complete_event, attention_stream)
  - cudaEventRecord(moe_complete_event, moe_stream)
  - cudaStreamWaitEvent(next_layer_stream, attention_complete_event)
  - cudaStreamWaitEvent(next_layer_stream, moe_complete_event)

### 3.4 Communication Optimization

**Gradient Compression Techniques**
- **Top-K Sparsification**: Applied to gradient tensors
- **Quantization**: Reduced precision formats
- **Asynchronous Accumulation**: Gradient accumulation

**Overlapping Communication-Computation**
```
while computation_not_complete:
    issue_async_communication()
    continue_computation()
    wait_for_communication()
```

**Hierarchical All-Reduce**
- **Two-Stage Process**:
  1. Intra-node reduction: `intra_node_reduce(attention_outputs)`
  2. Inter-node reduction: `inter_node_reduce(partial_results)`

## Model Architecture Parameters for DAG Generation

### Layer Specifications
- **Number of Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32 (distributed across 12 GPUs → 32/12 ≈ 2.67 heads per GPU)
- **MoE Experts**: 16 per layer
- **Expert Hidden Dimension**: 16384 (4× hidden dimension)
- **FFN Inner Dimension**: 16384 (standard MoE configuration)

### Attention Configuration
- **Head Dimension**: 4096/32 = 128 per head
- **Attention Dropout**: 0.1
- **Attention Type**: Multi-head self-attention
- **Position Encoding**: Rotary Position Embedding (RoPE)

### MoE Configuration
- **Router Type**: Top-K gating
- **Top-K Value**: K=2
- **Expert Capacity Factor**: 1.0
- **Auxiliary Loss**: Load balancing (0.01) + Router z-loss (0.001)
- **Expert Dropout**: 0.1
- **Expert Activation**: SwiGLU

### Critical Parameters for Device Mapping
- **Attention GPUs**: 12 devices (GPUs 0-11)
- **MoE GPUs**: 4 devices (GPUs 12-15)
- **Expert Distribution**: 4 experts per MoE GPU
- **Head Distribution**: 32 heads across 12 GPUs → non-uniform distribution
- **Sequence Processing**: 2048 tokens per sequence
- **Batch Processing**: 1024 sequences per batch (2M tokens total)

### Synchronization Points
- **Layer-wise Synchronization**: After each layer completion
- **Attention-MoE Sync**: Explicit barrier between attention and MoE phases
- **Gradient Sync**: End of backward pass for all parameters
- **Expert Routing Sync**: After token assignment to experts