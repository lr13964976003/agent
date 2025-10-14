# DAG Runtime Analysis Report

## Overview
This report analyzes the runtime performance of the transformer model DAG with tensor parallelism and pipeline parallelism.

## Model Configuration
- **Total Layers**: 16 transformer layers
- **Hidden Size**: 8192
- **Attention Heads**: 16 heads, 512 dimensions per head
- **FFN Hidden Size**: 32768
- **Batch Size**: 1024
- **Sequence Length**: 10000
- **Parallelism**: 2 pipeline stages × 8-way tensor parallelism

## Longest Path Analysis

The critical path through the DAG follows this sequence:
```
input → broadcast_0 → layer0_ln1 → layer0_qkv_split → layer0_q_linear → layer0_attn_score → layer0_attn_weights → layer0_attn_out → layer0_attn_gather → layer0_attn_proj_split → layer0_attn_proj → layer0_attn_allreduce → layer0_res1 → layer0_ln2 → layer0_mlp1_split → layer0_mlp1 → layer0_gelu → layer0_mlp2 → layer0_mlp_allreduce → layer0_res2 → send_layer0_to_layer1 → layer1_ln1 → layer1_qkv → layer1_attn → layer1_mlp → send_stage0_to_stage1 → receive_stage1 → layer8 → layer9 → layer10 → layer11 → layer12 → layer13 → layer14 → layer15 → output
```

## Matrix Multiplication Operations

### Per-Layer Matrix Multiplications (16 layers total)

#### Attention Component (6 matrix multiplications)
1. **Query Linear Projection**
   - Dimensions: [batch_size×seq_len, hidden_size/8] × [hidden_size/8, hidden_size/8]
   - Get_Time(1024×10000, 1024, 1024) = Get_Time(10240000, 1024, 1024)
   - Parallel across 8 GPUs (column parallel)

2. **Key Linear Projection**
   - Dimensions: [batch_size×seq_len, hidden_size/8] × [hidden_size/8, hidden_size/8]
   - Get_Time(10240000, 1024, 1024)
   - Parallel across 8 GPUs (column parallel)

3. **Value Linear Projection**
   - Dimensions: [batch_size×seq_len, hidden_size/8] × [hidden_size/8, hidden_size/8]
   - Get_Time(10240000, 1024, 1024)
   - Parallel across 8 GPUs (column parallel)

4. **Attention Score Q·K^T**
   - Dimensions: [batch_size×num_heads, seq_len, head_dim] × [batch_size×num_heads, head_dim, seq_len]
   - Get_Time(1024×2×10000, 512, 10000) = Get_Time(20480000, 512, 10000)
   - Parallel across 8 GPUs (heads distributed)

5. **Attention Output (Softmax·V)**
   - Dimensions: [batch_size×num_heads, seq_len, seq_len] × [batch_size×num_heads, seq_len, head_dim]
   - Get_Time(1024×2×10000, 10000, 512) = Get_Time(20480000, 10000, 512)
   - Parallel across 8 GPUs (heads distributed)

6. **Output Projection**
   - Dimensions: [batch_size×seq_len, hidden_size/8] × [hidden_size/8, hidden_size/8]
   - Get_Time(10240000, 1024, 1024)
   - Parallel across 8 GPUs (row parallel)

#### MLP Component (2 matrix multiplications)
7. **MLP Linear1 (Up-projection)**
   - Dimensions: [batch_size×seq_len, hidden_size/8] × [hidden_size/8, ffn_hidden_size/8]
   - Get_Time(10240000, 1024, 4096) = Get_Time(10240000, 1024, 4096)
   - Parallel across 8 GPUs (column parallel)

8. **MLP Linear2 (Down-projection)**
   - Dimensions: [batch_size×seq_len, ffn_hidden_size/8] × [ffn_hidden_size/8, hidden_size/8]
   - Get_Time(10240000, 4096, 1024)
   - Parallel across 8 GPUs (row parallel)

## Parallelism Analysis

### Tensor Parallelism (8-way)
- Each matrix multiplication is split across 8 GPUs
- Column parallel: Weight matrices split along output dimension
- Row parallel: Weight matrices split along input dimension
- All-reduce operations required for synchronization

### Pipeline Parallelism (2 stages)
- Stage 0: Layers 0-7 on GPUs 0-7
- Stage 1: Layers 8-15 on GPUs 8-15
- Pipeline communication between stages adds overhead

## Runtime Calculation

### Per-Layer Runtime (parallel across 8 GPUs)
The runtime for each layer is determined by the slowest matrix multiplication:

```
Layer_Runtime = max(
    Get_Time(10240000, 1024, 1024),    // Q/K/V projections
    Get_Time(20480000, 512, 10000),    // Q·K^T
    Get_Time(20480000, 10000, 512),    // Attention·V
    Get_Time(10240000, 1024, 4096),    // MLP up-projection
    Get_Time(10240000, 4096, 1024)     // MLP down-projection
)
```

### Total DAG Runtime
```
Total_Runtime = 16 × Layer_Runtime + Pipeline_Overhead
```

Where:
- **16** layers execute sequentially due to data dependencies
- **Pipeline_Overhead** includes communication between stages
- All matrix multiplications within each layer run in parallel across 8 GPUs

## Key Observations

1. **Longest Path**: 16 sequential transformer layers
2. **Parallelism**: 8-way tensor parallelism within each layer
3. **Bottleneck**: The largest matrix multiplication determines per-layer runtime
4. **No Duplication**: Matrix multiplications are not duplicated due to parallel execution
5. **Communication**: All-reduce operations between tensor parallel groups

## Expected Performance Characteristics

- **Memory Efficiency**: Reduced per-GPU memory due to tensor parallelism
- **Compute Efficiency**: Near-linear scaling with tensor parallelism
- **Communication Overhead**: All-reduce operations for tensor parallelism synchronization
- **Pipeline Efficiency**: Pipeline bubble overhead between stages