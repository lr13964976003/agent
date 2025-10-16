# DAG Runtime Analysis: Optimized Helix Model

## Overview
This analysis examines the runtime of the optimized helix model DAG based on matrix multiplication operations and parallel execution patterns. The model uses pipeline parallelism with two stages and tensor parallelism within each stage.

## Model Configuration
- **Batch Size**: 1024
- **Sequence Length**: 10000
- **Embedding Dimension**: 8192
- **Hidden Dimension**: 8192 (for MLP layers)
- **Number of Attention Heads**: 32 total (8 per head group)
- **Head Dimension**: 128 for Q/K, 128 for V
- **Pipeline Stages**: 2 (Stage 0: GPUs 0-7, Stage 1: GPUs 8-15)

## Matrix Multiplication Operations Analysis

### 1. MHA Layer Operations

#### QKV Projections (Parallel across head groups)
Each MHA layer has 3 separate matrix multiplications for Q, K, V projections:
- **Q Linear**: [batch_size×seq_len, embed_dim] × [embed_dim, heads×d_k]
  - Matrix multiplication: [1024×10000, 8192] × [8192, 32×128] = [1024×10000, 4096]
  - Since heads=32 and d_k=128, total output dimension = 4096
  - Parallelized across 4 head groups: Each group computes 8 heads
  - **Operation**: Get_Time(1024×10000, 8192, 4096)

- **K Linear**: Same dimensions as Q Linear
  - **Operation**: Get_Time(1024×10000, 8192, 4096)

- **V Linear**: [batch_size×seq_len, embed_dim] × [embed_dim, heads×d_v]
  - Matrix multiplication: [1024×10000, 8192] × [8192, 32×128] = [1024×10000, 4096]
  - **Operation**: Get_Time(1024×10000, 8192, 4096)

#### Attention Computation
For each head group (8 heads per group, 4 groups total):
- **Query-Key multiplication**: [batch_size, heads, seq_len, d_k] × [batch_size, heads, d_k, seq_len]
  - Matrix multiplication: [1024, 8, 10000, 128] × [1024, 8, 128, 10000] = [1024, 8, 10000, 10000]
  - **Operation**: Get_Time(1024×8×10000, 128, 10000)

- **Softmax and Dropout**: No matrix multiplication

- **Attention-Value multiplication**: [batch_size, heads, seq_len, seq_len] × [batch_size, heads, seq_len, d_v]
  - Matrix multiplication: [1024, 8, 10000, 10000] × [1024, 8, 10000, 128] = [1024, 8, 10000, 128]
  - **Operation**: Get_Time(1024×8×10000, 10000, 128)

#### Output Projection
- **Output Linear**: [batch_size×seq_len, total_head_dim] × [total_head_dim, embed_dim]
  - Matrix multiplication: [1024×10000, 4096] × [4096, 8192] = [1024×10000, 8192]
  - **Operation**: Get_Time(1024×10000, 4096, 8192)

### 2. MLP Layer Operations

#### FC1 Layer (Grouped Column Parallel)
- **FC1 Linear**: [batch_size×seq_len, embed_dim] × [embed_dim, hidden_dim]
  - Matrix multiplication: [1024×10000, 8192] × [8192, 8192] = [1024×10000, 8192]
  - Parallelized across 2 tensor groups (each group handles 4 GPUs)
  - **Operation**: Get_Time(1024×10000, 8192, 8192)

#### GELU Activation
- No matrix multiplication operation

#### FC2 Layer (Grouped Row Parallel)
- **FC2 Linear**: [batch_size×seq_len, hidden_dim] × [hidden_dim, embed_dim]
  - Matrix multiplication: [1024×10000, 8192] × [8192, 8192] = [1024×10000, 8192]
  - Output split across groups, then concatenated
  - **Operation**: Get_Time(1024×10000, 8192, 8192)

## Parallel Execution Analysis

### Parallel Computing Patterns

1. **Head Group Parallelism (MHA layers)**:
   - 4 head groups operate in parallel
   - Each group computes 8 attention heads simultaneously
   - **No duplication**: All groups execute concurrently

2. **Tensor Group Parallelism (MLP layers)**:
   - 2 tensor groups operate in parallel
   - Each group uses 4 GPUs for tensor parallelism
   - **No duplication**: Both groups execute concurrently

3. **Pipeline Stage Parallelism**:
   - Stage 0 (GPUs 0-7) and Stage 1 (GPUs 8-15) operate in pipeline fashion
   - While Stage 1 processes batch N, Stage 0 processes batch N+1
   - **Serial dependency**: Stage 1 must wait for Stage 0 completion per batch

## Longest Path Identification

### Critical Path Through the DAG
The longest path (critical path) is:

```
model_input → stage0 → stage0_to_stage1 → stage1 → model_output
```

Breaking down stage0 (longer critical path):

#### Stage 0 Critical Path (GPU 0-7):
```
mha_input → ln → q_hg0_g0 → attn_hg0_g0 → concat_g0 → final_concat → output_proj → residual → mha_output
                                                                                                   ↓
mlp_input → ln → input_split → fc1_tg0 → gelu_tg0 → fc2_split_tg0 → all_reduce_tg0 → final_concat → residual → mlp_output
```

#### Stage 1 Critical Path (GPU 8-15):
Same structure as Stage 0, but with GPUs 8-15

### Matrix Multiplication Sequence on Critical Path

For each stage, the critical path involves these sequential matrix multiplications:

1. **Q Linear**: Get_Time(1024×10000, 8192, 1024)
2. **K Linear**: Get_Time(1024×10000, 8192, 1024) 
3. **V Linear**: Get_Time(1024×10000, 8192, 1024)
4. **Attention Q×K**: Get_Time(1024×8×10000, 128, 10000)
5. **Attention ×V**: Get_Time(1024×8×10000, 10000, 128)
6. **Output Projection**: Get_Time(1024×10000, 1024, 8192)
7. **FC1**: Get_Time(1024×10000, 8192, 8192)
8. **FC2**: Get_Time(1024×10000, 8192, 8192)

Note: Dimensions adjusted for head group distribution (4096/4 = 1024 per head group)

## Runtime Calculation Summary

### Total Runtime per Stage
The runtime for each stage is determined by the critical path through sequential matrix multiplications. All parallel operations within a stage do not add to the critical path runtime.

**Stage 0 Runtime** = Get_Time(1024×10000, 8192, 1024) [Q] + Get_Time(1024×10000, 8192, 1024) [K] + Get_Time(1024×10000, 8192, 1024) [V] + Get_Time(1024×8×10000, 128, 10000) [Q×K] + Get_Time(1024×8×10000, 10000, 128) [×V] + Get_Time(1024×10000, 1024, 8192) [Output] + Get_Time(1024×10000, 8192, 8192) [FC1] + Get_Time(1024×10000, 8192, 8192) [FC2]

**Stage 1 Runtime** = Same as Stage 0

### Pipeline Overhead
Additional Get_Time for pipeline communication: Get_Time(1024×10000, 8192, 8192) [data transfer from GPU 7→8]

### Final Runtime Expression
**Total DAG Runtime** = Stage0_runtime + Pipeline_communication + Stage1_runtime

Where:
- Stage0_runtime is the sum of all matrix multiplications on Stage 0 critical path
- Pipeline_communication is the data transfer time between stages  
- Stage1_runtime is the sum of all matrix multiplications on Stage 1 critical path

## Key Observations

1. **Parallel Efficiency**: All head groups and tensor groups execute simultaneously, so only the critical path contributes to runtime
2. **Pipeline Efficiency**: The pipeline allows overlapping of stages across different batches
3. **No Duplication**: Matrix multiplication times are not duplicated for parallel operations
4. **Longest Path**: The critical path goes through sequential QKV projections → attention → output projection → FC1 → FC2 for each stage