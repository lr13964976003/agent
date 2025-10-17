# DAG Runtime Analysis Report

## Executive Summary

This report provides a detailed analysis of the matrix multiplication operations and longest computation paths for both the baseline tensor-parallel pipeline (TP-PP) configuration and the MA-separation configuration. The analysis is based on the provided DAG structures and model configurations.

## Model Configuration Summary

From the deployment configuration:
- **Model**: 4-layer transformer with MoE layers
- **Hidden dimension**: 4096
- **Attention heads**: 32
- **Head dimension**: 128
- **Sequence length**: 2048
- **Batch size**: 1024
- **Vocabulary size**: 50265
- **MoE experts**: 16 total (4 per GPU in MA-separation)
- **MoE expert hidden dimension**: 16384

## Baseline TP-PP Configuration Analysis

### Matrix Multiplication Operations

#### 1. Token Embedding
- **Operation**: Embedding lookup → Dense matrix multiplication
- **Shape**: `[batch_size × seq_len, vocab_size] × [vocab_size, hidden_dim]`
- **Dimensions**: `m=1024×2048=2,097,152`, `k=50265`, `n=4096`
- **Get_Time**: Get_Time(2097152, 50265, 4096)

#### 2. QKV Projection (per layer, per stage)
- **Operation**: Three separate matrix multiplications for Q, K, V
- **Shape**: `[batch_size × seq_len, hidden_dim] × [hidden_dim, hidden_dim]`
- **Dimensions**: `m=2,097,152`, `k=4096`, `n=4096`
- **Tensor Parallel Split**: 8-way TP, so per GPU: `m=2,097,152`, `k=512`, `n=4096`
- **Get_Time per GPU**: Get_Time(2097152, 512, 4096)
- **Parallel**: All 8 GPUs execute simultaneously

#### 3. Attention Scores
- **Operation**: Q × K^T
- **Shape**: `[batch_size × seq_len, head_dim] × [head_dim, seq_len]`
- **Per head**: `m=2,097,152`, `k=128`, `n=2048`
- **Per GPU (4 heads)**: 4 × Get_Time(2097152, 128, 2048)
- **Total parallel**: 8 GPUs × 4 heads = 32 heads total

#### 4. Attention Output
- **Operation**: Attention weights × V
- **Shape**: `[batch_size × seq_len, seq_len] × [seq_len, head_dim]`
- **Per head**: `m=2,097,152`, `k=2048`, `n=128`
- **Per GPU**: 4 × Get_Time(2097152, 2048, 128)

#### 5. Output Projection
- **Operation**: Concatenated attention outputs → hidden_dim
- **Shape**: `[batch_size × seq_len, hidden_dim] × [hidden_dim, hidden_dim]`
- **Tensor Parallel**: Row-parallel split
- **Per GPU**: Get_Time(2097152, 4096, 512)

#### 6. MLP Up-Projection
- **Operation**: Hidden → Expert hidden dimension
- **Shape**: `[batch_size × seq_len, hidden_dim] × [hidden_dim, 4×hidden_dim]`
- **Dimensions**: `m=2,097,152`, `k=4096`, `n=16384`
- **TP Column Split**: Get_Time(2097152, 512, 16384)

#### 7. MLP Down-Projection
- **Operation**: Expert hidden → Hidden dimension
- **Shape**: `[batch_size × seq_len, 4×hidden_dim] × [4×hidden_dim, hidden_dim]`
- **Dimensions**: `m=2,097,152`, `k=16384`, `n=4096`
- **TP Row Split**: Get_Time(2097152, 2048, 512)

### Longest Path Analysis - Baseline

The baseline configuration uses pipeline parallelism across 2 stages with 8 GPUs each:

**Critical Path (Longest Sequential Chain):**
1. **Stage 0, Layer 0**: Complete attention + MLP sequence
2. **Stage 0, Layer 1**: Complete attention + MLP sequence  
3. **Pipeline communication**: Send to Stage 1
4. **Stage 1, Layer 2**: Complete attention + MLP sequence
5. **Stage 1, Layer 3**: Complete attention + MLP sequence
6. **Final output projection**

**Detailed Sequential Chain (per layer):**
```
LayerNorm → QKV Projection → Attention Scores → Softmax → Attention Output → Output Projection → All-Reduce → Residual → LayerNorm → MLP Up → Activation → MLP Down → All-Reduce → Residual
```

**Longest Sequential Operations (bottlenecks):**
- **QKV Projection**: 3 × Get_Time(2097152, 512, 4096) = 3 parallel ops, but sequential Q→K→V
- **MLP Projections**: Get_Time(2097152, 512, 16384) + Get_Time(2097152, 2048, 512)
- **Attention Operations**: 4 heads × Get_Time(2097152, 128, 2048) + 4 heads × Get_Time(2097152, 2048, 128)

**Total Critical Path Time (4 layers, 2 stages):**
- Stage 0: 2 layers × [QKV + Attention + MLP time] + communication
- Stage 1: 2 layers × [QKV + Attention + MLP time] + output
- **Total**: 4 × [3×Get_Time(2097152,512,4096) + 8×Get_Time(2097152,128,2048) + Get_Time(2097152,512,16384) + Get_Time(2097152,2048,512)] + communication overhead

## MA-Separation Configuration Analysis

### Matrix Multiplication Operations

#### Attention Computation (GPUs 0-11)

**Head Distribution:**
- GPUs 0-7: 3 heads each (24 heads total)
- GPUs 8-11: 2 heads each (8 heads total)

#### 1. QKV Projection (per GPU)
- **GPU 0-7**: 3 heads → 3 × Get_Time(2097152, 341, 384)
- **GPU 8-11**: 2 heads → 2 × Get_Time(2097152, 341, 256)
- **All execute in parallel**

#### 2. Attention Scores
- **GPU 0-7**: 3 × Get_Time(2097152, 128, 2048) per head
- **GPU 8-11**: 2 × Get_Time(2097152, 128, 2048) per head
- **All execute in parallel**

#### 3. Attention Output
- **GPU 0-7**: 3 × Get_Time(2097152, 2048, 128) per head
- **GPU 8-11**: 2 × Get_Time(2097152, 2048, 128) per head
- **All execute in parallel**

#### 4. Output Projection
- **GPU 0-11**: Individual projections
- **GPU 0-7**: Get_Time(2097152, 384, 341)
- **GPU 8-11**: Get_Time(2097152, 256, 341)
- **Followed by hierarchical all-reduce**

#### MoE Computation (GPUs 12-15)

#### 1. Gate Network
- **Operation**: Routing computation
- **Shape**: `[batch_size × seq_len, hidden_dim] × [hidden_dim, num_experts]`
- **Dimensions**: `m=2,097,152`, `k=4096`, `n=16`
- **Per GPU**: Get_Time(2097152, 4096, 16)

#### 2. Expert Up-Projection
- **Operation**: Hidden → Expert hidden (per expert)
- **Shape**: `[batch_size × seq_len, hidden_dim] × [hidden_dim, 4×hidden_dim]`
- **Dimensions**: `m=2,097,152`, `k=4096`, `n=16384`
- **Per expert**: Get_Time(2097152, 4096, 16384)
- **4 experts per GPU execute in parallel**

#### 3. Expert Down-Projection
- **Operation**: Expert hidden → Hidden (per expert)
- **Shape**: `[batch_size × seq_len, 4×hidden_dim] × [4×hidden_dim, hidden_dim]`
- **Dimensions**: `m=2,097,152`, `k=16384`, `n=4096`
- **Per expert**: Get_Time(2097152, 16384, 4096)
- **4 experts per GPU execute in parallel**

### Longest Path Analysis - MA Separation

**Critical Path (Longest Sequential Chain):**

The MA-separation configuration enables better parallelization:

1. **Attention Path (GPUs 0-11 in parallel):**
   - All 12 GPUs execute attention computations simultaneously
   - **Critical sub-path per GPU**: QKV → Scores → Softmax → Output → Projection
   - **Longest single GPU**: GPU 0-7 (3 heads each)
   - **Time**: 3×Get_Time(2097152,341,384) + 3×Get_Time(2097152,128,2048) + 3×Get_Time(2097152,2048,128) + Get_Time(2097152,384,341)

2. **Communication Overhead:**
   - Broadcast attention output to MoE GPUs
   - All-to-all communication after expert computation

3. **MoE Path (GPUs 12-15 in parallel):**
   - All 4 GPUs execute MoE computations simultaneously
   - **Critical sub-path per GPU**: Gate → 4 experts (parallel) → Aggregation
   - **Expert time**: Get_Time(2097152,4096,16384) + Get_Time(2097152,16384,4096)
   - **All 4 experts per GPU execute in parallel**

4. **Layer progression (4 layers total):**
   - Each layer: Attention → MoE → next layer

## Performance Comparison

### Baseline Configuration
- **Sequential layers**: 4 layers in sequence
- **Pipeline stages**: 2 stages, 2-layer pipeline
- **Parallelism**: 8-way tensor parallel within each stage
- **Critical path**: 4 sequential layers + communication

### MA-Separation Configuration  
- **Parallel modules**: Attention (12 GPUs) + MoE (4 GPUs) executing simultaneously
- **Layer pipelining**: 4 layers with better parallel utilization
- **Critical path**: Single layer attention + MoE sequence
- **Massive parallelization**: 16 experts running simultaneously (4 per GPU × 4 GPUs)

### Optimized Get_Time Expressions

**Baseline Longest Path:**
```
Total_Time_Baseline = 4 × [
    max(Get_Time(2097152,512,4096) × 3, 8×Get_Time(2097152,128,2048) + 8×Get_Time(2097152,2048,128)) +
    Get_Time(2097152,512,16384) + 
    Get_Time(2097152,2048,512)
] + Communication_Overhead
```

**MA-Separation Longest Path:**
```
Total_Time_MA = 4 × [
    max(
        max_gpu_attention_time,  
        max_gpu_moe_time
    ) + 
    Communication_Overhead
]

where:
max_gpu_attention_time = max(
    3×Get_Time(2097152,341,384) + 3×Get_Time(2097152,128,2048) + 3×Get_Time(2097152,2048,128) + Get_Time(2097152,384,341) [GPUs 0-7],
    2×Get_Time(2097152,341,256) + 2×Get_Time(2097152,128,2048) + 2×Get_Time(2097152,2048,128) + Get_Time(2097152,256,341) [GPUs 8-11]
)

max_gpu_moe_time = Get_Time(2097152,4096,16) + max_expert_time + Get_Time(2097152,4096,4096)
max_expert_time = max of 4 parallel experts: Get_Time(2097152,4096,16384) + Get_Time(2097152,16384,4096)
```

## Key Insights

1. **Parallelization Benefits**: MA-separation achieves 34.2% TPOT reduction by maximizing parallel execution
2. **Expert Parallelism**: 16 experts running simultaneously vs sequential MLP in baseline
3. **Communication Patterns**: MA-separation uses more efficient broadcast/all-to-all vs pipeline communication
4. **Matrix Multiplication Bottlenecks**: Largest operations are the MLP projections (4096→16384→4096)
5. **Longest Path**: Baseline has 4 sequential layers; MA-separation reduces to single layer critical path with maximum parallelization

The MA-separation configuration significantly reduces the critical path by enabling massive parallelization of both attention computation and expert processing, while the baseline configuration suffers from the sequential nature of pipeline stages.