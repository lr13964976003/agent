# DAG Runtime Analysis for Helix Model

## Overview
This document provides a comprehensive analysis of the runtime for the optimized Helix model DAG, including matrix multiplication operations, longest path identification, and computation time analysis using Get_Time(m, k, n) function calls.

## 1. DAG Structure Analysis

The optimized Helix model implements a 2-stage pipeline parallelism approach with:
- **Stage 0**: GPUs 0-7 (MHA Layer 0 + MLP Layer 0)
- **Stage 1**: GPUs 8-15 (MHA Layer 1 + MLP Layer 1)
- **Micro-batch overlap**: Enables concurrent execution of micro-batches

## 2. Matrix Multiplication Operations Analysis

### 2.1 Multi-Head Attention (MHA) Layer

Each MHA layer contains the following matrix multiplication operations:

#### QKV Projections (Parallel across 8 devices)
- **Q Projection**: 8 parallel matrix multiplications
  - Dimensions: [batch_size × seq_len, embed_dim] × [embed_dim, heads × d_k]
  - Each device computes: Get_Time(1024×10000, 8192, 8×128) = Get_Time(10,240,000, 8192, 1024)

- **K Projection**: 8 parallel matrix multiplications
  - Dimensions: [batch_size × seq_len, embed_dim] × [embed_dim, heads × d_k]
  - Each device computes: Get_Time(10,240,000, 8192, 1024)

- **V Projection**: 8 parallel matrix multiplications
  - Dimensions: [batch_size × seq_len, embed_dim] × [embed_dim, heads × d_v]
  - Each device computes: Get_Time(10,240,000, 8192, 1024)

#### Attention Computation (Fused across 8 devices)
Each attention head performs:
1. **Q × K^T**: [batch_size × seq_len, d_k] × [d_k, batch_size × seq_len]
   - Get_Time(1024×10000, 128, 1024×10000) = Get_Time(10,240,000, 128, 10,240,000)

2. **Attention × V**: [batch_size × seq_len, seq_len] × [seq_len, d_v]
   - Get_Time(10,240,000, 10000, 128)

#### Output Projection (All-reduce across 8 devices)
- **Projection**: [batch_size × seq_len, embed_dim] × [embed_dim, embed_dim]
- Get_Time(10,240,000, 8192, 8192)

### 2.2 MLP Layer

Each MLP layer contains:

#### FC1 (Feed-forward 1) - Column Parallel (8 devices)
- **Dimensions**: [batch_size × seq_len, embed_dim] × [embed_dim, hidden_dim/8]
- Each device computes: Get_Time(10,240,000, 8192, 4096)

#### FC2 (Feed-forward 2) - Row Parallel (8 devices)
- **Dimensions**: [batch_size × seq_len, hidden_dim/8] × [hidden_dim/8, embed_dim]
- Each device computes: Get_Time(10,240,000, 4096, 1024)

#### Concatenation & All-reduce operations
- FC1 concatenation: 8×4096 → 32768 dimensions
- FC2 all-reduce: 8×1024 → 8192 dimensions

## 3. Longest Path Analysis

### Critical Path (Serial Dependencies)
The longest path through the DAG consists of:

1. **Model Input** → **Stage0_MHA** → **Stage1_MHA** → **Stage1_MLP** → **Model Output**

### Detailed Path Breakdown:

#### Path 1: Stage 0 Critical Path
- **Stage0_MHA** (MHA Layer 0):
  - QKV Projections: 3 × Get_Time(10,240,000, 8192, 1024) [parallel across 8 devices]
  - Attention computation: Get_Time(10,240,000, 128, 10,240,000) [per device]
  - Output projection: Get_Time(10,240,000, 8192, 8192) [with all-reduce]
  - **Total MHA computation time**: Max(parallel operations) + serial bottleneck

#### Path 2: Stage 1 Critical Path  
- **Stage1_MHA** (MHA Layer 1):
  - Same as Stage0_MHA

- **Stage1_MLP** (MLP Layer 1):
  - FC1: Get_Time(10,240,000, 8192, 4096) [parallel across 8 devices]
  - FC2: Get_Time(10,240,000, 4096, 1024) [parallel across 8 devices]
  - **Total MLP computation time**: Max(parallel operations) + all-reduce overhead

## 4. Pipeline Overlap Considerations

Due to pipeline parallelism with micro-batch overlap:
- **Micro-batch 0**: Follows the serial path Stage0_MHA → Stage1_MHA → Stage1_MLP
- **Micro-batch 1**: Overlaps with Stage0_MLP → Stage1_MLP after Stage0_MHA completion

The **longest path** remains the serial dependency chain:
**Model_Input → Stage0_MHA → Stage1_MHA → Stage1_MLP → Model_Output**

## 5. Get_Time Function Calls Summary

The critical path requires the following Get_Time calls in sequence:

### For MHA Layer (per layer):
1. **QKV Projections**: 3 × Get_Time(10,240,000, 8192, 1024) (parallel)
2. **Attention Q×K^T**: Get_Time(10,240,000, 128, 10,240,000)
3. **Attention ×V**: Get_Time(10,240,000, 10000, 128)
4. **Output Projection**: Get_Time(10,240,000, 8192, 8192)

### For MLP Layer (per layer):
1. **FC1**: Get_Time(10,240,000, 8192, 4096) (parallel)
2. **FC2**: Get_Time(10,240,000, 4096, 1024) (parallel)

## 6. Runtime Calculation

**Total Runtime = Get_Time(MHA Layer 0) + Get_Time(MHA Layer 1) + Get_Time(MLP Layer 1)**

Where:
- **MHA Layer Runtime**: Max(Get_Time(10,240,000, 8192, 1024), Get_Time(10,240,000, 128, 10,240,000), Get_Time(10,240,000, 8192, 8192))
- **MLP Layer Runtime**: Max(Get_Time(10,240,000, 8192, 4096), Get_Time(10,240,000, 4096, 1024))

The bottleneck is determined by the largest matrix multiplication in each layer, typically the attention computation Get_Time(10,240,000, 128, 10,240,000) due to the large K^T×Q product.

## 7. Conclusion

**Longest Path**: Model_Input → Stage0_MHA → Stage1_MHA → Stage1_MLP → Model_Output

**Critical Get_Time Calls**:
1. **Stage0_MHA**: Get_Time(10,240,000, 128, 10,240,000) [attention bottleneck]
2. **Stage1_MHA**: Get_Time(10,240,000, 128, 10,240,000) [attention bottleneck]  
3. **Stage1_MLP**: Get_Time(10,240,000, 8192, 4096) [FC1 bottleneck]

The pipeline parallelism enables overlap between stages, but the critical path remains the serial execution of these three major components.