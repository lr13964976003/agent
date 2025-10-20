# DAG Runtime Analysis for MA Separation Model

## Overview
This document analyzes the runtime of the Mixture of Experts (MoE) model DAG with 4 layers, analyzing matrix multiplication operations and identifying the longest path.

## Model Parameters
- **Layers**: 4
- **Hidden Dimension**: 4096
- **Sequence Length**: 2048
- **Attention Heads**: 32 (4 per GPU across 8 GPUs)
- **Head Dimension**: 128
- **Experts**: 16 total (2 per GPU across 8 MoE GPUs)
- **Expert Hidden Dimension**: 16384
- **Batch Size**: B (variable)

## Matrix Multiplication Operations Analysis

### 1. QKV Projections
**Location**: Each attention GPU (0-7) per layer
- **Input**: [B, 2048, 4096]
- **Weight Matrix**: [4096, 512] (for each of Q, K, V)
- **Output**: [B, 2048, 512] per projection type
- **Parallelization**: 8 GPUs execute simultaneously
- **Get_Time**: `Get_Time(B*2048, 4096, 512)` for each Q/K/V projection

### 2. Multi-Head Attention Computations
**Location**: Each attention GPU computes 4 parallel heads

#### 2.1 Q×K^T Multiplication
- **Input Q**: [B, 4, 2048, 128]
- **Input K^T**: [B, 4, 128, 2048]
- **Output**: [B, 4, 2048, 2048]
- **Matrix Dimensions**: [B×4×2048, 128] × [128, 2048] = [B×4×2048, 2048]
- **Get_Time**: `Get_Time(B*4*2048, 128, 2048)`

#### 2.2 Attention×V Multiplication
- **Input Attention**: [B, 4, 2048, 2048]
- **Input V**: [B, 4, 2048, 128]
- **Output**: [B, 4, 2048, 128]
- **Matrix Dimensions**: [B×4×2048, 2048] × [2048, 128] = [B×4×2048, 128]
- **Get_Time**: `Get_Time(B*4*2048, 2048, 128)`

### 3. Attention Output Projection
**Location**: Each attention GPU (0-7) per layer
- **Input**: [B, 2048, 4096] (after all-reduce)
- **Weight Matrix**: [4096, 4096]
- **Output**: [B, 2048, 4096]
- **Parallelization**: 8 GPUs execute simultaneously
- **Get_Time**: `Get_Time(B*2048, 4096, 4096)`

### 4. MoE Expert Computations
**Location**: Each MoE GPU (8-15) with 2 experts per GPU

#### 4.1 Expert Up-Projection
- **Input**: [B, 2048, 4096]
- **Weight Matrix**: [4096, 16384]
- **Output**: [B, 2048, 16384]
- **Get_Time**: `Get_Time(B*2048, 4096, 16384)`

#### 4.2 Expert Down-Projection
- **Input**: [B, 2048, 16384]
- **Weight Matrix**: [16384, 4096]
- **Output**: [B, 2048, 4096]
- **Get_Time**: `Get_Time(B*2048, 16384, 4096)`

## Longest Path Analysis

### Critical Path per Layer
The longest path through the DAG follows this sequence for each layer:

1. **LayerNorm 1** (negligible time)
2. **QKV Projections** (parallel across 8 GPUs)
   - Slowest operation: `Get_Time(B*2048, 4096, 512)` × 3 (Q, K, V)
3. **Multi-Head Attention** (parallel across 8 GPUs)
   - Q×K^T: `Get_Time(B*4*2048, 128, 2048)`
   - Attention×V: `Get_Time(B*4*2048, 2048, 128)`
4. **All-Reduce** (communication, not matmul)
5. **Attention Output Projection** (parallel across 8 GPUs)
   - `Get_Time(B*2048, 4096, 4096)`
6. **LayerNorm 2** (negligible time)
7. **Gate Computation** (routing, not matmul)
8. **Expert Processing** (parallel across 8 MoE GPUs with 2 experts each)
   - Expert Up-Projection: `Get_Time(B*2048, 4096, 16384)`
   - Expert Down-Projection: `Get_Time(B*2048, 16384, 4096)`
9. **Expert Aggregation** (reduction, not matmul)

### Total Longest Path
Since there are 4 sequential layers and operations within each layer layer are parallelized, the total runtime is:

**Total Runtime = 4 × max(layer_operations)**

Where `layer_operations` includes:
- QKV projections: 3 × `Get_Time(B*2048, 4096, 512)`
- Attention Q×K^T: `Get_Time(B*4*2048, 128, 2048)`
- Attention×V: `Get_Time(B*4*2048, 2048, 128)`
- Output projection: `Get_Time(B*2048, 4096, 4096)`
- Expert up-projection: `Get_Time(B*2048, 4096, 16384)`
- Expert down-projection: `Get_Time(B*2048, 16384, 4096)`

Given the dimensions, the **MoE expert computations** (16384 dimension) are likely the bottleneck per layer.

### Final Longest Path Expression
The critical path runtime for the entire DAG is:

**Total Runtime = 4 × [Get_Time(B*2048, 4096, 16384) + Get_Time(B*2048, 16384, 4096)]**

This assumes that:
1. All operations within a layer are parallelized across available GPUs
2. The expert computations (with largest matrix dimensions) dominate the runtime
3. Communication costs (all-reduce, broadcast) are overlapped with computation
4. The sequential nature of the 4 layers determines the total runtime

## Parallelism Notes
- **Attention**: 8 GPUs parallelize 32 heads (4 heads per GPU)
- **MoE**: 8 GPUs parallelize 16 experts (2 experts per GPU)
- **QKV and Output Projections**: Replicated across attention GPUs
- **Expert Processing**: Parallel across MoE GPUs with top-2 routing

The DAG has no cycles (confirmed by Extract Info From DAG), ensuring correct execution order.