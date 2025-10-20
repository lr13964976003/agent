# Helix Two-Level Attention Partitioning DAG Runtime Analysis

## Overview
This document analyzes the runtime of the Helix Two-Level Attention Partitioning DAG, which implements a 2-layer transformer model with the following configuration:
- **Batch size**: 1024
- **Sequence length**: 10000
- **Hidden size**: 8192
- **Attention heads**: 16 (split into 4 groups of 4 heads each)
- **Head dimensions**: 512 dimensions per head (split into 4 segments of 128 dimensions each)
- **MLP hidden size**: 32768 (split across 16 GPUs)
- **Total GPUs**: 16 arranged in a 4x4 grid

## Matrix Multiplication Operations

### Layer 1: Attention Block

#### QKV Projections (Parallel across 16 GPUs)
Each of the 16 GPUs performs 3 separate matrix multiplications:
- **Operation**: [1024, 10000, 8192] × [8192, 4×128] → [1024, 10000, 4, 128]
- **Parallelism**: All 16 GPUs execute simultaneously
- **Total operations**: 16 GPUs × 3 projections × [1024×10000×8192×512]
- **Effective dimensions**: m=1024×10000, k=8192, n=512

#### Scaled Dot-Product Attention
Each attention head computation involves:
1. **Q×K^T**: [1024, 10000, 4, 128] × [1024, 10000, 128, 4] → [1024, 10000, 4, 10000]
   - **Dimensions**: m=1024×10000×4, k=128, n=10000
2. **Attention×V**: [1024, 10000, 4, 10000] × [1024, 10000, 10000, 128] → [1024, 10000, 4, 128]
   - **Dimensions**: m=1024×10000×4, k=10000, n=128

#### Attention Output Projection
- **Operation**: [1024, 10000, 8192] × [8192, 8192] → [1024, 10000, 8192]
- **Dimensions**: m=1024×10000, k=8192, n=8192

### Layer 1: MLP Block

#### Column Parallel Linear (Parallel across 16 GPUs)
- **Operation**: [1024, 10000, 8192] × [8192, 2048] → [1024, 10000, 2048]
- **Parallelism**: 16 GPUs execute simultaneously
- **Dimensions per GPU**: m=1024×10000, k=8192, n=2048

#### Row Parallel Linear (Parallel across 16 GPUs)
- **Operation**: [1024, 10000, 2048] × [2048, 512] → [1024, 10000, 512]
- **Parallelism**: 16 GPUs execute simultaneously
- **Dimensions per GPU**: m=1024×10000, k=2048, n=512

#### MLP Output Projection
- **Operation**: [1024, 10000, 8192] × [8192, 8192] → [1024, 10000, 8192]
- **Dimensions**: m=1024×10000, k=8192, n=8192

### Layer 2: Attention Block (Identical to Layer 1)
Same operations as Layer 1 with identical dimensions.

### Layer 2: MLP Block (Identical to Layer 1)
Same operations as Layer 1 with identical dimensions.

## Longest Path Analysis

The critical path through the DAG follows this sequence:

1. **Input preprocessing**: Input → Layer Norm 1
2. **QKV projections**: Layer Norm 1 → 48 projection operations (16 GPUs × 3 projections)
3. **Attention computation**: QKV projections → 16 attention operations → concatenation → output projection
4. **MLP computation**: Attention output → Layer Norm 2 → 16 column parallel operations → 16 row parallel operations → output projection
5. **Layer 2**: Repeat of steps 2-4
6. **Final output**: Layer 2 MLP → final residual → output

### Critical Path Matrix Operations (Serial Dependencies)

The longest path involves the following sequential matrix multiplications:

#### Layer 1:
1. **Q Projection**: Get_Time(1024×10000, 8192, 512)
2. **K Projection**: Get_Time(1024×10000, 8192, 512)  (parallel with Q)
3. **V Projection**: Get_Time(1024×10000, 8192, 512)  (parallel with Q/K)
4. **Q×K^T**: Get_Time(1024×10000×4, 128, 10000)
5. **Attention×V**: Get_Time(1024×10000×4, 10000, 128)
6. **Attention output projection**: Get_Time(1024×10000, 2048, 8192)
7. **MLP column parallel**: Get_Time(1024×10000, 8192, 2048)
8. **MLP row parallel**: Get_Time(1024×10000, 2048, 512)
9. **MLP output projection**: Get_Time(1024×10000, 8192, 8192)

#### Layer 2 (Identical to Layer 1):
10. **Q2 Projection**: Get_Time(1024×10000, 8192, 512)
11. **K2 Projection**: Get_Time(1024×10000, 8192, 512)
12. **V2 Projection**: Get_Time(1024×10000, 8192, 512)
13. **Q2×K2^T**: Get_Time(1024×10000×4, 128, 10000)
14. **Attention×V2**: Get_Time(1024×10000×4, 10000, 128)
15. **Attention2 output projection**: Get_Time(1024×10000, 2048, 8192)
16. **MLP2 column parallel**: Get_Time(1024×10000, 8192, 2048)
17. **MLP2 row parallel**: Get_Time(1024×10000, 2048, 512)
18. **MLP2 output projection**: Get_Time(1024×10000, 8192, 8192)

## Parallel Execution Summary

### Parallel Operations
- **QKV Projections**: 16 GPUs execute 3 projections each in parallel
- **Attention heads**: 16 attention heads computed in parallel across 16 GPUs
- **MLP operations**: Column and row parallel operations distributed across 16 GPUs

### Serial Bottlenecks
The total runtime is determined by the longest sequential path through the model:

**Total Runtime** = Get_Time(1024×10000, 8192, 512) × 2 (QKV projections per layer) 
                  + Get_Time(1024×10000×4, 128, 10000) × 2 (Q×K^T per layer)
                  + Get_Time(1024×10000×4, 10000, 128) × 2 (Attention×V per layer)
                  + Get_Time(1024×10000, 2048, 8192) × 2 (Attention output per layer)
                  + Get_Time(1024×10000, 8192, 2048) × 2 (MLP column per layer)
                  + Get_Time(1024×10000, 2048, 512) × 2 (MLP row per layer)
                  + Get_Time(1024×10000, 8192, 8192) × 2 (MLP output per layer)

Note: The actual matrix dimensions for the Get_Time function should use the flattened batch×sequence dimensions as the m parameter, and the appropriate feature dimensions for k and n parameters.