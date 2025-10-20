# DAG Runtime Analysis for Large Language Model Deployment

## Overview
This document analyzes the runtime characteristics of two transformer architectures:
1. **Baseline Transformer** - 4-layer model with Tensor Parallelism (TP=8) and Pipeline Parallelism (PP=2)
2. **FA Pool Transformer** - 4-layer model with dynamic attention pooling for sequence parallelism

## Matrix Multiplication Operations Analysis

### Baseline Transformer (TP=8, PP=2)

#### Layer Structure per Transformer Layer
Each transformer layer contains the following matrix multiplication operations:

**1. QKV Projection (3 separate matrix multiplications)**
- **Operation**: Linear projection of input to query, key, and value matrices
- **Dimensions**: [batch_size × seq_len × hidden_dim] × [hidden_dim × 3 × num_heads × head_dim]
- **Given**: batch_size=1024, hidden_dim=4096, num_heads=32, head_dim=128
- **MM shapes**: 
  - Input: [1024 × seq_len, 4096]
  - Weight: [4096, 32 × 128 × 3 = 12288]
  - Output: [1024 × seq_len, 12288]
- **TP=8 partition**: Each GPU handles [1024 × seq_len, 512] × [512, 12288]

**2. Attention Score Calculation**
- **Operation**: Query × Key^T for attention weights
- **Dimensions**: [batch_size × seq_len × num_heads × head_dim] × [batch_size × num_heads × head_dim × seq_len]
- **MM shapes**: [1024 × seq_len, 32 × 128] × [32 × 128, seq_len]

**3. Attention Weighted Sum**
- **Operation**: Attention weights × Value
- **Dimensions**: [batch_size × seq_len × seq_len] × [batch_size × seq_len × num_heads × head_dim]
- **MM shapes**: [1024 × seq_len, seq_len] × [seq_len, 32 × 128]

**4. Attention Output Projection**
- **Operation**: Concatenated attention heads projection
- **Dimensions**: [batch_size × seq_len × hidden_dim] × [hidden_dim × hidden_dim]
- **MM shapes**: [1024 × seq_len, 4096] × [4096, 4096]
- **TP=8 partition**: [1024 × seq_len, 512] × [512, 4096]

**5. FFN Operations (2 matrix multiplications per layer)**
- **FFN Up Projection**:
  - Dimensions: [batch_size × seq_len × hidden_dim] × [hidden_dim × ffn_hidden_dim]
  - MM shapes: [1024 × seq_len, 4096] × [4096, 16384]
  - TP=8 partition: [1024 × seq_len, 512] × [512, 2048]

- **FFN Down Projection**:
  - Dimensions: [batch_size × seq_len × ffn_hidden_dim] × [ffn_hidden_dim × hidden_dim]
  - MM shapes: [1024 × seq_len, 16384] × [16384, 4096]
  - TP=8 partition: [1024 × seq_len, 2048] × [2048, 512]

#### Total Matrix Multiplications per Layer
- **QKV Projections**: 3 separate MMs (Q, K, V)
- **Attention**: 2 MMs (scores + weighted sum)
- **Output Projection**: 1 MM
- **FFN**: 2 MMs (up + down)
- **Total**: 8 matrix multiplications per transformer layer

#### Total for 4-Layer Model
- **Total MMs**: 4 × 8 = 32 matrix multiplications
- **Pipeline stages**: 2 stages (layers 0-1 on GPUs 0-7, layers 2-3 on GPUs 8-15)

### FA Pool Transformer (Sequence Parallelism)

#### Layer Structure with Attention Pooling
The FA Pool architecture separates attention computation from FFN using dedicated GPU pools:

**1. Attention Pool Operations (parallel across 5 GPUs)**
- **QKV Projection per block**: Same as baseline but sequence-parallel
- **Block size**: seq_len/5 per GPU
- **MM shapes per block**: [1024 × (seq_len/5), 512] × [512, 12288] for QKV

**2. Attention Computation per Block**
- **Score calculation**: [1024 × (seq_len/5), 32 × 128] × [32 × 128, seq_len]  
- **Weighted sum**: [1024 × (seq_len/5), seq_len] × [seq_len, 32 × 128]
- **Output projection**: [1024 × (seq_len/5), 512] × [512, 4096]

**3. FFN Operations (on base GPUs 0-7)**
- **Parallel to attention**: While attention is computed on pool GPUs, FFN runs in parallel
- **Same MM dimensions as baseline**: 2 MMs per layer

#### Key Parallelism Differences
- **Attention computation**: Parallel across 5 GPUs by sequence partitioning
- **FFN computation**: Parallel across 8 GPUs by tensor partitioning
- **Overlap**: Attention and FFN can run concurrently

## Longest Path Analysis

### Baseline Transformer Longest Path
The critical path follows the sequential execution through all layers:

```
Input → Layer0_RMSNorm → Layer0_QKV → Layer0_Attention → Layer0_FFN → 
Layer1_RMSNorm → Layer1_QKV → Layer1_Attention → Layer1_FFN → 
Pipeline_Comm → Layer2_RMSNorm → Layer2_QKV → Layer2_Attention → Layer2_FFN → 
Layer3_RMSNorm → Layer3_QKV → Layer3_Attention → Layer3_FFN → Output_Proj → Output
```

**Critical Path Length**: 4 complete transformer layers + pipeline communication

### FA Pool Transformer Longest Path
The longest path involves the full sequence through attention pools:

```
Input → Send_to_Pool → 
Layer0_Attention_All_Blocks (parallel) → Layer0_Attention_Aggregate → Recv_from_Pool → 
Layer0_FFN → 
Send_to_Pool_1 → 
Layer1_Attention_All_Blocks (parallel) → Layer1_Attention_Aggregate → Recv_from_Pool_1 → 
Layer1_FFN → 
Send_to_Pool_2 → 
Layer2_Attention_All_Blocks (parallel) → Layer2_Attention_Aggregate → Recv_from_Pool_2 → 
Layer2_FFN → 
Send_to_Pool_3 → 
Layer3_Attention_All_Blocks (parallel) → Layer3_Attention_Aggregate → Recv_from_Pool_3 → 
Layer3_FFN → Output_Proj → Output
```

**Critical Path Length**: 4 attention pool transfers + 4 attention computations + 4 FFN computations

## Runtime Calculation using Get_Time(m, k, n)

### Baseline Transformer Runtime
For each transformer layer, the critical path includes:

**Layer 0 (GPUs 0-7):**
- QKV Projection: 3× Get_Time(1024×seq_len, 512, 12288) - parallel across 8 GPUs
- Attention Score: Get_Time(1024×seq_len, 32×128, seq_len) 
- Weighted Sum: Get_Time(1024×seq_len, seq_len, 32×128)
- Output Projection: Get_Time(1024×seq_len, 512, 4096)
- FFN Up: Get_Time(1024×seq_len, 512, 2048)
- FFN Down: Get_Time(1024×seq_len, 2048, 512)

**Layer 1 (GPUs 0-7):** Same as Layer 0
**Pipeline Communication:** GPU[7] → GPU[8] transfer
**Layer 2 (GPUs 8-15):** Same as Layer 0
**Layer 3 (GPUs 8-15):** Same as Layer 0

**Total Runtime**: 4 × [Layer computation time] + Pipeline communication overhead

### FA Pool Transformer Runtime
For each layer, the critical path includes:

**Per Layer in FA Pool:**
- Send to Pool: Communication overhead
- Attention per block (parallel): 
  - QKV: 3× Get_Time(1024×(seq_len/5), 512, 12288) - 5 blocks in parallel
  - Score: Get_Time(1024×(seq_len/5), 32×128, seq_len) - with full sequence
  - Weighted Sum: Get_Time(1024×(seq_len/5), seq_len, 32×128)
  - Output: Get_Time(1024×(seq_len/5), 512, 4096)
- Attention Aggregation: Synchronization overhead
- Receive from Pool: Communication overhead
- FFN (parallel): 
  - FFN Up: Get_Time(1024×seq_len, 512, 2048) - 8 GPUs in parallel
  - FFN Down: Get_Time(1024×seq_len, 2048, 512) - 8 GPUs in parallel

**Total Runtime**: 4 × [max(Attention pool time, FFN time) + communication overhead]

## Key Performance Insights

1. **Parallelism Trade-offs**:
   - Baseline: Tensor parallelism (TP=8) + Pipeline parallelism (PP=2)
   - FA Pool: Sequence parallelism (5 GPUs) + Tensor parallelism (8 GPUs)

2. **Critical Path Differences**:
   - Baseline: Sequential through 4 layers with pipeline stall
   - FA Pool: Overlapped attention and FFN computation with communication overhead

3. **Matrix Multiplication Scaling**:
   - Baseline: Fixed per-GPU workload regardless of sequence length
   - FA Pool: Scales with sequence length (divided by 5 for attention)

4. **Communication Patterns**:
   - Baseline: Single pipeline communication between layers 1-2
   - FA Pool: 4 send/receive operations per layer for attention pooling

This analysis provides the foundation for understanding the runtime characteristics and optimizing the parallel strategies for large language model deployment.