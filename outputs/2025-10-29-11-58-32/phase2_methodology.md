# Helix Parallelism: Methodology Extraction (Phase 2)

## Core Methodology

### 1. Helix Parallelism Architecture
Helix introduces a temporal pipeline where the same set of N GPUs is reused across attention and FFN computation, applying different parallelism strategies for each phase.

### 2. Attention Phase Partitioning

#### 2.1 KV Parallelism (KVP)
- **Configuration**: N = KVP × TPA GPUs where TPA ≤ K (number of KV heads)
- **Sharding**: KV cache sharded along sequence dimension across KVP GPUs
- **Eliminates**: Full cache replication and reduces DRAM footprint
- **Memory distribution**: Each KVP GPU holds only S/KVP tokens

#### 2.2 QKV Projection Strategy
- **Input processing**: Each KVP GPU independently computes full QKV projections
- **Input dimensions**: [B, H] batch processed by each GPU
- **Weight matrices**:
  - WQ: H × (H/TPA)
  - WK: H × (⌈K/TPA⌉·Hsz)
  - WV: H × (⌈K/TPA⌉·Hsz)
- **Output**: Each GPU produces partial attention output + log-sum-exp scalar

#### 2.3 All-to-All Communication
- **Exchange pattern**: Single round over query-head axis
- **Communication volume**: Independent of sequence length S, scales with B × H
- **Data exchanged**: Partial attention outputs and log-sum-exp scalars
- **Result**: Exact softmax-normalized attention without extra synchronization

#### 2.4 Post-Attention Linear Projection
- **Configuration**: TP across N = KVP × TPA GPUs
- **Weight matrix**: Shard of shape (H/N) × H per GPU
- **Computation**: B × (H/N) local matrix multiply
- **Communication**: All-Reduce over N GPUs to aggregate B × H output

### 3. FFN Phase Partitioning

#### 3.1 Dense FFN (EP = 1)
- **Configuration**: TPF = N GPUs in tensor-parallel mode
- **Computation pattern**: [B, H] → [B, F/N] → [B, H]
- **Communication**: TP All-Reduce after FFN computation

#### 3.2 MoE FFN (EP > 1)
- **Configuration**: N GPUs repartitioned into TPF × EP grid
- **Expert routing**: Tokens routed to appropriate experts
- **Computation layers**: TP applied within each expert group
- **Communication sequence**:
  1. Intra-expert All-Reduce
  2. Inter-expert All-Gather
  3. Local reduction to yield [B, H] output

### 4. Helix HOP-B Overlap Strategy

#### 4.1 Batch-wise Pipelining
- **Mechanism**: Overlaps All-to-All communication with attention computation
- **Implementation**: 
  - Attention output for token i computed
  - Communication initiated for token i simultaneously
  - Attention computation proceeds for token i+1
- **Result**: Communication latency hidden behind computation

#### 4.2 Communication-Computation Mapping
- **Without HOP-B**: Sequential execution (attention → communication → next)
- **With HOP-B**: Overlapped execution (communication of token i concurrent with compute of token i+1)

### 5. Distributed KV Concatenation

#### 5.1 Staged Update Strategy
- **Mechanism**: Round-robin KV updates across KVP ranks
- **Pattern**: Fixed number of tokens (e.g., 16) appended to each rank in sequence
- **Cycle**: KVP Rank 0 → KVP Rank 1 → ... → KVP Rank KVP-1 → Repeat
- **Benefit**: Balanced memory growth across all GPUs

#### 5.2 Broadcast Protocol
- **Current token**: Broadcast to all KVP GPUs for immediate access
- **KV storage**: Staged across ranks to maintain uniform distribution
- **Hot spot prevention**: No single GPU becomes memory bottleneck

### 6. Parameter Formulations

#### 6.1 KV Cache Read Time
```
Time_KV = (B × 2 × ⌈K/TPA⌉ × Hsz × S) / (KVP × MemBW) × bytes_param
```

#### 6.2 Weight Read Time
```
Time_Weights = ((2 × H × Q/TPA × Hsz) + (2 × H × ⌈K/TPA⌉ × Hsz) + (3 × H × F/TPF)) / MemBW × bytes_param
```

### 7. Dimensional Specifications
- **B**: Batch size
- **Q**: Query heads (128 for Llama-405B)
- **K**: KV heads (8 for Llama-405B)  
- **Hsz**: Attention head size (128)
- **H**: Hidden dimension = Q × Hsz
- **F**: Intermediate FFN dimension (65536)
- **S**: KV sequence length (up to 1M+ tokens)
- **TPA**: TP width for attention (TPA ≤ K)
- **TPF**: TP width for FFN
- **KVP**: KV parallelism width
- **EP**: Expert parallelism width (for MoE)
- **N**: Total GPUs = KVP × TPA

### 8. Hardware Assumptions
- **Memory bandwidth**: 8000 GB/s (GB200 NVL72)
- **Precision**: FP4 for weights and KV cache
- **Network**: Large NVLink domains (Blackwell optimized)
- **Scale**: 1-64 GPUs within single GB200 node