# Helix Parallelism: Methodology Extraction

## Mathematical Foundations

### Key Dimensions
- **B**: Batch Size
- **Q**: Query heads (128 for Llama-405B)
- **K**: KV heads (8 for Llama-405B, 1 for DeepSeek-R1 with MLA)
- **Hsz**: Attention head size (128)
- **S**: KV sequence length (up to 1M+ tokens)
- **H**: Hidden dimension = Q × Hsz
- **F**: Intermediate hidden dimension (65536)
- **TPA**: TP width for Attention (TPA ≤ K)
- **TPF**: TP width for FFN
- **KVP**: KV P width
- **N**: Total GPUs = KVP × TPA

## Phase 1: Attention Partitioning

### 1.1 KV Parallelism (KVP)
**GPU Configuration**: N = KVP × TPA GPUs
**Sequence Sharding**: KV cache sharded along sequence dimension across KVP GPUs

**Per-GPU Memory Requirements**:
- Each KVP GPU holds sequence slice: S/KVP tokens
- KV cache per GPU: 2 × ⌈K/TPA⌉ × Hsz × (S/KVP) × bytes_param
- No KV duplication when TPA ≤ K

**Compute Flow**:
1. **QKV Projection**: Each GPU independently computes full QKV projections
   - Input: [B, H] broadcast to all GPUs
   - Weights: WQ ∈ RH×(H/TPA), WK ∈ RH×(⌈K/TPA⌉·Hsz), WV ∈ RH×(⌈K/TPA⌉·Hsz)
   - Output: Each GPU has full query heads and partial KV heads

2. **FlashAttention**: Each GPU runs FlashAttention on its KV shard
   - Input: Local KV shard [B, ⌈K/TPA⌉, Hsz, S/KVP]
   - Output: Partial attention results + log-sum-exp scalars

3. **All-to-All Communication**: 
   - Exchange fragments across query-head dimension
   - Volume: Independent of S, scales with B×H only
   - Output: Exact softmax-normalized attention

### 1.2 Optimized Communication
**All-to-All Pattern**: Query-head axis exchange
**Communication Volume**: B × H × bytes_per_element (constant w.r.t. sequence length)
**HOP-B Overlap**: Batch-wise pipelining to hide communication behind computation

## Phase 2: Post-Attention Linear Projection

**TP Configuration**: Same N GPUs in TP=KVP×TPA layout
**Weight Sharding**: Linear projection weight W ∈ RH×H sharded as H/N × H per GPU
**Compute**:
1. Each GPU computes local projection: [B, H/N] × [H/N, H] → [B, H]
2. All-Reduce across N GPUs for final [B, H] output

## Phase 3: FFN Partitioning

### Dense Models (EP = 1)
**TP Configuration**: TPF = N (all GPUs)
**Weight Sharding**: FFN weights sharded across N devices
- FC1: H × F → H/N × F per GPU
- FC2: F × H → F × H/N per GPU

**Compute Flow**:
1. **FC1 (Column-parallel)**: [B, H] × [H, F/N] → [B, F/N]
2. **Activation**: GELU/GLU applied element-wise
3. **FC2 (Row-parallel)**: [B, F/N] × [F/N, H] → [B, H]
4. **All-Reduce**: Sum partial results across N GPUs

### MoE Models (EP > 1)
**GPU Grid**: TPF × EP grid from N GPUs
**Expert Assignment**: Tokens routed to appropriate experts
**Within Expert**:
- FC layers: Column-parallel (TPF) → Row-parallel (TPF)
- **All-Reduce**: Intra-expert after FC2
- **All-Gather**: Inter-expert for final aggregation

## Distributed KV Concatenation

**Update Strategy**: Round-robin KV appends across KVP ranks
- **Batch Size**: Fixed tokens per rank (e.g., 16 tokens)
- **Distribution**: Ensures balanced memory growth
- **Synchronization**: New tokens broadcast to all KVP GPUs

## Hardware Mapping

### GB200 NVL72 Configuration
- **Memory Bandwidth**: 8000 GB/s per GPU
- **NVLink**: Large domain for efficient All-to-All
- **Precision**: FP4 (4-bit floating point)
- **GPU Count**: 1-64 GPUs within single node

### Roofline Analysis Equations

**KV Cache Read Time**:
```
T_KV = (B × 2 × ⌈K/TPA⌉ × Hsz × S) / (KVP × MemBW) × bytes_param
```

**FFN Weight Read Time**:
```
T_FFN = ((2 × H × Q/TPA × Hsz) + (2 × H × ⌈K/TPA⌉ × Hsz) + (3 × H×F/TPF)) × bytes_param / MemBW
```

### Communication Patterns

**All-to-All (Attention)**: 
- Shape: [B, H/(KVP×TPA)] exchanged across KVP×TPA devices
- Volume: B × H × bytes_per_element

**All-Reduce (FFN)**:
- Shape: [B, H] reduction across TPF devices
- Volume: B × H × bytes_per_element