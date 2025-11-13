# Phase 2: Methodology Extraction - Context Parallelism for Scalable Million-Token Inference

## 1. Context Parallelism Architecture

### 1.1 Basic CP Design
- **Sequence Dimension Sharding**: Input tokens distributed along sequence length across multiple GPUs
- **KV Cache Distribution**: Key-value embeddings distributed across CP ranks
- **Hybrid Parallelism**: CP + Tensor Parallelism (TP8 within each node)

### 1.2 Parallelization Strategy
```
System Configuration:
- N CP nodes/ranks
- TP8 within each node (8 GPUs per node)
- Total GPUs: N × 8
- CP communication group: One per KV head (8 total)
```

## 2. Ring Attention Algorithms

### 2.1 Pass-KV Ring Attention
**Use Case**: Full prefill (P = 0) and partial prefill with low KV cache hit rate

**Algorithm**:
1. **Load-balanced sharding**: Sequence partitioned into 2×N chunks, each rank takes (C_i, C_{2N-i-1})
2. **KV distribution**: Pass KV embeddings of length max(P_i) + ⌈T/N⌉ around ring
3. **Computation**: N partial attention computations per rank
4. **Merge**: Combine partial results using merge attention operator

**Communication Pattern**:
- Ring SendRecv for KV embeddings
- Overlap communication with attention computation
- Message size: 2×(T+P)×D×e×NKV/NH bytes

### 2.2 Pass-Q Ring Attention
**Use Case**: Decode and partial prefill with high KV cache hit rate

**Algorithm**:
1. **Query distribution**: Pass Q embeddings around ring
2. **Stationary KV**: Keep K/V embeddings stationary
3. **Partial results**: Attention outputs scattered across ranks
4. **Recovery**: All2All to restore partial outputs to source ranks

**Communication Pattern**:
- Ring SendRecv for Q embeddings
- All2All for partial attention outputs
- Message size: T×D×e bytes for Q, additional All2All overhead

### 2.3 Adaptive Selection Heuristic
**Decision Criteria**:
```
if T ≥ (N×C×NKV×e)/(2×NH×BW) OR T/(T+P) ≥ 2×(NKV/NH)
    use pass-KV
else
    use pass-Q
```

**Parameters**:
- T: new token length
- P: cached KV length
- N: number of CP ranks
- C: peak compute capacity
- BW: communication bandwidth
- NH: number of query heads (128)
- NKV: number of KV heads (8)
- e: bytes per element

## 3. Load Balancing

### 3.1 Full Prefill Sharding
- **Sequence partitioning**: Even split into 2×N chunks
- **Chunk assignment**: Rank i takes chunks (C_i, C_{2N-i-1})
- **Padding**: Handle variable sequence lengths with padding

### 3.2 Partial Prefill Sharding
- **New token focus**: Load-balanced sharding applied only to new tokens
- **KV cache**: Distributed based on original cached tokens
- **Fused sequences**: Handle multiple sequences in batch

### 3.3 Decode Sharding
- **Round-robin**: Offset by 1 index each decode iteration
- **Token distribution**: Even sharding across ranks
- **KV cache capacity**: Utilize full capacity from all CP ranks

## 4. Multi-Turn Conversation Support

### 4.1 Persistent KV Cache
- **Storage**: KV embeddings preserved between conversation turns
- **Distribution**: Maintained across CP ranks
- **Access pattern**: Partial prefill attends to both new and cached tokens

### 4.2 Prefill/Decode Pipeline
**Three Stages**:
1. **Full prefill**: Initial prompt processing
2. **Partial prefill**: Follow-up prompts with cached history
3. **Decode**: Auto-regressive token generation

## 5. Communication Optimization

### 5.1 Ring Communication
- **Pattern**: 8-way SendRecv within CP groups
- **Overlap**: Communication hidden under computation
- **Bandwidth efficiency**: Optimized for low inter-node bandwidth

### 5.2 Message Size Analysis
**Pass-KV**:
- Message: 2×(T+P)×D×e×NKV/NH bytes
- Advantage: Smaller for GQA models (16× reduction for Llama3 405B)

**Pass-Q**:
- Message: T×D×e bytes
- Advantage: Smaller when T << (T+P)

### 5.3 Bandwidth Requirements
- **RDMA (400Gb/s)**: Full overlap achieved
- **TCP (100Gb/s)**: Sufficient for pass-KV up to 4 nodes
- **Threshold**: ~3GB/s per rank minimum for overlap

## 6. Implementation Details

### 6.1 Model Configuration
- **Model**: Llama3 405B
- **Quantization**: Row-wise FP8 for feedforward layers
- **Attention**: Flash Attention 3 for prefill, Flash Decoding for decode
- **CUDA optimizations**: CUDA Graphs for decode

### 6.2 Memory Layout
- **TP8**: Model fits in 8×96GB H100
- **KV head distribution**: 1 KV head per GPU in TP group
- **Parameter sharding**: Alternating column/row parallelism

### 6.3 Performance Considerations
- **Batch size**: 1 for latency measurements
- **Context window**: 128K native, 1M with CP scaling
- **Efficiency metrics**: Parallelization efficiency 93%, FLOPS utilization 63%