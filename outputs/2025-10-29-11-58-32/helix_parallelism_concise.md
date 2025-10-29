# Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding

**Authors**: Nidhi Bhatia, Ankit More, Ritika Borkar, Tiyasa Mitra, Ramon Matas, Ritchie Zhao, Max Golub, Dheevatsa Mudigere, Brian Pharris, Bita Rouhani

## Abstract

As LLMs scale to multi-million-token KV histories, real-time autoregressive decoding under tight Token-to-Token Latency (TTL) constraints faces growing pressure. Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size. Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency.

We introduce Helix Parallelism, a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TP×Expert Parallel (EP) in MoEs during FFN computation. To preserve exact attention behavior, Helix includes a lightweight communication step. To minimize the exposed communication cost, we introduce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead through batchwise overlap, preserving low TTL while improving GPU efficiency.

Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32× larger batches under the same latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto on Blackwell and making real-time inference with ultra-long-sequence practical.

## 1. Introduction

Large Language Models (LLMs) must handle ultra-long histories (millions of tokens) while delivering millisecond-level Token-to-Token Latency (TTL). This exposes two fundamental bottlenecks:

1. **KV cache reads**: Linear scaling with context length and batch size, overwhelming DRAM capacity and bandwidth
2. **FFN weight reads**: Small batch sizes cannot amortize the cost of loading large FFN weights

Modern attention variants like GQA, MQA, and MLA reduce KV-cache pressure by merging keys/values into shared representations (K << Q). However, Tensor Parallelism (TP) has a critical limitation: when TP > K (number of KV heads), each shard must store full KV cache copies, creating inefficiencies.

## 2. Helix Parallelism

### 2.1 Core Architecture
Helix introduces a temporal pipeline where N GPUs are reused across attention and FFN phases with different parallelism strategies:

1. **Attention phase**: Applies KV Parallelism (KVP) to shard KV cache across sequence dimension
2. **FFN phase**: Reuses same GPUs for TP (dense) or TP×EP (MoE)
3. **Communication**: Lightweight All-to-All with batch-wise overlap (HOP-B)

### 2.2 Attention Partitioning

#### KV Parallelism (KVP)
- **Configuration**: N = KVP × TPA GPUs (TPA ≤ K)
- **Sharding**: KV cache along sequence dimension → S/KVP tokens per GPU
- **QKV computation**: Each GPU independently computes full projections:
  - Input: [B, H]
  - Weights: WQ (H×H/TPA), WK (H×⌈K/TPA⌉·Hsz), WV (H×⌈K/TPA⌉·Hsz)
- **Communication**: Single All-to-All over query-head axis → exact attention

#### Communication Optimization
- **HOP-B**: Overlaps All-to-All with next token computation
- **Scalability**: Communication volume = B×H (independent of sequence length S)

### 2.3 FFN Partitioning
- **Dense models**: TPF = N (all GPUs in TP)
- **MoE models**: N GPUs → TPF×EP grid with expert routing
- **Communication**: TP All-Reduce after FFN computation

### 2.4 KV Concatenation
- **Staged updates**: Round-robin KV appends (e.g., 16 tokens per rank)
- **Balanced growth**: Prevents memory hotspots
- **Broadcast**: Current token available to all KVP GPUs

## 3. Evaluation

### 3.1 Experimental Setup
- **Hardware**: GB200 NVL72, FP4 precision
- **Models**: Llama-405B (dense), DeepSeek-R1 (671B MoE)
- **Context**: 1 million token KV cache simulation
- **Baseline**: Exhaustive search over TP, EP, PP, KVP configurations (>100,000 points)

### 3.2 Results

#### DeepSeek-R1 (MoE)
- **TTL improvement**: 1.5× reduction
- **Batch scaling**: 32× more concurrent users
- **Throughput**: 32× higher tokens/sec/GPU

#### Llama-405B (Dense)
- **Interactivity**: 1.13× improvement
- **Throughput**: 4× higher vs. TP sharding
- **Batch capacity**: 4× larger batches

### 3.3 HOP-B Impact
- **Without HOP-B**: Sequential communication → 12% degradation (Llama)
- **With HOP-B**: Overlapped communication → recovers lost performance
- **DeepSeek-R1**: ~1% impact due to communication being small fraction of total

## 4. Dimensional Specifications

### Model Parameters
- **B**: Batch size
- **Q**: Query heads (128 Llama-405B)
- **K**: KV heads (8 Llama-405B)
- **Hsz**: Head size (128)
- **H**: Hidden dimension (16,384 = Q×Hsz)
- **F**: FFN dimension (65,536)
- **S**: Sequence length (1M+ tokens)

### Parallelism Parameters
- **TPA**: Attention TP width (1 ≤ TPA ≤ K)
- **TPF**: FFN TP width (1 ≤ TPF ≤ 64)
- **KVP**: KV parallelism width (1 ≤ KVP ≤ 32)
- **EP**: Expert parallelism (for MoE)
- **N**: Total GPUs = KVP × TPA ≤ 64

### Performance Formulas
```
KV read time: (B×2×⌈K/TPA⌉×Hsz×S)/(KVP×MemBW)×bytes_param
Weight read: ((2×H×Q/TPA×Hsz)+(2×H×⌈K/TPA⌉×Hsz)+(3×H×F/TPF))/MemBW×bytes_param
```

## 5. Conclusion

Helix Parallelism represents a paradigm shift in decoding efficiency for ultra-long-context LLMs. By decoupling parallelism strategies for attention and FFN layers through a temporal pipeline, Helix overcomes key bottlenecks in long-context decoding, boosting system interactivity and efficiency. The approach is compatible with modern LLM architectures (GQA, MLA, MoE) and aligned with emerging GPU platforms like Blackwell.

## References
[As provided in original paper]