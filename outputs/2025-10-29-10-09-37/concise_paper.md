# Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding

## Abstract
As LLMs scale to multi-million-token KV histories, real-time autoregressive decoding under tight Token-to-Token Latency (TTL) constraints faces growing pressure. Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size. Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency.

We introduce Helix Parallelism, a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TP×Expert Parallel (EP) in MoEs during FFN computation. To preserve exact attention behavior, Helix includes a lightweight communication step. To minimize the exposed communication cost, we introduce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead through batchwise overlap, preserving low TTL while improving GPU efficiency.

Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32× larger batches under the same latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto on Blackwell and making real-time inference with ultra-long-sequence practical.

## 1 Introduction
As LLMs handle multi-million-token contexts, real-time decoding faces fundamental bottlenecks: (1) KV cache reads scaling linearly with context length and batch size, and (2) FFN weight reads requiring significant DRAM bandwidth. Traditional Tensor Parallelism (TP) fails when TP width exceeds KV heads (K), causing inefficient KV duplication. Helix decouples attention and FFN parallelism through temporal pipelining.

## 2 Helix Parallelism

### 2.1 Attention Partitioning

**KV Parallelism (KVP)**: Shards KV cache along sequence dimension across N = KVP×TPA GPUs, where TPA ≤ K to avoid duplication. Each GPU computes QKV projections independently and uses FlashAttention on its sequence shard.

**Communication**: Single All-to-All exchange across query-head dimension, volume independent of sequence length (scales with B×H only).

**HOP-B Overlap**: Batch-wise pipelining hides communication latency behind computation.

### 2.2 FFN Partitioning

Post-attention, same N GPUs are reconfigured for FFN:
- **Dense models**: TP across all N GPUs (TPF = N)
- **MoE models**: TP×EP grid (TPF×EP = N)

### 2.3 Distributed KV Updates
Round-robin KV append strategy (e.g., 16 tokens per GPU) ensures balanced memory growth across KVP ranks.

## 3 Evaluation

### 3.1 Setup
- **Hardware**: GB200 NVL72 with FP4 precision
- **Models**: DeepSeek-R1 (671B MoE, MLA) and Llama-405B (405B dense, GQA)
- **Context**: 1M tokens
- **Baselines**: TP, PP, EP, vanilla KVP
- **Configs**: >100,000 simulated combinations

### 3.2 Results

**DeepSeek-R1**: 1.5× TTL reduction, 32× batch scaling, significant throughput improvement

**Llama-405B**: 1.13× TTL improvement, 4× batch scaling vs TP baseline

### 3.3 Ablation
HOP-B overlap critical for Llama-405B (12% degradation when disabled) but minimal impact on DeepSeek-R1 (1% degradation).

## 4 Related Work
Unlike sequence parallelism for training/prefill, Helix specifically targets decoding bottlenecks with temporal pipeline and exact attention preservation.

## 5 Conclusion
Helix Parallelism enables practical real-time inference with multi-million-token contexts through hybrid sharding and temporal pipelining.

## Technical Specifications

### Model Dimensions
- **B**: Batch size
- **Q**: Query heads (128)
- **K**: KV heads (8 for Llama, 1 for DeepSeek-R1 MLA)
- **Hsz**: Head size (128)
- **H**: Hidden dimension = Q×Hsz
- **F**: FFN hidden dimension (65536)
- **S**: Sequence length (1M tokens)

### Memory Requirements
- **KV Cache per GPU**: 2×⌈K/TPA⌉×Hsz×(S/KVP)×bytes_param
- **FFN Weights**: (3×H×F/TPF)×bytes_param

### Performance Equations
- **KV Read Time**: (B×2×⌈K/TPA⌉×Hsz×S)/(KVP×MemBW)
- **FFN Read Time**: ((2×H×Q/TPA×Hsz)+(3×H×F/TPF))/MemBW

### Supported Configurations
- **Dense Models**: KVP×TPA → TP (TPF = N)
- **MoE Models**: KVP×TPA → TP×EP (TPF×EP = N)
- **Constraint**: TPA ≤ K (number of KV heads)
- **Hardware**: 1-64 GPUs within GB200 node
- **Precision**: FP4 (4-bit floating point)