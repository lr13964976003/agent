# Helix Parallelism: Key Points Extraction

## Abstract (Retained in full)
As LLMs scale to multi-million-token KV histories, real-time autoregressive decoding under tight Token-to-Token Latency (TTL) constraints faces growing pressure. Two core bottlenecks dominate: accessing Feed-Forward Network (FFN) weights and reading long KV caches. While Tensor Parallelism (TP) helps mitigate the cost of FFN weight reads, it does not scale well for attention. When TP width exceeds the number of KV heads, it leads to inefficient KV duplication, limits parallelism, and constrains batch size. Simultaneously, DRAM reads for long KV histories scale linearly with batch size, further capping efficiency.

We introduce Helix Parallelism, a hybrid execution strategy that applies KV parallelism during attention to shard KV caches across GPUs, then reuses the same GPUs for TP in dense LLMs or TP×Expert Parallel (EP) in MoEs during FFN computation. To preserve exact attention behavior, Helix includes a lightweight communication step. To minimize the exposed communication cost, we introduce Helix HOP-B. Helix HOP-B effectively minimizes communication overhead through batchwise overlap, preserving low TTL while improving GPU efficiency.

Compared to conventional parallelism approaches, Helix reduces TTL by up to 1.5x at fixed batch sizes and supports up to 32× larger batches under the same latency budget for DeepSeek-R1, pushing forward the throughput-latency Pareto on Blackwell and making real-time inference with ultra-long-sequence practical.

## Key Technical Contributions

### Core Problem Identified
1. **KV Cache Scaling**: DRAM reads scale linearly with both context length and batch size
2. **FFN Weight Bottleneck**: Large FFN weights require significant DRAM bandwidth
3. **TP Limitation**: When TP width > KV heads, leads to inefficient KV duplication
4. **Latency Constraints**: Real-time applications need millisecond-level Token-to-Token Latency (TTL)

### Helix Parallelism Solution
1. **Hybrid Strategy**: Combines KV Parallelism (KVP) during attention with TP/EP during FFN
2. **Temporal Pipeline**: Reuses same GPUs for different phases (attention → FFN)
3. **Exact Attention**: Preserves exact attention behavior without approximation
4. **HOP-B Optimization**: Batch-wise overlap to hide communication latency

### Architecture Compatibility
- **Attention Types**: GQA, MQA, MLA
- **Model Types**: Dense LLMs and MoE models
- **Hardware**: Blackwell GPU systems with NVLink domains
- **Precision**: FP4 low-precision support

### Performance Gains
- **TTL Reduction**: Up to 1.5× at fixed batch sizes
- **Batch Scaling**: Up to 32× larger batches under same latency budget
- **Throughput**: Significant improvements in tokens/sec/GPU
- **Context Length**: Enables multi-million-token inference