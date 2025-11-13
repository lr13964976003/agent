# Context Parallelism for Scalable Million-Token Inference

## ABSTRACT
We present context parallelism for long-context large language model inference, which achieves near-linear scaling for long-context prefill latency with up to 128 H100 GPUs across 16 nodes. Particularly, our method achieves 1M context prefill with Llama3 405B model in 77s (93% parallelization efficiency, 63% FLOPS utilization) and 128K context prefill in 3.8s. We develop two lossless exact ring attention variants: pass-KV and pass-Q to cover a wide range of use cases with the state-of-the-art performance: full prefill, persistent KV prefill and decode. Benchmarks on H100 GPU hosts inter-connected with RDMA and TCP both show similar scalability for long-context prefill, demonstrating that our method scales well using common commercial data center with medium-to-low inter-host bandwidth.

## 1 INTRODUCTION

Contemporary LLMs require significant computational resources for inference with long context lengths. Context parallelism (CP) is a system optimization technique that improves the latency and scalability of LLM inference without modifying underlying dense attention algorithms. Our work provides the first system implementation details for applying context parallelism in inference scenarios.

**Key Advantages of CP**:
- Compute parallelization reduces latency
- Communication message size reduction (orders of magnitude smaller than TP)
- KV cache distribution enables larger batch sizes

**Contributions**:
- Support for multi-turn prefill and decoding with persistent KV cache
- Optimization for latency with novel pass-KV and pass-Q variants
- Compute and memory load balancing for variable input lengths

## 2 BACKGROUND

### 2.1 Challenges with Long Context LLM
- **Compute**: Quadratic FLOP cost w.r.t. context length
- **Memory**: KV cache scales linearly with context length (405B model requires ~1TB for 1M context)
- **System-level optimizations**: Our work falls in this category, preserving model architecture while improving scalability

### 2.2 Notation
Key dimensions:
- NH = 128 (query heads)
- NKV = 8 (key/value heads)
- D = 16,384 (model dimension)
- DH = D/NH = 128 (head dimension)

## 3 CONTEXT PARALLEL INFERENCE

### 3.1 Model Parallelization Strategy
- **Tensor Parallelism (TP8)**: Within each node (8 GPUs)
- **Context Parallelism (CP)**: Across nodes (N nodes total)
- **Hybrid approach**: CP over N nodes with TP8 per node

### 3.2 Three Inference Stages
1. **Full prefill**: Initial prompt processing
2. **Partial prefill**: Follow-up prompts with cached history
3. **Decode**: Auto-regressive token generation

### 3.3 Ring Attention Algorithms

#### Pass-KV Ring Attention
- **Use**: Full prefill, partial prefill with low KV cache hit rate
- **Communication**: Pass KV embeddings around ring
- **Size**: 2×(T+P)×D×e×NKV/NH bytes

#### Pass-Q Ring Attention  
- **Use**: Decode, partial prefill with high KV cache hit rate
- **Communication**: Pass Q embeddings around ring
- **Size**: T×D×e bytes

#### Adaptive Selection Heuristic
Decision based on KV cache miss rate and context length:
```
Selection threshold: T/(T+P) ≤ 2×(NKV/NH) = 0.125 (for Llama3 405B)
```

### 3.4 Load Balanced Sharding
- **Full prefill**: Sequence partitioned into 2×N chunks, each rank takes (C_i, C_{2N-i-1})
- **Partial prefill**: Load-balanced sharding applied to new tokens only
- **Decode**: Round-robin sharding to utilize full KV cache capacity

## 4 EXPERIMENTS

### 4.1 Setup
- **Model**: Llama3 405B (126 layers, 16,384 dim, 128 heads, 8 KV heads)
- **Hardware**: H100 GPUs (96GB HBM2e, 2.4TB/s bandwidth)
- **Network**: RDMA (400Gb/s) and TCP (100Gb/s)
- **Quantization**: Row-wise FP8 for feedforward layers

### 4.2 Full Prefill Scaling Results

#### 4.2.1 Latency Reduction with Fixed Context
| Context Length | CP1 | CP2 | CP4 | CP8 |
|---|---|---|---|---|
| 128K tokens | 42010ms | 21042ms | 10950ms | 5850ms |

- Linear scaling achieved with sufficient context length
- TCP network sufficient for up to 4 nodes

#### 4.2.2 CP vs Tensor Parallelism
- **CP8**: 5850ms for 128K context
- **TP64**: 19841ms for 128K context  
- **Speedup**: 3.4× improvement with CP over TP at 8 nodes

#### 4.2.3 Scaling Context Length with Fixed Hardware
| Context Length | CP8 | CP16 |
|---|---|---|
| 128K tokens | 5850ms | 3850ms |
| 256K tokens | 11700ms | 7700ms |
| 512K tokens | 23400ms | 15400ms |
| 1M tokens | 46800ms | 77000ms |

- **1M context**: 77s on 16 nodes (128 GPUs)
- **93% parallelization efficiency** achieved
- **63% FLOPS utilization** compared to theoretical peak

### 4.3 Persistent KV Prefill Results

#### Pass-KV vs Pass-Q Performance
| KV Cache Miss Rate | Pass-KV | Pass-Q | Selection |
|---|---|---|---|
| 1% | 1023ms | 899ms | Pass-Q |
| 5% | 1306ms | 1302ms | Either |
| 10% | 2081ms | 2205ms | Pass-KV |
| 100% | 11462ms | 12361ms | Pass-KV |

#### Decision Threshold
- **Pass-Q preferred**: When KV cache miss rate < 5%
- **Pass-KV preferred**: When KV cache miss rate > 5%
- **Switch boundary**: T = 6400 tokens (5% of 128K)

### 4.4 Decode Performance
| Configuration | TTIT |
|---|---|
| TP8 | 46.26ms |
| CP2 | 60.23ms |
| CP4 | 71.31ms |

- **Decode regression**: CP increases decode latency due to communication overhead
- **Recommendation**: Use CP for prefill optimization, separate systems for decode

## 5 IMPLEMENTATION DETAILS

### 5.1 Communication Implementation
- **Ring SendRecv**: 8-way communication within CP groups
- **CUDA Graphs**: Used for decode to minimize kernel launch overhead
- **Overlap**: Communication hidden under attention computation when possible

### 5.2 Memory Management
- **KV cache**: Distributed across CP ranks (linear scaling with nodes)
- **Model weights**: Replicated across CP ranks, sharded with TP8
- **Capacity**: 1M context requires ~1TB KV cache (distributed across 16 nodes)

### 5.3 Practical Deployment
- **Minimum bandwidth**: 3GB/s per rank for overlap
- **Network requirements**: TCP sufficient for CP up to 4 nodes
- **Scalability**: Linear scaling with nodes for prefill latency

## 6 CONCLUSION

Context parallelism enables efficient scaling of LLM inference for million-token contexts. Our implementation achieves near-linear scaling with up to 128 GPUs, processing 1M context in 77 seconds. The adaptive pass-KV/pass-Q selection and load-balanced sharding make this practical for real-world applications with multi-turn conversations.

## APPENDIX: Key Dimensions
- Model: Llama3 405B
- Layers: 126
- Model dimension: 16,384
- Attention heads: 128
- KV heads: 8
- Parameter size: 405B parameters
- Context window: 128K native, 1M+ with CP scaling