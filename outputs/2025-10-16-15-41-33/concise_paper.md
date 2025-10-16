# FA Pool: A Dynamic Parallel Strategy for Scaling Attention Mechanisms in Large Language Models (Concise Version)

## Abstract

The computational complexity of attention mechanisms in transformer-based models grows quadratically with sequence length, creating a significant bottleneck for processing long sequences. We propose FA Pool (Flash Attention Pool), a novel dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds. When input sequences exceed a predetermined length, FA Pool activates additional GPU resources to form a computation pool dedicated to parallel attention calculations, thereby reducing the computational burden on individual GPUs. Our approach combines the benefits of Flash Attention's memory-efficient algorithms with dynamic resource allocation to achieve superior scaling characteristics. Experimental results on a 4-layer Dense model demonstrate that FA Pool achieves significant improvements in both Time Per Output Token (TPOT) and Tokens Per Second (TPS) metrics compared to traditional static parallelization strategies (TP=8, PP=2 baseline). The strategy shows particular effectiveness for long sequence processing, achieving up to 3.2x improvement in TPOT and 2.8x improvement in TPS for sequences exceeding 8K tokens.

## 1. Introduction

Large language models face significant challenges due to the O(n²) complexity of attention mechanisms with respect to sequence length. Traditional static parallelization strategies (TP and PP) lead to suboptimal resource utilization when dealing with variable sequence lengths. We introduce FA Pool, a dynamic parallel strategy that addresses these limitations through adaptive resource allocation based on sequence length thresholds, combining Flash Attention efficiency with dynamic GPU resource scaling.

## 2. FA Pool Methodology

### 2.1 System Architecture

**Base Layer**: 8 GPUs maintaining core model components (embedding, positional encoding, output layers, FFN)
- Fixed allocation
- Maintains model coherence
- 8-way tensor parallelism for FFN

**Attention Pool**: 0-32 additional GPUs dynamically allocated
- Dedicated to attention computation
- Activated when sequence length > 4096 tokens
- GPU allocation: p = min(ceil(n/1024), 32)

**Resource Manager**: Monitors sequence length and manages allocation/deallocation

### 2.2 Model Configuration
- **Architecture**: 4-layer Dense transformer
- **Parameters**: ~13B parameters
- **Dimensions**:
  - Hidden dimension: 4096
  - Attention heads: 32
  - Feed-forward dimension: 16384
  - Batch size: 1024
- **Layers**: Each layer has 1 multi-head attention + 1 FFN

### 2.3 Dynamic Resource Allocation

**Sequence Threshold**: 4096 tokens (empirically determined)

**Allocation Formula**:
```
if sequence_length <= 4096:
    pool_gpus = 0
else:
    pool_gpus = min(ceil(sequence_length/1024), 32)
```

**Attention Parallelization**:
```
Block size: b = ceil(sequence_length / pool_gpus)
Per GPU computation:
- Local query: Q_i = Q[i*b:(i+1)*b] 
- Full key/value cache: K_i = K[0:n], V_i = V[0:n] (replicated)
- Result: O_i = FlashAttention(Q_i, K, V)
- Aggregation: O = concat(O_0, O_1, ..., O_{p-1})
```

### 2.4 Communication Optimization
- **KV Cache Sharing**: Full replication across pool GPUs
- **Asynchronous Execution**: Attention computation overlaps with FFN
- **Hierarchical Reduction**: Tree-based result aggregation (log2(pool_size) steps)

### 2.5 Memory Allocation
- **Base Layer GPUs**: 65GB per GPU
- **Attention Pool GPUs**: 45GB per GPU
- **KV Cache**: seq_len * 4096 * 2 * 4 bytes per GPU

## 3. Experimental Setup

### 3.1 Baseline Configuration
- **Strategy**: Static parallelization
- **Tensor Parallelism**: 8-way
- **Pipeline Parallelism**: 2-way
- **Total GPUs**: 16
- **Configuration**: TP=8, PP=2

### 3.2 FA Pool Configuration
- **Base Layer**: 8 GPUs (fixed)
- **Maximum Pool**: 32 GPUs (dynamic)
- **Total Range**: 8-40 GPUs
- **Sequence Threshold**: 4096 tokens

### 3.3 Hardware Setup
- **GPU**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 + InfiniBand
- **CPU**: AMD EPYC 7763
- **Memory**: 2TB DDR4

### 3.4 Evaluation Metrics
- **TPOT**: Time Per Output Token (ms)
- **TPS**: Tokens Per Second (input + output)

## 4. Results

### 4.1 Performance Improvements
| Sequence Length | TPOT Improvement | TPS Improvement | Pool GPUs |
|----------------|------------------|-----------------|-----------|
| 512 tokens     | 1.1x             | 1.2x            | 0         |
| 2048 tokens    | 1.4x             | 1.6x            | 2         |
| 8192 tokens    | 2.1x             | 2.5x            | 8         |
| 16384 tokens   | 3.2x             | 2.8x            | 16        |

### 4.2 Resource Utilization
- **GPU Utilization**: 85-92% (pool) vs 45-60% (baseline)
- **Communication Overhead**: <15% of total time
- **Scaling Efficiency**: Near-linear up to 16K tokens

### 4.3 Comparative Analysis
- **vs TP=16,PP=2 (32 GPUs)**: FA Pool (24 GPUs) achieves 1.8x better TPOT
- **Memory Efficiency**: 15% lower per GPU memory usage
- **Energy Efficiency**: 120-140 mJ/token vs 144-178 mJ/token (baseline)

## 5. Key Technical Details for Deployment

### 5.1 Model Parameters
- **Total Parameters**: ~13B
- **Layer Count**: 4
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **FFN Dimension**: 16384

### 5.2 GPU Requirements
- **Base GPUs**: 8 (fixed)
- **Pool GPUs**: 0-32 (dynamic)
- **Memory per GPU**: 65GB (base), 45GB (pool)
- **Minimum Total**: 8 GPUs
- **Maximum Total**: 40 GPUs

### 5.3 Threshold Configuration
- **Activation Threshold**: 4096 tokens
- **GPU Scaling**: 1 GPU per 1024 tokens above threshold
- **Maximum Pool**: 32 GPUs

### 5.4 Communication Patterns
- **KV Cache Broadcast**: seq_len * 4096 * 2 * 4 bytes
- **Result Reduction**: seq_len * 4096 * 4 bytes / pool_size
- **Synchronization**: Hierarchical tree reduction

## 6. Implementation Notes

### 6.1 Flash Attention Integration
- **Block Size**: 256 tokens for local computation
- **Memory Efficient**: Avoids full attention matrix materialization
- **Sequence Packing**: Efficient batching of variable lengths

### 6.2 Dynamic Resource Management
- **Allocation Time**: 50-100ms for new GPU activation
- **Deallocation Time**: Near-instantaneous
- **Health Monitoring**: 100ms interval checks
- **Failure Recovery**: Automatic redistribution to remaining GPUs

### 6.3 Optimization Strategies
- **Asynchronous Execution**: Overlap attention and FFN computation
- **Memory Pooling**: Reuse GPU memory across allocations
- **Communication Batching**: Group small messages to reduce overhead