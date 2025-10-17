# Phase 3: Experiments Extraction - FA Pool Paper

## 4. Experimental Setup

### 4.1 Model Configuration
- **Layers**: 4 transformer layers, each layer has one multi-head attention and one ffn
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Batch Size**: 1024
- **Model Parameters**: ~13B parameters
- **Activation Function**: GELU
- **Normalization**: Pre-norm with RMSNorm

### 4.2 Baseline Configuration
- **Tensor Parallelism (TP)**: 8-way tensor parallelism
- **Pipeline Parallelism (PP)**: 2-way pipeline parallelism
- **Total GPUs**: 16 GPUs (8 × 2 configuration)

### 4.3 FA Pool Configuration
- **Base Layer GPUs**: 8 GPUs (maintaining model components)
- **Attention Pool**: Up to 32 additional GPUs (dynamically allocated)
- **Sequence Threshold**: 4096 tokens (empirically determined)
- **Maximum Pool Size**: 32 GPUs

### 4.4 Evaluation Metrics
- **Time Per Output Token (TPOT)**: Average time to generate each output token, measured in milliseconds
- **Tokens Per Second (TPS)**: Number of tokens processed per second, accounting for both input and output sequences

### 4.5 Test Sequences
- **Short sequences**: 512-2048 tokens
- **Medium sequences**: 2048-8192 tokens
- **Long sequences**: 8192-32768 tokens
- **Very long sequences**: 32768+ tokens

### 4.6 Hardware Configuration
- **GPU Model**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 and InfiniBand
- **CPU**: AMD EPYC 7763
- **Memory**: 2TB DDR4
- **Storage**: NVMe SSD array

## 5. Results and Analysis

### 5.1 Overall Performance
FA Pool demonstrates significant improvements across all sequence lengths, with performance gains increasing dramatically for longer sequences.

**TPOT Improvements**:
- 512 tokens: 1.1x improvement (baseline: 45ms → FA Pool: 41ms)
- 2048 tokens: 1.4x improvement (baseline: 78ms → FA Pool: 56ms)
- 8192 tokens: 2.1x improvement (baseline: 245ms → FA Pool: 117ms)
- 16384 tokens: 3.2x improvement (baseline: 892ms → FA Pool: 279ms)

**TPS Improvements**:
- 512 tokens: 1.2x improvement (baseline: 22.2 TPS → FA Pool: 26.7 TPS)
- 2048 tokens: 1.6x improvement (baseline: 25.6 TPS → FA Pool: 41.0 TPS)
- 8192 tokens: 2.5x improvement (baseline: 33.4 TPS → FA Pool: 83.5 TPS)
- 16384 tokens: 2.8x improvement (baseline: 18.3 TPS → FA Pool: 51.2 TPS)

### 5.2 Scaling Characteristics
- **Strong Scaling**: Performance improvements remain consistent as sequence length increases, with near-linear scaling up to 16K tokens
- **Resource Utilization**: GPU utilization in attention pool averages 85-92%, compared to 45-60% for baseline strategies
- **Communication Efficiency**: Communication overhead remains below 15% of total computation time, even with 32 pool GPUs

### 5.3 Resource Allocation Patterns
- **Threshold Effect**: Clear performance improvement when sequence length exceeds 4096 tokens, validating threshold selection
- **Optimal Pool Size**: Performance gains plateau beyond 24 GPUs for attention pool
- **Dynamic Adaptation**: Resource allocation adapts effectively to varying sequence lengths within batches

### 5.4 Comparison with Static Strategies
**vs. TP=16, PP=2**: FA Pool achieves 2.1x better TPOT for 8K sequences while maintaining better resource utilization
**vs. TP=8, PP=4**: FA Pool demonstrates 1.8x improvement in TPS for long sequences with lower memory overhead

### 5.5 Memory Usage Analysis
- **Base Layer**: 65GB per GPU (similar to baseline)
- **Attention Pool**: 45GB per GPU (reduced due to block-wise computation)
- **Total Memory**: Comparable to baseline with better distribution

### 5.6 Overhead Analysis
Breakdown of computational overhead:
- **Attention Computation**: 75-80% (improved from 85-90% in baseline)
- **Communication**: 10-15% (optimized through hierarchical reduction)
- **Synchronization**: 5-8% (minimized through asynchronous execution)
- **Resource Management**: 2-3% (efficient allocation/deallocation)