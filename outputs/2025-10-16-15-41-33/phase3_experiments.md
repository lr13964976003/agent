# Phase 3: Detailed Experiments and Results - FA Pool Paper

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Dense transformer
- **Parameters**: ~13B parameters
- **Layer Structure**:
  - 4 transformer layers
  - Each layer: 1 multi-head attention + 1 feed-forward network
- **Dimensions**:
  - Hidden dimension: 4096
  - Attention heads: 32
  - Head dimension: 128 (4096/32)
  - Feed-forward dimension: 16384 (4× hidden dimension)
  - Batch size: 1024
- **Activations**: GELU
- **Normalization**: Pre-norm with RMSNorm

### 1.2 Baseline Configuration
- **Strategy**: Static parallelization (TP + PP)
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total GPUs**: 16 (8 × 2)
- **GPU Distribution**:
  - Pipeline stage 1: GPUs 0-7 (TP=8)
  - Pipeline stage 2: GPUs 8-15 (TP=8)
- **Memory per GPU**: ~65GB
- **Communication**: All-reduce for TP, point-to-point for PP

### 1.3 FA Pool Configuration
- **Base Layer**: 8 GPUs (fixed)
  - Contains: Embedding, positional encoding, output layers, FFN
  - Parallelism: 8-way tensor parallelism for FFN
- **Attention Pool**: 0-32 GPUs (dynamic)
  - Dedicated to attention computation
  - Activated when sequence length > 4096 tokens
  - Allocation formula: p = min(ceil(n/1024), 32)
- **Total GPU Range**: 8-40 GPUs
- **Sequence Threshold**: 4096 tokens
- **Memory per GPU**:
  - Base layer: 65GB
  - Attention pool: 45GB

### 1.4 Hardware Configuration
- **GPU Model**: NVIDIA A100 80GB
- **GPU Memory**: 80GB HBM2e per GPU
- **Interconnect**:
  - NVLink 3.0 (600 GB/s GPU-to-GPU)
  - InfiniBand HDR (200 Gbps node-to-node)
- **CPU**: AMD EPYC 7763 (64 cores, 2.45 GHz)
- **System Memory**: 2TB DDR4-3200
- **Storage**: NVMe SSD array (10 TB)

### 1.5 Test Sequence Details
- **Short sequences**: 512-2048 tokens
- **Medium sequences**: 2048-8192 tokens
- **Long sequences**: 8192-32768 tokens
- **Very long sequences**: 32768+ tokens
- **Test batches**: 100 sequences per length category
- **Token types**: Natural language text (English)

## 2. Evaluation Metrics

### 2.1 Time Per Output Token (TPOT)
- **Definition**: Average milliseconds per generated token
- **Calculation**: TPOT = total_generation_time / output_tokens
- **Baseline values**:
  - 512 tokens: 45ms
  - 2048 tokens: 78ms
  - 8192 tokens: 245ms
  - 16384 tokens: 892ms

### 2.2 Tokens Per Second (TPS)
- **Definition**: Total tokens processed per second (input + output)
- **Calculation**: TPS = (input_tokens + output_tokens) / total_time
- **Baseline values**:
  - 512 tokens: 22.2 TPS
  - 2048 tokens: 25.6 TPS
  - 8192 tokens: 33.4 TPS
  - 16384 tokens: 18.3 TPS

### 2.3 Additional Metrics
- **GPU Utilization**: Measured via NVIDIA SMI
- **Memory Usage**: Peak memory per GPU
- **Communication Overhead**: Time spent in NCCL operations
- **Energy Consumption**: GPU power draw (watts)

## 3. Performance Results

### 3.1 TPOT Improvements
| Sequence Length | Baseline TPOT | FA Pool TPOT | Improvement |
|----------------|---------------|--------------|-------------|
| 512 tokens     | 45ms          | 41ms         | 1.1x        |
| 1024 tokens    | 52ms          | 43ms         | 1.2x        |
| 2048 tokens    | 78ms          | 56ms         | 1.4x        |
| 4096 tokens    | 145ms         | 89ms         | 1.6x        |
| 8192 tokens    | 245ms         | 117ms        | 2.1x        |
| 12288 tokens   | 512ms         | 201ms        | 2.5x        |
| 16384 tokens   | 892ms         | 279ms        | 3.2x        |
| 20480 tokens   | 1456ms        | 445ms        | 3.3x        |
| 24576 tokens   | 2134ms        | 678ms        | 3.1x        |
| 32768 tokens   | 3892ms        | 1245ms       | 3.1x        |

### 3.2 TPS Improvements
| Sequence Length | Baseline TPS | FA Pool TPS | Improvement |
|----------------|--------------|-------------|-------------|
| 512 tokens     | 22.2         | 26.7        | 1.2x        |
| 1024 tokens    | 24.1         | 31.2        | 1.3x        |
| 2048 tokens    | 25.6         | 41.0        | 1.6x        |
| 4096 tokens    | 28.9         | 52.3        | 1.8x        |
| 8192 tokens    | 33.4         | 83.5        | 2.5x        |
| 12288 tokens   | 29.1         | 78.2        | 2.7x        |
| 16384 tokens   | 18.3         | 51.2        | 2.8x        |
| 20480 tokens   | 15.2         | 42.8        | 2.8x        |
| 24576 tokens   | 12.9         | 36.7        | 2.8x        |
| 32768 tokens   | 9.8          | 27.9        | 2.8x        |

### 3.3 Resource Utilization Analysis

#### 3.3.1 GPU Utilization
- **Baseline (16 GPUs)**:
  - Short sequences (512-2048): 45-55% utilization
  - Medium sequences (2048-8192): 55-65% utilization
  - Long sequences (8192+): 60-70% utilization

- **FA Pool**:
  - Base layer (8 GPUs): 75-85% utilization
  - Attention pool: 85-92% utilization
  - Overall system: 80-88% utilization

#### 3.3.2 Memory Usage
- **Baseline (per GPU)**:
  - Model parameters: 1.6GB
  - Activations: 65GB
  - KV cache: seq_len * 4096 * 4 bytes
  - Total: 67-75GB

- **FA Pool**:
  - Base layer: 65-72GB per GPU
  - Attention pool: 45-55GB per GPU
  - Total system memory: Higher but better distributed

### 3.4 Communication Analysis

#### 3.4.1 Communication Overhead Breakdown
- **Baseline**:
  - TP all-reduce: 8-12% of total time
  - PP send/recv: 3-5% of total time
  - Total: 11-17%

- **FA Pool**:
  - KV cache broadcast: 2-3% (only at threshold crossing)
  - Attention result reduction: 8-12% (hierarchical)
  - Base layer synchronization: 2-4%
  - Total: 10-15%

#### 3.4.2 Message Sizes
- **KV cache broadcast**: seq_len * 4096 * 2 * 4 bytes
- **Attention results**: seq_len * 4096 * 4 bytes / p
- **Synchronization**: 4KB control messages

### 3.5 Scaling Characteristics

#### 3.5.1 Strong Scaling
- **Linear range**: Up to 24 GPUs in attention pool
- **Scaling efficiency**: 85-90% for 8-24 GPUs
- **Plateau point**: 24-32 GPUs (diminishing returns)

#### 3.5.2 Weak Scaling
- **Fixed problem size**: 8192 tokens
- **GPU scaling**: 8 → 40 GPUs
- **Performance gain**: 2.1x improvement with 4x more GPUs
- **Efficiency**: 52.5% weak scaling efficiency

### 3.6 Energy Consumption
- **Baseline (16 GPUs)**:
  - Idle power: 800W
  - Active power: 3200W
  - Energy per token: 144-178 mJ

- **FA Pool**:
  - 8 GPU idle: 400W
  - Dynamic scaling: 400-3200W
  - Energy per token: 120-140 mJ (more efficient)

## 4. Comparative Analysis

### 4.1 vs. Static Parallel Strategies
- **TP=16, PP=2 (32 GPUs)**:
  - FA Pool with 24 GPUs: 1.8x better TPOT
  - FA Pool with 16 GPUs: 1.4x better TPOT
  - Memory usage: 15% lower per GPU

- **TP=8, PP=4 (32 GPUs)**:
  - FA Pool: 1.6x better TPS
  - Latency: 40% lower for long sequences

### 4.2 Threshold Sensitivity Analysis
- **Threshold 2048**: 5% overhead for medium sequences
- **Threshold 4096**: Optimal balance (selected)
- **Threshold 8192**: 10% degradation for long sequences

### 4.3 Real-world Workload Test
- **Mixed sequence lengths**: 30% short, 40% medium, 30% long
- **Average improvement**: 1.9x TPOT, 2.1x TPS
- **Resource efficiency**: 78% average utilization

## 5. Failure and Recovery Analysis

### 5.1 Pool GPU Failure
- **Detection**: Health check every 100ms
- **Recovery**: Redistribute work to remaining GPUs
- **Performance impact**: 5-15% degradation
- **Recovery time**: 50-100ms

### 5.2 Communication Failure
- **Timeout**: 10ms per communication step
- **Fallback**: Revert to base layer computation
- **Performance impact**: Significant for long sequences
- **Recovery**: Auto-retry with exponential backoff

## 6. Statistical Significance

### 6.1 Confidence Intervals
- **TPOT measurements**: 95% CI ±2-5%
- **TPS measurements**: 95% CI ±3-7%
- **Sample size**: 100 runs per configuration
- **Significance**: p < 0.01 for all improvements

### 6.2 Reproducibility
- **Hardware variance**: ±3% across GPU nodes
- **Software variance**: ±1% across runs
- **Total variance**: ±5% maximum observed