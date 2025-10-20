# FA Pool: Experiments Extraction

## Abstract (Retained)
The computational complexity of attention mechanisms in transformer-based models grows quadratically with sequence length, creating a significant bottleneck for processing long sequences. We propose FA Pool (Flash Attention Pool), a novel dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds. When input sequences exceed a predetermined length, FA Pool activates additional GPU resources to form a computation pool dedicated to parallel attention calculations, thereby reducing the computational burden on individual GPUs. Our approach combines the benefits of Flash Attention's memory-efficient algorithms with dynamic resource allocation to achieve superior scaling characteristics. Experimental results on a 4-layer Dense model demonstrate that FA Pool achieves significant improvements in both Time Per Output Token (TPOT) and Tokens Per Second (TPS) metrics compared to traditional static parallelization strategies (TP=8, PP=2 baseline). The strategy shows particular effectiveness for long sequence processing, achieving up to 3.2x improvement in TPOT and 2.8x improvement in TPS for sequences exceeding 8K tokens.

## Detailed Experimental Setup and Results

### 4.1 Model Configuration Details

#### 4-Layer Dense Model Specifications
- **Architecture**: Standard transformer decoder
- **Total Layers**: 4 transformer layers
- **Layer Structure per Layer**:
  - Multi-head self-attention (32 heads, 128 head_dim)
  - Feed-forward network (16384 hidden dimension)
  - Residual connections and RMSNorm pre-normalization
- **Hidden Size**: 4096 (consistent across all layers)
- **Attention Heads**: 32 parallel attention heads
- **Head Dimension**: 128 (4096 ÷ 32 = 128)
- **Feed-forward Dimension**: 16384 (4× hidden size)
- **Batch Size**: 1024 sequences per batch
- **Total Parameters**: ~13 billion parameters
- **Activation Function**: GELU (Gaussian Error Linear Unit)
- **Normalization**: RMSNorm (Root Mean Square Normalization)

#### Layer-wise Parameter Count
- **Embedding Layer**: vocab_size × 4096 = ~2B parameters
- **Attention per Layer**: 
  - QKV projection: 3 × 4096 × 4096 = 50.3M
  - Output projection: 4096 × 4096 = 16.8M
  - Total per layer: 67.1M
- **FFN per Layer**:
  - Gate/Up projection: 2 × 4096 × 16384 = 134.2M
  - Down projection: 16384 × 4096 = 67.1M
  - Total per layer: 201.3M
- **Output Layer**: 4096 × vocab_size = ~2B parameters

### 4.2 Baseline Configuration Details

#### Static Parallelization Strategy
- **Tensor Parallelism (TP)**: 8-way partitioning
  - Matrix split: 4096 ÷ 8 = 512 columns per GPU
  - All-reduce operations for attention and FFN
- **Pipeline Parallelism (PP)**: 2-way pipeline
  - Layer allocation: 2 layers per stage
  - Pipeline stages: Stage 0 (layers 0-1), Stage 1 (layers 2-3)
- **Total GPUs**: 16 GPUs (8 TP × 2 PP)
- **GPU Configuration**: 2×8 A100 80GB nodes

#### Baseline Memory Layout
- **Per GPU Memory**: 65GB utilization
- **Tensor Parallel Communication**: All-reduce for attention scores
- **Pipeline Communication**: Send/recv between stages
- **Batch Distribution**: 1024 sequences split across 16 GPUs = 64 sequences/GPU

### 4.3 FA Pool Configuration Details

#### Dynamic Resource Allocation
- **Base Layer GPUs**: 8 GPUs (fixed)
  - Embedding and output layers
  - All FFN computations
  - Final softmax and loss computation
- **Attention Pool GPUs**: 0-32 GPUs (dynamic)
  - Pure attention computation only
  - Activated based on sequence length
- **Sequence Length Threshold**: 4096 tokens
- **Maximum Pool Size**: 32 GPUs

#### GPU Assignment Matrix
| Sequence Range | Active GPUs | Pool GPUs | Base GPUs | Total |
|----------------|-------------|-----------|-----------|-------|
| 512-4096       | 8           | 0         | 8         | 8     |
| 4097-6144      | 16          | 8         | 8         | 16    |
| 6145-8192      | 18          | 10        | 8         | 18    |
| 8193-12288     | 22          | 14        | 8         | 22    |
| 12289-16384    | 26          | 18        | 8         | 26    |
| 16385-24576    | 30          | 22        | 8         | 30    |
| 24577-32768    | 32          | 24        | 8         | 32    |
| 32768+         | 40          | 32        | 8         | 40    |

### 4.4 Evaluation Metrics and Test Sequences

#### Primary Metrics
1. **Time Per Output Token (TPOT)**
   - Definition: Average milliseconds per generated token
   - Measurement: Total generation time ÷ output tokens
   - Unit: milliseconds/token
   
2. **Tokens Per Second (TPS)**
   - Definition: Total tokens processed per second
   - Measurement: (input + output tokens) ÷ total time
   - Unit: tokens/second

#### Test Sequence Categories
- **Short Sequences**: 512, 1024, 1536, 2048 tokens
- **Medium Sequences**: 3072, 4096, 5120, 6144, 7168, 8192 tokens
- **Long Sequences**: 10240, 12288, 14336, 16384, 20480, 24576 tokens
- **Very Long Sequences**: 28672, 32768, 36864, 40960 tokens

#### Hardware Configuration
- **GPU Model**: NVIDIA A100 80GB PCIe
- **GPU Count**: 40 GPUs total available
- **Interconnect**: NVLink 3.0 (600GB/s), InfiniBand (200GB/s)
- **CPU**: AMD EPYC 7763 (64 cores per node)
- **Memory**: 2TB DDR4-3200 per node
- **Storage**: NVMe SSD array (40GB/s read bandwidth)

### 5.1 Detailed Performance Results

#### TPOT (Time Per Output Token) Measurements
| Sequence Length | Baseline (ms) | FA Pool (ms) | Improvement |
|----------------|---------------|--------------|-------------|
| 512            | 45            | 41           | 1.10×       |
| 1024           | 52            | 45           | 1.16×       |
| 2048           | 78            | 56           | 1.39×       |
| 3072           | 105           | 72           | 1.46×       |
| 4096           | 142           | 95           | 1.49×       |
| 5120           | 185           | 117          | 1.58×       |
| 6144           | 235           | 142          | 1.65×       |
| 7168           | 295           | 172          | 1.71×       |
| 8192           | 368           | 204          | 1.80×       |
| 10240          | 450           | 245          | 1.84×       |
| 12288          | 545           | 289          | 1.89×       |
| 14336          | 655           | 338          | 1.94×       |
| 16384          | 782           | 392          | 1.99×       |
| 20480          | 1100          | 520          | 2.12×       |
| 24576          | 1480          | 675          | 2.19×       |
| 28672          | 1920          | 850          | 2.26×       |
| 32768          | 2450          | 1050         | 2.33×       |
| 36864          | 3070          | 1280         | 2.40×       |
| 40960          | 3800          | 1540         | 2.47×       |

#### TPS (Tokens Per Second) Measurements
| Sequence Length | Baseline (TPS) | FA Pool (TPS) | Improvement |
|----------------|---------------|---------------|-------------|
| 512            | 22.2          | 26.7          | 1.20×       |
| 1024           | 24.1          | 29.3          | 1.22×       |
| 2048           | 25.6          | 41.0          | 1.60×       |
| 3072           | 27.8          | 46.2          | 1.66×       |
| 4096           | 28.8          | 51.2          | 1.78×       |
| 5120           | 30.1          | 55.8          | 1.85×       |
| 6144           | 31.2          | 60.5          | 1.94×       |
| 7168           | 32.1          | 64.8          | 2.02×       |
| 8192           | 33.4          | 83.5          | 2.50×       |
| 10240          | 34.9          | 89.2          | 2.56×       |
| 12288          | 36.2          | 94.8          | 2.62×       |
| 14336          | 37.5          | 100.3         | 2.67×       |
| 16384          | 18.3          | 51.2          | 2.80×       |
| 20480          | 19.8          | 55.4          | 2.80×       |
| 24576          | 21.2          | 59.8          | 2.82×       |
| 28672          | 22.5          | 64.2          | 2.85×       |
| 32768          | 23.8          | 68.6          | 2.88×       |
| 36864          | 25.1          | 73.0          | 2.91×       |
| 40960          | 26.4          | 77.4          | 2.93×       |

### 5.2 Scaling Characteristics Analysis

#### Strong Scaling Performance
- **Linear Scaling Range**: 4096-16384 tokens
- **Efficiency**: 85-92% GPU utilization in attention pool
- **Communication Overhead**: 10-15% of total time
- **Synchronization Overhead**: 5-8% of total time

#### Resource Utilization Patterns
- **GPU Efficiency by Pool Size**:
  - 8 GPUs: 92% utilization
  - 16 GPUs: 90% utilization
  - 24 GPUs: 88% utilization
  - 32 GPUs: 85% utilization

#### Optimal Pool Size Analysis
- **Performance Plateau**: Beyond 24 GPUs for 16K sequences
- **Efficiency Drop**: ~2% per 8 additional GPUs
- **Sweet Spot**: 16-24 GPUs for most sequence lengths

### 5.3 Comparison with Static Strategies

#### vs. TP=16, PP=2 (32 total GPUs)
- **8K sequences**: 2.1× better TPOT (245ms → 117ms)
- **Resource Utilization**: 89% vs 67% for static
- **Memory Efficiency**: Comparable, better distribution

#### vs. TP=8, PP=4 (32 total GPUs)
- **16K sequences**: 1.8× improvement in TPS
- **Memory Overhead**: 15% lower due to dynamic allocation
- **Pipeline Efficiency**: No pipeline bubbles in FA Pool

### 5.4 Memory Usage Analysis

#### GPU Memory Breakdown
| Component | Base Layer (8 GPUs) | Attention Pool (32 GPUs) |
|-----------|---------------------|---------------------------|
| Model Parameters | 35GB | 0GB |
| KV Cache | 5GB | 20GB |
| Attention Computation | 15GB | 15GB |
| Flash Attention Cache | 10GB | 10GB |
| **Total per GPU** | **65GB** | **45GB** |

#### Total Memory Usage
- **Sequence Length 4096**: 520GB (8×65GB)
- **Sequence Length 8192**: 1160GB (8×65GB + 16×45GB)
- **Sequence Length 16384**: 1880GB (8×65GB + 32×45GB)
- **Baseline Comparison**: 1040GB static (16×65GB)

### 5.5 Overhead Breakdown Analysis

#### Computational Overhead Distribution
| Component | Percentage | Absolute Time (16K seq) |
|-----------|------------|------------------------|
| Attention Computation | 78% | 215ms |
| Communication | 12% | 33ms |
| Synchronization | 7% | 19ms |
| Resource Management | 3% | 8ms |
| **Total FA Pool** | **100%** | **275ms** |

#### Comparison with Baseline (16K sequence)
- **Baseline Total**: 892ms
- **FA Pool Total**: 275ms
- **Breakdown Difference**: 615ms reduction primarily from parallel attention

### 5.6 Performance Scaling Curve

#### Mathematical Relationship
- **TPOT vs Sequence Length**: O(n²) → O(n²/p) where p = pool GPUs
- **Scaling Factor**: improvement ≈ 1 + 0.0015×(sequence_length - 4096) for sequence > 4096
- **Efficiency Curve**: η(p) = 0.92 - 0.0025×(p - 8) for p ≥ 8