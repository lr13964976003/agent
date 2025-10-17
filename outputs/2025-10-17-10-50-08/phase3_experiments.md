# Phase 3: Experiments Extraction - MA Separation

## 4. Experimental Setup

### 4.1 Model Configuration
**Model Architecture:**
- Number of layers: 4
- Hidden dimension: 4096
- Attention heads: 32
- MoE experts per layer: 16
- Expert hidden dimension: 16384
- Top-K routing: K=2
- Activation function: GELU
- Sequence length: 2048 tokens

**MoE Configuration:**
- Expert capacity factor: 1.0
- Load balancing loss coefficient: 0.01
- Router z-loss coefficient: 0.001
- Expert dropout: 0.1
- Expert type: Feed-forward network with SwiGLU activation

### 4.2 Hardware Configuration
**GPU Setup:**
- Total GPUs: 16 × NVIDIA A100 80GB
- GPU memory per device: 80GB HBM2e
- Interconnect: NVLink 3.0 (600 GB/s) and InfiniBand HDR (200 Gb/s)
- System architecture: 4 nodes × 4 GPUs per node
- CPU: AMD EPYC 7763 64-Core per node
- System memory: 1TB DDR4 per node

### 4.3 Baseline Configuration
**Baseline: Hybrid TP+PP (TP=8, PP=2)**
- Combined tensor and pipeline parallelism
- 8-way tensor parallelism within each pipeline stage
- Same layer distribution as PP=2

### 4.5 Dataset and Training Configuration
**Dataset:**
- Training data: C4 (Colossal Clean Crawled Corpus)
- Validation data: 10% held-out from C4
- Sequence length: 2048 tokens
- Vocabulary size: 50,265 (GPT-2 tokenizer)

**Training Configuration:**
- Batch size: 1024 sequences (2M tokens)
- Learning rate: 1e-4 with cosine decay
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Weight decay: 0.1
- Gradient clipping: 1.0
- Training steps: 50,000
- Warmup steps: 5,000

### 4.6 Evaluation Metrics
**Performance Metrics:**
- Time per Output Token (TPOT)
- Tokens per Second (TPS)
- Throughput
- GPU Utilization
- Memory Efficiency

**Efficiency Metrics:**
- Communication Overhead
- Load Balance
- Scalability
- Energy Efficiency

**Model Quality Metrics:**
- Perplexity
- Convergence Speed
- Expert Utilization
- Load Balancing Loss

### 4.7 Implementation Details
**Software Stack:**
- Deep learning framework: PyTorch 2.0 with CUDA 11.8
- Distributed computing: NCCL 2.15 for GPU communication
- Profiling tools: Nsight Systems and Nsight Compute
- Memory management: Custom CUDA kernels for optimized operations

## 5. Experimental Results and Analysis

### 5.1 Performance Metrics Comparison
| Metric | TP=8 | PP=2 | TP=8, PP=2 | MA Separation | Improvement |
|--------|------|------|------------|---------------|-------------|
| **TPOT (ms/token)** | 2.84 | 3.12 | 2.76 | 1.82 | **34.2% reduction** |
| **TPS (tokens/s)** | 8,450 | 7,692 | 8,696 | 13,289 | **52.8% increase** |
| **Throughput (tokens/s)** | 135,200 | 123,072 | 139,136 | 212,624 | **52.8% increase** |
| **GPU Utilization (%)** | 68.4 | 62.1 | 71.2 | 89.7 | **25.9% increase** |
| **Memory Efficiency (%)** | 72.3 | 69.8 | 74.1 | 85.4 | **15.2% increase** |

### 5.2 Scalability Analysis
- Linear Scalability: MA Separation maintains near-linear scalability up to 16 GPUs
- Scaling Efficiency: 87% efficiency at 16 GPUs
- Break-even Point: MA Separation outperforms baselines starting from 8 GPUs
- Speedup_16GPUs = 13,289 / 8,696 = 1.528 (52.8% improvement)

### 5.3 Communication Overhead Analysis
| Communication Type | TP=8 | PP=2 | TP=8, PP=2 | MA Separation |
|-------------------|------|------|------------|---------------|
| **Attention All-Reduce (%)** | 12.3 | 0 | 11.8 | 8.4 |
| **MoE All-to-All (%)** | 0 | 0 | 0 | 6.2 |
| **Gradient Synchronization (%)** | 3.2 | 2.8 | 3.1 | 2.9 |
| **Parameter Broadcast (%)** | 1.1 | 1.2 | 1.1 | 1.3 |
| **Total Communication (%)** | 16.6 | 4.0 | 16.0 | 18.8 |

### 5.4 Memory Utilization Analysis
| Component | TP=8 | PP=2 | TP=8, PP=2 | MA Separation |
|-----------|------|------|------------|---------------|
| **Model Parameters (GB)** | 18.2 | 36.4 | 18.2 | 23.1 |
| **Activations (GB)** | 22.4 | 11.2 | 22.4 | 18.7 |
| **Gradients (GB)** | 18.2 | 36.4 | 18.2 | 23.1 |
| **Optimizer States (GB)** | 36.4 | 72.8 | 36.4 | 46.2 |
| **Communication Buffers (GB)** | 8.3 | 4.1 | 8.3 | 12.6 |
| **Total Memory Usage (GB)** | 103.5 | 160.9 | 103.5 | 123.7 |
| **Memory Efficiency (%)** | 72.3 | 69.8 | 74.1 | 85.4 |

### 5.7 Inference Performance Analysis
| Sequence Length | TP=8 TPOT | MA Separation TPOT | Improvement |
|-----------------|-----------|-------------------|-------------|
| **512** | 1.23 ms | 0.89 ms | 27.6% |
| **1024** | 1.84 ms | 1.21 ms | 34.2% |
| **2048** | 2.84 ms | 1.82 ms | 35.9% |
| **4096** | 5.67 ms | 3.41 ms | 39.9% |

### 5.8 Energy Efficiency Analysis
- Total Energy per Token: 0.82 mJ (MA Separation) vs 1.24 mJ (baseline)
- Energy Efficiency: 33.9% improvement
- PUE: 1.08 vs 1.12 for baseline
- Carbon Footprint: 34.2% reduction in CO₂ emissions per token