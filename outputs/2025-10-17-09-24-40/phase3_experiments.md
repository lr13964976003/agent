# Phase 3: Experiments Extraction - MA Separation

## 4. Experimental Setup (Complete)

### 4.1 Model Configuration
Our experimental evaluation employs a 4-layer MoE transformer model with the following specifications:

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

**Network Topology:**
- Intra-node communication: NVLink mesh topology
- Inter-node communication: Fat-tree InfiniBand topology
- Network latency: < 1μs intra-node, < 5μs inter-node

### 4.3 Baseline Configuration
We compare MA Separation against traditional parallel strategies:

**Baseline: Hybrid TP+PP (TP=8, PP=2)**
- Combined tensor and pipeline parallelism
- 8-way tensor parallelism within each pipeline stage
- Same layer distribution as PP=2

### 4.4 MA Separation Configuration
**Attention Parallelization:**
- Attention GPUs: 12 (out of 16 total)
- Attention replication factor: 2× for redundancy

**MoE Parallelization:**
- MoE GPUs: 4 (out of 16 total)
- Experts per GPU: 4 (16 experts total)
- Expert replication: None (experts are unique per GPU)
- Load balancing: Dynamic based on expert utilization

**Synchronization Settings:**
- Time prediction model: Neural network with 3 hidden layers
- Synchronization interval: Every 100 iterations
- Load balancing threshold: 5% execution time difference
- Communication compression: 8-bit quantization for gradients

### 4.5 Dataset and Training Configuration
**Dataset:**
- Training data: C4 (Colossal Clean Crawled Corpus) [17]
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
- **Time per Output Token (TPOT)**: Average time to generate one output token during inference
- **Tokens per Second (TPS)**: Number of tokens processed per second during training/inference
- **Throughput**: Total tokens processed per unit time across all GPUs
- **GPU Utilization**: Average GPU compute utilization percentage
- **Memory Efficiency**: Memory bandwidth utilization percentage

**Efficiency Metrics:**
- **Communication Overhead**: Time spent in inter-GPU communication
- **Load Balance**: Standard deviation of execution times across GPUs
- **Scalability**: Performance improvement with increasing GPU count
- **Energy Efficiency**: Performance per watt of power consumption

**Model Quality Metrics:**
- **Perplexity**: Language modeling perplexity on validation set
- **Convergence Speed**: Training loss reduction rate
- **Expert Utilization**: Percentage of experts used during training
- **Load Balancing Loss**: MoE routing balance metric

## 5. Experimental Results and Analysis (Key Metrics)

### 5.1 Performance Metrics Comparison
| Metric | TP=8, PP=2 | MA Separation | Improvement |
|--------|------------|---------------|-------------|
| **TPOT (ms/token)** | 2.76 | 1.82 | **34.2% reduction** |
| **TPS (tokens/s)** | 8,696 | 13,289 | **52.8% increase** |
| **Throughput (tokens/s)** | 139,136 | 212,624 | **52.8% increase** |
| **GPU Utilization (%)** | 71.2 | 89.7 | **25.9% increase** |
| **Memory Efficiency (%)** | 74.1 | 85.4 | **15.2% increase** |

### 5.2 Memory Utilization Analysis
| Component | TP=8, PP=2 | MA Separation |
|-----------|------------|---------------|
| **Model Parameters (GB/GPU)** | 18.2 | 23.1 |
| **Activations (GB/GPU)** | 22.4 | 18.7 |
| **Total Memory Usage (GB/GPU)** | 103.5 | 123.7 |
| **Memory Efficiency (%)** | 74.1 | 85.4 |

### 5.3 Communication Overhead Analysis
| Communication Type | TP=8, PP=2 | MA Separation |
|-------------------|------------|---------------|
| **Attention All-Reduce (%)** | 11.8 | 8.4 |
| **MoE All-to-All (%)** | 0 | 6.2 |
| **Total Communication (%)** | 16.0 | 18.8 |

### 5.4 Scalability Analysis
- **Linear Scalability**: Maintained up to 16 GPUs
- **Scaling Efficiency**: 87% at 16 GPUs
- **Break-even Point**: Outperforms baselines starting from 8 GPUs

## Critical Dimensions for Deployment
- **Sequence Length**: 2048 tokens
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Expert Count**: 16 per layer
- **Expert Hidden Dimension**: 16384
- **Batch Size**: 1024 sequences (2M tokens)
- **GPU Count**: 16 total (12 attention + 4 MoE)
- **Expert Distribution**: 4 experts per MoE GPU
- **Attention Head Distribution**: 32 heads across 12 GPUs