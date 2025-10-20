# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16× NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only

### Model Architecture
- **Model Type**: Dense Transformer
- **Layers**: 4 layers, standard feed-forward architecture
- **Precision**: FP16
- **Batch Size**: 1024 (fixed)
- **Sequence Length**: 10000 tokens (fixed)
- **Attention Heads**: 16 (fixed)
- **Head Dimension**: 512 (fixed)
- **MLP Hidden Size**: 32768 (fixed)

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **No sequence parallelism or ring-based attention**

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Optimization goal: Higher is better

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token in milliseconds
   - Optimization goal: Lower is better

## Results

### Performance Comparison Table
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

### Performance Improvements
- **Dense Model Results**:
  - TPS improvement: **+20.8%** (1.45M vs 1.20M)
  - TPOT reduction: **-17.6%** (0.70ms vs 0.85ms)

## Analysis

### Key Performance Factors
1. **Ring-based Communication Pattern**
   - Avoids peak bandwidth demands of all-to-all exchanges
   - Sequential communication reduces synchronization overhead

2. **Memory Savings from Sequence Parallelism**
   - Reduced activation footprint
   - Improved kernel scheduling efficiency

### Scalability Insights
- Benefits grow with sequence length (L) and number of devices (P)
- Particularly effective for sequences longer than 16k tokens
- Communication-computation overlap achieved through ring topology

### Limitations Noted
- Experiments conducted in inference-only setting
- Results specific to 16×H100 configuration
- Fixed sequence length (10k tokens) and batch size (1024) in tests