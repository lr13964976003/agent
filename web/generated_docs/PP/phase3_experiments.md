# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform:** 16 NVIDIA H100 GPUs
- **Total GPUs:** 16
- **Precision:** FP16 for all computations

### Model Architecture
- **Dense Model:** 16-layer fully connected network
- **Fixed Parameters:**
  - Batch size: 1024
  - Sequence length: 10000
  - Number of heads: 16
  - Dimension per head: 512
  - MLP hidden size: 32768
  - Total hidden dimension: 16 × 512 = 8192

### Baseline Configuration
- **Method:** Standard tensor parallelism (TP) + pipeline parallelism (PP)
- **Configuration:** TP=8, PP=2 (fully utilizes 16 GPUs: 8 × 2 = 16)
- **Mapping:** 8-way tensor parallelism within each pipeline stage, 2 pipeline stages across 16 GPUs

### Proposed Method Configuration
- **Method:** Layer-wise deployment with cache-aware partitioning
- **Partitioning:** 16 layers distributed across 16 GPUs
- **Constraint:** Each partition fits within SRAM/L2 cache of single GPU
- **Strategy:** Greedy layer aggregation algorithm

## Performance Metrics

### Results Table
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Analysis
- **Throughput improvement:** 20% increase in TPS (15,360 vs 12,800)
- **Latency reduction:** 17% reduction in TPOT (0.065ms vs 0.078ms)
- **Efficiency gain:** Better cache utilization compared to baseline

### Baseline vs Proposed Method
- **Baseline characteristics:** TP=8 splits layers across 8 devices for tensor parallelism, PP=2 creates 2 pipeline stages
- **Proposed characteristics:** Each GPU gets contiguous layers that fit in cache, minimal inter-GPU communication during layer execution
- **Key difference:** Proposed method explicitly considers on-chip memory constraints while baseline focuses on parallel computation

## Memory Footprint Calculation for Dense Model
Given the architecture parameters:
- **Layer structure:** Each layer includes attention (16 heads × 512 dims) + MLP (32768 hidden)
- **Weight sizes:** Based on hidden_size=8192, ffn_hidden_size=32768
- **Activation sizes:** Calculated for batch_size=1024, sequence_length=10000
- **Cache constraint:** Each partition must fit within single H100's SRAM/L2 cache

## Experimental Validation Points
1. **Cache-fit verification:** Ensure each layer group fits within target cache capacity
2. **Scalability test:** 16-way distribution across available GPUs
3. **Performance comparison:** Direct comparison with state-of-practice TP+PP baseline
4. **Reproducibility:** Fixed experimental parameters ensure consistent results