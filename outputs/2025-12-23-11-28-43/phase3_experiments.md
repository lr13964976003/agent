# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 16-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 64 experts
- **Precision**: FP8
- **Batch size**: 128 sequences per batch
- **Sequence length**: 128 tokens per sequence
- **Token dimension**: 1024
- **Multi-head attention**: 16 heads, 64 dimensions per head
- **MoE hidden size**: 2048

### Hardware Environment
- **GPUs**: H100 GPUs (adequate number)
- **Setting**: Inference-only evaluation
- **Network**: High-performance interconnect (NVLink, InfiniBand, NVSwitch)

## Parallel Deployment Configurations

### Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Tensor Parallelism**: TP=8 (each GPU holds 8 tensor-parallel shards for all layers)
- **Pipeline Parallelism**: PP=2 (tokens flow sequentially through pipeline stages)
- **Expert Placement**: Experts colocated on 16 GPUs, sharing compute resources
- **Processing**: Sequential pipeline with shared expert computation

### Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 16 H100 GPUs
- **Expert Placement**: Each GPU hosts exactly one expert per layer (corrected: 4 experts distributed across layers)
- **Parallel Strategy**: Large EP with topology-aware distribution
- **Routing**: Dynamic token routing to GPU holding corresponding expert
- **Communication**: Asynchronous token batch transfer with minimal idle time

## Performance Results

### Throughput Comparison
| Method | TPS (Tokens/s) | Improvement |
|--------|----------------|-------------|
| Baseline (TP=8, PP=2) | 120,000 | Baseline |
| Proposed Cross-Node EP | 450,000 | 3.75× higher |

### Latency Comparison
| Method | TPOT (ms) | Improvement |
|--------|-----------|-------------|
| Baseline (TP=8, PP=2) | 8.3 | Baseline |
| Proposed Cross-Node EP | 2.2 | 3.8× lower |

## Key Performance Insights

### Baseline Limitations
- **Intra-GPU contention**: Multiple experts share GPU resources
- **Pipeline stalls**: Sequential processing creates bottlenecks
- **Resource underutilization**: GPUs not fully utilized for expert computation

### Proposed Method Advantages
- **Expert-level parallelism**: All 16 experts per layer compute in parallel (corrected: 4 experts per GPU distributed)
- **Maximal GPU utilization**: Dedicated expert computation per GPU
- **Asynchronous operation**: Minimal waiting through overlapping communication

## Experimental Validation

### Scalability Demonstration
- **Large EP regime**: EP ≥ 16 validates the large-scale expert parallelism concept
- **Near-linear scaling**: 16 GPUs demonstrate scalable performance
- **Communication efficiency**: Network overhead mitigated through careful scheduling

### Load Balancing Effectiveness
- **Dynamic routing**: Prevents expert overloading through adaptive gating
- **Token batching**: Reduces network congestion through efficient grouping
- **Balanced workload**: All experts receive approximately equal token distribution

## Performance Metrics Analysis

### Throughput Gains
- **3.75× improvement**: 450,000 vs 120,000 TPS
- **Root cause**: Elimination of intra-GPU expert contention
- **Scalability factor**: Large EP enables concurrent expert processing

### Latency Reduction
- **3.8× improvement**: 2.2ms vs 8.3ms TPOT
- ** Contributing factors**: Asynchronous routing, pipelined processing
- **Communication impact**: Overlapped transfer reduces waiting time

## Experimental Conclusions

### Validation of Hypothesis
- **Large EP effectiveness**: EP ≥ 16 regime demonstrates significant performance gains
- **Single expert per GPU**: Principle validated through improved throughput and latency
- **Cross-node distribution**: Topology-aware placement successfully minimizes bottlenecks

### Practical Implications
- **HPC environments**: Method particularly effective in high-performance computing clusters
- **Resource requirements**: Requires abundant GPU resources (16+ H100s)
- **Network dependence**: Relies on high-bandwidth, low-latency interconnects