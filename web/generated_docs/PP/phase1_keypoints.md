# Phase 1: Keypoints Extraction

## Paper Title: Layer-wise Deployment Strategy for Large Neural Networks

## Key Contributions:
1. **Novel layer-wise partitioning strategy** that distributes model layers across multiple accelerator cards
2. **SRAM/L2 cache optimization** - ensures each partition fits entirely within on-chip memory
3. **Memory footprint estimation method** for accurate partition sizing
4. **Greedy and dynamic programming algorithms** for optimal layer grouping
5. **20% performance improvement** over baseline tensor+pipeline parallelism

## Core Problem Addressed:
- Large models require off-chip memory access, causing latency and bandwidth bottlenecks
- Traditional parallelism methods don't explicitly consider on-chip memory constraints
- Need for deployment strategies that maximize fast memory utilization

## Technical Innovation:
- **Memory constraint formulation**: Each partition P_i must satisfy S(P_i) ≤ C (cache capacity)
- **Comprehensive memory estimation**: Includes weights + activations + temporary buffers
- **Contiguous layer assignment**: Preserves execution order while optimizing memory locality
- **Inter-card communication minimization**: Only transfer intermediate outputs between partitions

## Experimental Validation:
- Hardware: 16 NVIDIA H100 GPUs
- Model: 16-layer dense network (FP16, batch=1024, seq_len=10000)
- Baseline: TP=8, PP=2 configuration
- Results: 15,360 TPS vs 12,800 TPS (20% improvement), 0.065ms vs 0.078ms TPOT (17% reduction)

## Deployment Advantages:
- Reduced memory access latency through SRAM/L2 cache utilization
- Improved throughput via parallel execution on multiple cards
- Scalability across varying model sizes and hardware configurations
- Minimized inter-card communication overhead

## Critical Technical Details:
- Memory footprint formula: size(l_j) = weight_size + activation_size + buffer_size
- Greedy algorithm accumulates layers until cache capacity is reached
- Dynamic programming option for balanced partition sizes
- Edge case handling for oversized layers (quantization/pruning)
- Batch size tuning for activation memory optimization