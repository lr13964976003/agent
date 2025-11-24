# Phase 1: Key Points of the Paper

## Core Problem
- Large neural network models require efficient deployment across multiple accelerator cards
- Limited on-chip memory (SRAM/L2 cache) creates bottlenecks for memory access
- Off-chip memory access introduces latency and bandwidth limitations

## Key Innovation
- **Layer-wise partitioning strategy**: Distribute model layers across multiple processing units
- **Cache-aware allocation**: Ensure each partition fits entirely within SRAM/L2 cache of a single device
- **Systematic estimation**: Analytical method to evaluate partition sizes and allocate to hardware resources

## Technical Approach
1. **Problem Formulation**: Partition n layers into k groups where each group fits within cache capacity C
2. **Memory Estimation**: Calculate footprint including weights, activations, and temporary buffers
3. **Partitioning Algorithms**: 
   - Greedy layer aggregation (simple, efficient)
   - Dynamic programming for balanced partitions (optional)
4. **Deployment Strategy**: Load entire partition into cache, execute sequentially, transfer outputs between cards

## Performance Gains
- **Dense 16-layer model**: 20% increase in TPS (12,800 → 15,360 tokens/sec)
- **Latency reduction**: 17% reduction in TPOT (0.078 → 0.065 ms)
- **Hardware**: 16 NVIDIA H100 GPUs, BF16 precision, 30B parameter model

## Baseline Comparison
- Standard tensor parallelism (TP=8) + pipeline parallelism (PP=2)
- Total 16 GPUs utilized in both methods
- Proposed method outperforms by considering on-chip memory constraints explicitly

## Key Dimensions
- Model: 16-layer dense network
- Precision: BF16 (2 bytes per parameter)
- Batch size: 128
- Sequence length: 10,000
- Heads: 32
- Head dimension: 128
- MLP hidden size: 16,384
- Total model size: 30B parameters