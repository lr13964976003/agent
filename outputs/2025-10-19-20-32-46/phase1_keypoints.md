# Phase 1: Key Points Extraction

## Core Problem
- Challenge: Large neural networks exceed on-chip memory (SRAM/L2 cache) capacity
- Impact: Frequent off-chip memory access causes latency and bandwidth bottlenecks
- Objective: Fit model partitions entirely within fast on-chip memory

## Proposed Solution
- **Layer-wise deployment strategy**: Partition model layers across multiple accelerator cards
- **Key constraint**: Each partition must fit within target card's SRAM/L2 cache capacity C
- **Method**: Systematic partitioning algorithm considering memory footprint of weights, activations, and buffers

## Memory Footprint Components
1. **Weights**: Parameter tensors (datatype size × parameter count)
2. **Activations**: Intermediate outputs (batch size × sequence length × hidden dimensions)
3. **Temporary Buffers**: Operator workspace memory

## Partitioning Algorithms
1. **Greedy Layer Aggregation**:
   - Iteratively add layers until capacity C is reached
   - Simple, efficient, guarantees cache fit
2. **Dynamic Programming** (optional):
   - More balanced partitions
   - Minimizes maximum partition size

## Deployment Process
1. Partition model into k groups P₁, P₂, ..., Pₖ
2. Map each partition to separate accelerator card
3. Load weights, pre-allocate activation/buffer memory in cache
4. Execute sequentially on assigned card
5. Transfer intermediate outputs only between partitions

## Experimental Results
- **Platform**: 16 NVIDIA H100 GPUs
- **Test Model**: 16-layer dense network
- **Configuration**: FP16, batch=1024, seq_len=10000, 16 heads×512 dim, MLP hidden=32768
- **Performance**: 
  - Baseline (TP=8, PP=2): 12,800 TPS, 0.078ms TPOT
  - Proposed: 15,360 TPS (+20%), 0.065ms TPOT (-17%)

## Key Advantages
- Reduced memory access latency
- Improved throughput via parallel execution
- Scalable to varying model sizes and hardware configurations
- Explicit consideration of cache constraints

## Critical Parameters for Deployment
- SRAM/L2 cache capacity C per device
- Model layer count n
- Layer memory footprints: weights, activations, buffers
- Batch size impact on activation memory
- Device count k for partitioning