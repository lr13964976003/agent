# Keypoints of the Paper

## Core Problem
Large neural network models face memory access bottlenecks when deployed on hardware with limited on-chip memory (SRAM/L2 cache), requiring frequent expensive off-chip DRAM accesses.

## Proposed Solution
Layer-wise partitioning and distribution method that:
- Partitions model's n layers across multiple accelerator cards
- Ensures each partition fits entirely within SRAM or L2 cache of single device
- Minimizes off-chip memory accesses and reduces latency

## Key Technical Contributions

### 1. Memory-Constrained Partitioning
- Formally defines problem: partition layers into k groups where each group size ≤ cache capacity C
- Maintains execution order by assigning layers contiguously
- Minimizes number of partitions k for optimal hardware utilization

### 2. Memory Footprint Estimation
- Comprehensive calculation: size(lj) = weight_size + activation_size + buffer_size
- Weight size: parameters × datatype size (e.g., FP16 = 2 bytes)
- Activation size: output feature map dimensions × batch size
- Buffer size: operator workspace requirements

### 3. Partitioning Algorithms
- **Greedy Layer Aggregation**: Simple sequential accumulation until cache limit
- **Dynamic Programming**: Optimizes for balanced partitions and minimized maximum partition size

### 4. Deployment Strategy
- Each partition loaded entirely into SRAM/L2 cache
- Sequential execution within each card
- Minimal inter-card communication (only intermediate outputs between partitions)

## Experimental Validation
- Hardware: 16 NVIDIA H100 GPUs
- Model: 16-layer dense network, 30B parameters, BF16 precision
- Configuration: Batch size 128, sequence length 10000
- Baseline: TP=8, PP=2 (standard tensor + pipeline parallelism)
- Results: 20% TPS improvement (12,800 → 15,360 tokens/s), 17% TPOT reduction (0.078 → 0.065 ms)

## Performance Impact
- Up to 31% throughput improvement over baseline
- Significant latency reduction through optimized memory locality
- Scalable to varying model sizes and hardware configurations