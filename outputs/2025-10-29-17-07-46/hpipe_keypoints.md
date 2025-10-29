# HPipe Key Points Summary

## Problem Statement
- **Target Users**: Micro-enterprises and individual developers who need long-context LLM analysis
- **Challenge**: Deploying LLMs on heterogeneous commodity devices with unreliable interconnections
- **Limitations**: Existing parallel techniques cannot fully perform in limited environments

## Key Innovations

### 1. HPipe Framework
- **Type**: Pipeline inference framework for content comprehension with private LLMs
- **Core Innovation**: Pipeline parallelism on token dimension instead of batch dimension
- **Target**: Migrate LLMs from high-performance clusters to heterogeneous commodity devices

### 2. Balanced Distribution Strategy
- **Approach**: Distribute LLMs based on computing capabilities and transmission conditions
- **Granularity**: Uses layer-level partition instead of transformer block-level for better balance
- **Optimization**: Dynamic programming algorithm to minimize workload imbalance

### 3. Dynamic Sequence Slicing
- **Purpose**: Handle extended context by slicing sequences into optimal segments
- **Method**: Dynamic programming algorithm to find optimal slicing granularity
- **Key Insight**: Longer slices at beginning, shorter slices toward end due to increasing computation with token position

## Performance Results
- **Speedup**: Up to 2.28× improvement in both latency and throughput
- **Energy**: 68.2% reduction in energy consumption
- **Models Tested**: LLaMA-7B and GPT3-2B
- **Environment**: Heterogeneous cluster (4×P100 + 2×RTX3090)

## Technical Foundations
- **Parallel Strategy**: Pipeline parallelism on token dimension
- **Communication**: Lightweight transmission only for intermediate results
- **Memory**: Caches K,V values for subsequent token calculations
- **Utilization**: Maximizes device utilization by matching sequence length to hardware capabilities