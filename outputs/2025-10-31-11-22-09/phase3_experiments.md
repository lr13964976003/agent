# HPipe: Experimental Analysis

## 4.1 Experimental Setup

### Hardware Configuration
```
Cluster Setup:
- Host 1: 4 × NVIDIA P100 GPUs
- Host 2: 2 × RTX3090 GPUs
- Inter-host Network: 1000 Mbps wired
- Intra-host Communication: PCIe

Heterogeneous Cluster Design:
- Purpose: Mimic commodity hardware setup
- GPU Types: Pascal (P100) + Ampere (RTX3090) architecture mix
- Communication: PCIe vs Network bandwidth disparity
```

### Model Specifications
```
Models Tested:
- LLaMA-7B: 7 billion parameters, 32 transformer layers
- GPT3-2B: 2.7 billion parameters, 24 transformer layers

Input Configuration:
- Sequence Length: 2048 tokens (long context simulation)
- Batch Sizes: GPT3-2B = 12, LLaMA-7B = 6
- Task: Content analysis for long sequences
```

### Baseline Methods
```
1. Base: Uniform LLM distribution, sequential inference
2. GPipe (GP): Even LLM distribution, micro-batch pipeline
3. GP-B: GPipe with HPipe's balanced distribution
4. Megatron-LM (MG): Tensor parallelism + pipeline parallelism
5. Terapipe: Token-level pipeline with even distribution
6. TP-T: Tensor parallelism + Terapipe combination
```

## 4.2 Performance Results

### 4.2.1 Latency and Throughput Analysis

#### LLaMA-7B Results:
```
Performance Metrics (2048 tokens, batch=6):
- Base: 20.3s latency, 0.56k tokens/s throughput
- HPipe: 2.24s latency, 5.03k tokens/s throughput
- Speedup: 9.06× latency reduction, 9× throughput improvement
- GPipe vs HPipe: 51-56% latency reduction with balanced distribution
- Terapipe vs HPipe: 33.1-39.3% latency reduction with sequence scheduling
```

#### GPT3-2B Results:
```
Performance Metrics (2048 tokens, batch=12):
- Similar trend: 2.28× speedup observed
- Balanced distribution: 2.06-2.28× throughput enhancement
- Sequence scheduling: Significant pipeline efficiency improvement
```

#### Key Findings:
- **Tensor Parallelism Limitation**: MG shows transmission overhead for large synchronized data
- **Token Pipeline Advantage**: Fine-grained execution granularity improves parallelism
- **Device Utilization**: RTX3090 shows much faster execution (tiny execution time vs P100)

### 4.2.2 Energy Consumption Analysis

#### Energy Efficiency Results:
```
LLaMA-7B Energy Consumption:
- HPipe achieves lowest dynamic energy consumption
- 68.2% reduction compared to other methods
- Optimization considers device power characteristics

Energy Optimization Strategy:
- Joint optimization of computation vs communication trade-off
- High resource utilization → reduced energy waste
- Sequence length selection for maximum cluster utilization
```

#### GPT3-2B Energy Consumption:
- Similar trends observed across different models
- Energy-aware workload distribution based on device capabilities

### 4.2.3 Memory Footprint Analysis

#### Memory Utilization Results:
```
Memory Requirements (MB):

LLaMA-7B:
- Base: 11479 MB per device (uniform distribution)
- HPipe: 1873-9977 MB range (heterogeneous distribution)
- Key observation: P@4 and R@1 have lower memory (fewer layers allocated)

GPT3-2B:
- Base: OOM (Out of Memory) on some devices
- HPipe: 4693-8555 MB range
- Successful deployment achieved through layer-level distribution
```

#### Memory Distribution Strategy:
- **Tensor Parallelism**: Reduces memory pressure by distributing weights
- **Balanced Distribution**: Allocates layers based on device memory capacity
- **Communication Overhead Trade-off**: Devices with higher communication overhead get fewer layers

## 4.3 Resource Utilization Visualization

### Inference Timeline Analysis

#### Uniform Distribution Results:
```
Device Performance Characteristics:
- RTX3090: Tiny execution time per subsequence
- P100: Much longer execution time
- Problem: RTX3090 falls into waiting state → resource underutilization
- Uniform slicing: Longer execution for subsequent subsequences → pipeline bottleneck
```

#### HPipe Optimized Distribution:
```
Optimization Results:
- Computationally powerful devices handle heavier tasks
- Approximate execution time per subsequence across devices
- Increasingly shorter subsequences balance pipeline stages
- Minimal idle waiting time between stages
```

### Ablation Studies

#### Sequence Schedule Impact:
```
Uniform vs Dynamic Slicing (2048 tokens):
- Uniform slicing: 1-128 slices tested
- Dynamic scheduling: Outperforms best uniform configuration
- Fine-grained: GPU underutilization
- Coarse-grained: Large pipeline bubbles
- Optimal granularity: Achieved through dynamic programming
```

#### Model Scaling Analysis:
```
Performance Across Models:
- LLaMA-7B: 32 layers, 7B parameters
- GPT3-2B: 24 layers, 2.7B parameters
- Consistent performance improvement across model sizes
- Scalability demonstrated through heterogeneous deployment
```

## Experimental Validation Summary

### Key Achievements:
1. **Latency Reduction**: Up to 9.06× improvement over baseline
2. **Throughput Enhancement**: Up to 9× tokens per second improvement
3. **Energy Efficiency**: 68.2% reduction in energy consumption
4. **Memory Optimization**: Successful deployment avoiding OOM conditions
5. **Scalability**: Validated across different model sizes and architectures

### Practical Deployment Insights:
- **Heterogeneous Hardware**: Successfully handles P100 + RTX3090 mix
- **Network Constraints**: Effective under 1000Mbps inter-host bandwidth
- **Long Context Handling**: Optimal for 2048+ token sequences
- **Batch Processing**: Validated for batch sizes 6-12 depending on model size