# HPipe Experiments

## Experimental Setup

### Hardware Configuration
- **Cluster**: Two host machines
  - Machine 1: 4× Pascal P100 GPUs
  - Machine 2: 2× RTX 3090 GPUs
- **Network**: 1000 Mbps wired inter-host communication
- **Intra-host**: PCIe communication

### Models and Parameters
- **GPT3-2B**: Batch size = 12, Sequence length = 2048 tokens
- **LLaMA-7B**: Batch size = 6, Sequence length = 2048 tokens

### Baseline Methods
1. **Base**: Uniform LLM distribution across GPUs, sequential inference
2. **GPipe**: Even LLM distribution with micro-batch pipelining
3. **GP-B**: GPipe with HPipe's balanced workload distribution
4. **Megatron-LM**: Tensor parallelism combined with GPipe
5. **Terapipe**: Even distribution with token-level pipeline
6. **TP-T**: Tensor parallelism combined with Terapipe

## Performance Results

### Latency and Throughput

#### LLaMA-7B Results
- **Base**: 20.3s latency, 0.56k tokens/s throughput
- **HPipe**: 2.24s latency, 5.03k tokens/s throughput
- **Speedup**: 9.06× latency reduction, 8.98× throughput improvement
- **vs GP-B**: 33.1% latency reduction from balanced distribution

#### GPT3-2B Results
- **Base**: Out of memory on P100 devices
- **HPipe**: Achieved successful inference with significant speedup
- **Overall**: 2.28× maximum speedup in both latency and throughput

### Energy Consumption
- **HPipe**: Lowest dynamic energy consumption among all methods
- **Improvement**: 68.2% energy reduction compared to baseline methods
- **Reason**: High resource utilization through optimal sequence length selection

### Memory Footprint Analysis

#### LLaMA-7B Memory Usage (MB)
| Method | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 |
|--------|-----|-----|-----|-----|-----|-----|
| Base   | 11479 | 11479 | 11019 | 11019 | 11461 | 11461 |
| GP     | 7031 | 7031 | 6593 | 6593 | 5509 | 5509 |
| GP-B   | 2897 | 3135 | 3655 | 3030 | 1969 | 11073 |
| MG     | 5851 | 5851 | 5493 | 5493 | 5943 | 5943 |
| TP     | 5459 | 5459 | 4505 | 4505 | 4957 | 4957 |
| TP-P   | 4869 | 4869 | 4583 | 4583 | 5013 | 5013 |
| **HPipe** | **1873** | **2977** | **3143** | **1991** | **8713** | **10087** |

#### GPT3-2B Memory Usage (MB)
| Method | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 |
|--------|-----|-----|-----|-----|-----|-----|
| Base   | OOM | OOM | OOM | OOM | - | - |
| GP     | 7031 | 7031 | 6593 | 6593 | 5509 | 5509 |
| GP-B   | 3665 | 3505 | 3495 | 3177 | 8525 | 8627 |
| MG     | 4695 | 4695 | 4595 | 4595 | 5057 | 5043 |
| TP     | 6601 | 6601 | 6629 | 6629 | 6681 | 6681 |
| TP-P   | 4952 | 4952 | 5032 | 5032 | 5433 | 5437 |
| **HPipe** | **4693** | **4651** | **3153** | **295** | **3975** | **7985** |

### Resource Utilization Analysis

#### Execution Visualization
- **Uniform Distribution**: RTX 3090 shows tiny execution time vs P100, causing significant idle time
- **HPipe Schedule**: Balanced execution time across devices through proper workload distribution
- **Pipeline Efficiency**: Minimized bubbles through optimal sequence slicing

#### Sequence Length Impact
- **FLOPs Utilization**: Initially increases then decreases with sequence length due to I/O bottlenecks
- **Tile Quantization**: Matrix dimensions must be divisible by GPU tile size for optimal utilization
- **Optimal Length**: Dynamic programming finds length that maximizes device utilization

## Ablation Studies

### Dynamic Sequence Schedule Validation
- **Test Setup**: 2048 token sequence with uniform slicing (1-128 slices)
- **Results**: HPipe with dynamic schedule outperforms best uniform configuration
- **Finding**: Fine granularity causes underutilization, coarse granularity creates pipeline bubbles

### Sensitivity Analysis
- **Network Bandwidth**: Performance scales with inter-host bandwidth
- **Device Heterogeneity**: Greater performance gains with more diverse hardware
- **Sequence Length**: Optimal length selection becomes critical for longer contexts

## Key Findings

### Comparative Analysis
1. **Tensor Parallelism Limitations**: Not suitable for constrained environments due to high communication overhead
2. **Token-level Pipeline**: Superior to batch-level pipeline for memory-constrained scenarios
3. **Balanced Distribution**: 51-56% latency reduction from workload balancing alone
4. **Combined Optimization**: Token-level pipeline adds 33.1-39.3% latency reduction on top of balanced distribution

### Practical Implications
- **Deployment**: Successfully enables LLM deployment on commodity hardware
- **Cost**: Significant cost reduction compared to high-performance clusters
- **Scalability**: Method scales with device heterogeneity and network conditions