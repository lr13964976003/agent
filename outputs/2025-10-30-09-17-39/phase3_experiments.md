# HPipe Experiments - Detailed Extraction

## 4.1 Experimental Setup

### Hardware Configuration
- **Cluster**: Two host machines
  - Machine 1: 4× Pascal P100 GPUs
  - Machine 2: 2× RTX 3090 GPUs
- **Network**: 
  - Inter-host: 1000Mbps wired network
  - Intra-host: PCIe communication
- **Purpose**: Mimic commodity hardware setup for micro-enterprises

### Model Configurations
- **Models**: LLaMA-7B, GPT3-2B
- **Sequence Length**: 2048 tokens (long context simulation)
- **Batch Sizes**: 
  - GPT3-2B: 12
  - LLaMA-7B: 6

## 4.2 Performance Comparison

### 4.2.1 Baseline Methods
1. **Base**: Uniform LLM distribution, sequential inference across cluster
2. **GPipe (GP)**: Even distribution, micro-batch pipeline
3. **GP-B**: GPipe with HPipe workload distribution
4. **Megatron-LM (MG)**: Tensor parallelism + GPipe
5. **Terapipe**: Even distribution, token-dimension pipeline
6. **TP-T**: Tensor parallelism + Terapipe

### 4.2.2 Latency and Throughput Results

#### LLaMA-7B Results
- **HPipe (HP)**: 2.24s latency, 5.03k tokens/s throughput
- **Base**: 20.3s latency, 0.56k tokens/s throughput
- **Speedup**: 9.06× latency reduction, 8.98× throughput increase
- **GP vs HP**: 51-56% latency reduction, 2.06-2.28× throughput enhancement
- **Token pipeline benefit**: 33.1-39.3% additional latency reduction

#### GPT3-2B Results
- Similar performance trends as LLaMA-7B
- Consistent acceleration across different model scales

### 4.2.3 Energy Consumption
- **HPipe**: Lowest dynamic energy consumption
- **Improvement**: 68.2% reduction vs other methods
- **Reason**: High resource utilization + balanced workload distribution
- **Mechanism**: Optimal sequence length selection for maximum cluster utilization

### 4.2.4 Memory Footprint Analysis

#### Device Memory Usage (MB)
| Model | Method | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 |
|-------|--------|-----|-----|-----|-----|-----|-----|
| LLaMA-7B | Base | 11479 | 11479 | 11019 | 11019 | 11461 | 11461 |
| LLaMA-7B | GP | 7031 | 7031 | 6593 | 6593 | 5509 | 5509 |
| LLaMA-7B | HP | 1873 | 2977 | 3143 | 1991 | 8713 | 10087 |
| GPT3-2B | Base | OOM | OOM | OOM | OOM | - | - |
| GPT3-2B | HP | 4693 | 4651 | 3153 | 295 | 3975 | 7985 |

#### Key Observations
- **Tensor parallelism**: Reduces memory pressure by weight distribution
- **Balanced distribution**: Apportions LLM according to device capabilities
- **Heterogeneous advantage**: Lower-memory devices get fewer layers
- **OOM handling**: Some baselines fail on GPT3-2B due to memory constraints

## 4.3 Resource Utilization Analysis

### Visualization Results
- **Uniform distribution**: RTX 3090 underutilized (tiny execution time vs P100)
- **HPipe scheduling**: Approximate equal execution time across devices
- **Communication gaps**: Visualized as gaps between colored blocks in Fig 6
- **Device utilization**: Significant improvement with balanced workload

### Ablation Study: Sequence Schedule

#### Dynamic vs Uniform Slicing
- **Test setup**: GPT3-2B and LLaMA-7B, 2048 tokens
- **Uniform slices**: 1 to 128 slices tested
- **Results**: HPipe with dynamic schedule outperforms best uniform configuration
- **Conclusion**: Proper sequence scheduling crucial for optimal performance

#### Slicing Granularity Impact
- **Fine granularity**: GPU underutilization
- **Coarse granularity**: Large pipeline bubbles, increased latency
- **Optimal balance**: Dynamic programming finds sweet spot

## Performance Metrics Summary

### Key Achievements
1. **Latency**: Up to 9.06× reduction
2. **Throughput**: Up to 8.98× improvement
3. **Energy**: 68.2% consumption reduction
4. **Memory**: Better distribution across heterogeneous devices
5. **Scalability**: Works across different model sizes (7B, 2B parameters)

### Technical Comparisons
- **Pipeline vs Tensor**: Token-level pipeline superior for heterogeneous devices
- **Static vs Dynamic**: Dynamic programming crucial for heterogeneous optimization
- **Communication vs Computation**: Balances both for optimal performance