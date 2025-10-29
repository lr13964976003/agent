# Helix Parallelism: Experiments and Results

## Experimental Setup

### Hardware Configuration
- **Platform**: GB200 NVL72
- **Precision**: FP4 (4-bit floating point)
- **Memory Bandwidth**: 8000 GB/s per GPU
- **GPU Range**: 1-64 GPUs per node
- **Communication**: NVLink with large domain support

### Models Evaluated

#### 1. DeepSeek-R1 (MoE, 671B parameters)
- **Architecture**: MoE with MLA attention
- **Attention**: MLA (single KV head shared across 128 query heads)
- **Context Length**: 1 million tokens
- **KV Heads**: K=1 (MLA latent representation)
- **Query Heads**: Q=128

#### 2. Llama-405B (Dense, 405B parameters)
- **Architecture**: Dense transformer with GQA
- **Attention**: GQA (Grouped Query Attention)
- **Context Length**: 1 million tokens
- **KV Heads**: K=8
- **Query Heads**: Q=128
- **Head Size**: Hsz=128

### Baseline Configurations

#### Parallelism Strategies Compared
1. **Tensor Parallelism (TP)**: Traditional sharding
2. **Pipeline Parallelism (PP)**: Layer-wise partitioning
3. **Expert Parallelism (EP)**: MoE-specific expert routing
4. **Vanilla KV Parallelism (KVP)**: Medha-style sequence sharding
5. **Helix Parallelism**: Proposed hybrid approach

#### Search Space
- **Total Configurations**: >100,000 simulated
- **Variables**: Parallelism strategies, batch sizes, GPU counts
- **Constraint**: Single GB200 node (1-64 GPUs)

## Performance Metrics

### Throughput vs. Interactivity Pareto
- **X-axis**: User interactivity = 1/TTL (tokens/sec/user)
- **Y-axis**: System throughput (tokens/sec/GPU)
- **Batch Scalability**: Max concurrent requests under fixed TTL

## Results Summary

### DeepSeek-R1 Results
- **TTL Improvement**: Up to 1.5× reduction in token-to-token latency
- **Batch Scaling**: 32× larger batches under same latency budget
- **Throughput**: Significant improvement in tokens/sec/GPU
- **Context**: 1M+ token KV cache support

### Llama-405B Results
- **TTL Improvement**: 1.13× reduction in maximum latency
- **Batch Scaling**: 4× larger batches vs. TP baseline
- **Throughput**: 4× higher system throughput
- **Comparison**: Consistently outperforms Medha baseline

## Detailed Performance Analysis

### Ablation Study: HOP-B Impact

#### DeepSeek-R1 (HOP-B OFF)
- **Degradation**: ~1% performance drop
- **Reason**: All-to-all exchange is ~1% of decode latency
- **Dominant**: Latent projections and expert computation

#### Llama-405B (HOP-B OFF)  
- **Degradation**: ~12% performance drop
- **Reason**: Communication forms larger fraction of TTL
- **Recovery**: HOP-B overlap recovers most lost performance

### Configuration Examples

#### Optimal DeepSeek-R1 Configuration
- **KVP**: 8 (sequence sharding)
- **TPA**: 1 (since K=1 in MLA)
- **TPF**: 8 (FFN tensor parallelism)
- **Total GPUs**: 8
- **Batch Size**: Scaled to 32× baseline

#### Optimal Llama-405B Configuration
- **KVP**: 4 (sequence sharding)
- **TPA**: 4 (maximizing K=8 constraint)
- **TPF**: 8 (FFN tensor parallelism)
- **Total GPUs**: 16
- **Batch Size**: Scaled to 4× baseline

## Empirical Validation

### Roofline Analysis Results
- **KV Cache Scaling**: Sublinear with KVP (theoretical vs. observed)
- **FFN Weight Scaling**: Linear improvement with TPF
- **Communication Overhead**: Validated HOP-B overlap effectiveness

### Memory Utilization
- **KV Cache Distribution**: Even across KVP ranks
- **Weight Sharding**: Optimized for DRAM bandwidth
- **Memory Growth**: Balanced via round-robin KV updates

## Key Findings

### Architectural Insights
1. **MLA Compatibility**: Helix naturally supports single-head attention
2. **MoE Integration**: Seamless EP integration without KV duplication
3. **Dense Model Performance**: Significant gains for traditional GQA

### Scalability Patterns
- **Linear Scaling**: With KVP for KV cache
- **Superlinear Scaling**: Combined KVP+TP optimization
- **Communication**: Constant overhead independent of sequence length

### Practical Deployment
- **GPU Count**: 8-64 GPUs optimal range
- **Context Length**: 1M+ tokens practical for real-time
- **Batch Sizes**: 4-32× increase possible under TTL constraints