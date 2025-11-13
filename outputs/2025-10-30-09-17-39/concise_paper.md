# HPipe: Large Language Model Pipeline Parallelism for Long Context on Heterogeneous Cost-effective Devices

## Abstract (Full)
Micro-enterprises and individual developers emerge long context analysis demands with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques cannot fully perform in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the inference through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for long context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.

## 1. Introduction and Problem Statement

### Core Challenges
1. **Extended text processing**: Longer inputs (2048+ tokens) create arithmetic pressure on commodity devices
2. **Communication discrepancy**: 
   - Intra-device: PCIe communication (limited bandwidth)
   - Inter-device: Network communication (1000Mbps in experiments)
3. **Device heterogeneity**: Mix of P100 (Pascal) and RTX 3090 GPUs with different compute capabilities
4. **Memory constraints**: Commodity devices have limited VRAM compared to data center GPUs

### Key Insight
Traditional approaches fail because:
- Tensor parallelism requires high-bandwidth NVLink (900GB/s vs 1GB/s commodity)
- Pipeline parallelism in batch dimension limited by small batch sizes
- Uniform distribution ignores device capabilities

## 2. HPipe Framework Design

### 2.1 Two-Phase Architecture

#### Prepare Phase (Static Optimization)
1. **Device profiling**: Measure compute time per layer per device
2. **Network profiling**: Measure communication time between device pairs
3. **Workload distribution**: Dynamic programming to partition layers
4. **Sequence slicing**: Optimal token segmentation for pipeline efficiency

#### Runtime Phase (Pipeline Execution)
1. **Token-level pipeline**: Process sequences in segments across devices
2. **Causal attention**: Maintain decoder causal mask across subsequence boundaries
3. **KV cache reuse**: Cache attention states for subsequent subsequences
4. **Pipeline scheduling**: Overlap computation across devices

### 2.2 Mathematical Formulation

#### Model Structure
- LLM with L transformer layers
- N heterogeneous devices
- Input sequence length S tokens

#### Optimization Variables
- **Layer partition**: {b₁, b₂, ..., b_N} where b_i is layer range for device i
- **Sequence partition**: {s₁, s₂, ..., s_M} where s_j is token range for subsequence j

#### Objective Function
Minimize total latency:
```
T* = min_{partition} [max_i(Σ_j t_{i,j}) + (N-1)max_{i,j} t_{i,j}]
```

Where:
- t_{i,j} = computation_time(device_i, subsequence_j) + communication_time(device_i→device_{i+1})
- First term: slowest device completion time
- Second term: pipeline bubble overhead

### 2.3 Dynamic Programming Algorithms

#### 3.3 Workload Distribution Algorithm
**Input**: Per-layer compute times per device, inter-device communication times
**Output**: Layer ranges per device
**Complexity**: O(L²N) where L=layers, N=devices

Pseudocode:
```
for layers l=1 to L:
    for devices d=1 to N:
        A[l][d] = min_k max(A[k][d-1], T(k+1,l,d))
```

#### 3.4 Sequence Schedule Algorithm
**Input**: Maximum per-slice compute time, per-token compute scaling
**Output**: Optimal token segmentation
**Key insight**: Linear scaling of compute time with token position

## 3. Experimental Validation

### 3.1 Hardware Setup
- **Devices**: 4×P100 + 2×RTX3090 across 2 hosts
- **Network**: 1000Mbps inter-host, PCIe intra-host
- **Models**: LLaMA-7B (7B params), GPT3-2B (2.7B params)
- **Sequence**: 2048 tokens
- **Batch**: 6 (LLaMA-7B), 12 (GPT3-2B)

### 3.2 Performance Results

#### Latency and Throughput
| Method | LLaMA-7B Latency | LLaMA-7B Throughput | Speedup |
|--------|------------------|-------------------|---------|
| Base | 20.3s | 0.56k tokens/s | 1× |
| GPipe | ~4.6s | ~2.4k tokens/s | 4.4× |
| Megatron-LM | ~3.8s | ~3.1k tokens/s | 5.3× |
| **HPipe** | **2.24s** | **5.03k tokens/s** | **9.06×** |

#### Energy Consumption
- **Reduction**: 68.2% vs best baseline
- **Mechanism**: High utilization reduces idle energy waste

#### Memory Distribution
| Device | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 |
|--------|-----|-----|-----|-----|-----|-----|
| Memory (MB) | 1873 | 2977 | 3143 | 1991 | 8713 | 10087 |
| Layers | 4 | 5 | 5 | 4 | 14 | 16 |

## 4. Implementation Details for Deployment

### 4.1 Required Measurements
1. **Compute profiling**: Time per transformer layer per device
   - Measure with representative input sizes
   - Account for memory bandwidth differences
2. **Communication profiling**: 
   - PCIe bandwidth within hosts
   - Network bandwidth between hosts
   - Include serialization overhead

### 4.2 Deployment Pipeline
1. **Offline phase**: Run optimization algorithms
2. **Model partitioning**: Split weights according to layer ranges
3. **Runtime setup**: Configure device communication topology
4. **Sequence processing**: Implement token-level pipeline

### 4.3 Critical Dimensions
- **Layer dimensions**: hidden_size, ffn_hidden_size, num_heads
- **Sequence dimensions**: max_position_embeddings (2048 tested)
- **Memory dimensions**: Per-layer parameter counts, activation sizes
- **Communication dimensions**: Activation tensor sizes between layers