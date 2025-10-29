# HPipe: Large Language Model Pipeline Parallelism for Long Context on Heterogeneous Cost-effective Devices

## Abstract
Micro-enterprises and individual developers emerge long context analysis demands with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques cannot fully perform in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the inference through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for long context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.

## 1. Introduction

The emergence of LLMs has significantly enhanced automated content comprehension, as they adeptly capture semantic information within extensive contexts. Enterprises employ techniques such as sentiment analysis and content analysis to harness the potential value to facilitate the anticipation of user engagement and strategic decision-making. However, due to the stringent memory and computational requirements of LLMs, they are commonly deployed on high-performance computing clusters with advanced devices and high-velocity transmission like NV-link (900GB/s).

While micro-enterprises introduce demands to leverage private LLMs, they only have inconsistent weaker devices with interconnection via wireless networks (up to 1GB/s). Thus, customized LLM deployment schemas for micro-enterprises deserve further exploration.

### Deployment Challenges for Micro-enterprises
1. **Extended text**: Longer inputs bring higher arithmetic pressure, causing micro-batch pipeline inefficiency
2. **Communication discrepancy**: Discrepant communication conditions (PCIe vs network) impede tensor parallelism
3. **Heterogeneous devices**: Dual heterogeneity of computation and transmission with expensive communication

To address these challenges, we propose HPipe, a pipeline inference framework dedicated to content comprehension for private LLMs. It deploys LLMs on heterogeneous devices with pipeline parallelism on the token dimension, achieving up to 2.28× increase in both latency and throughput, alongside a 68.2% reduction in energy consumption.

## 2. Background and Motivation

### 2.1 Parallelism Limitations
- **Tensor parallelism**: Requires guaranteed high-bandwidth transmission
- **Pipeline parallelism**: Memory constraints limit batch size, reducing parallelism degree
- **Key insight**: Decoder-based transformers enable pipeline on token dimension without affecting results

### 2.2 Utilization Analysis
- **FLOPs Utilization**: Relationship between sequence length and resource utilization shows optimal length exists
- **Tile Quantization**: Matrix dimensions must align with GPU tile size for optimal utilization

### 2.3 Motivation
Pipeline parallelism advantages for LLMs in constrained environments:
- Reduces computational load with tolerant communication requirements
- Decoder transformers facilitate token-level pipelining
- Caches K,V values for subsequent calculations
- Maximizes resource utilization through fine-grained execution

## 3. Method

### 3.1 Workflow Overview
HPipe operates in two phases:
1. **Prepare Phase**: Determines optimal workload distribution and sequence slicing through dynamic programming
2. **Runtime Phase**: Executes inference with token-level pipeline parallelism

### 3.2 Mathematical Formulation

#### Model Partitioning
- **Model**: LLM with n layers {l₁, l₂, ..., lₙ}
- **Distribution**: N blocks {b₁, b₂, ..., bₙ} across N devices
- **Sequence**: Input segmented into M subsequences in token dimension

#### Execution Time Model
For subsequence sᵢ on device dⱼ:
```
tᵢⱼ = tᶜ(sᵢ, Σₖ₌₁ⁱ⁻¹ sₖ; dⱼ) + tᵗ(lⱼ, sᵢ, B)
```

Where:
- tᶜ: Computation time for sᵢ with previous context on device dⱼ
- tᵗ: Transmission time for intermediate activations via bandwidth B

#### Optimization Objective
```
T* ≤ maxᵢ∈N(Σⱼ₌₀ᴹ tᵢⱼ) + (N-1) max₀≤i<M,0≤j<N{tᵢⱼ}
```

### 3.3 Distribution Balance Algorithm

#### Dynamic Programming Approach
```
T(a,b,m) = Σₖ₌ₐᵇ t_comp(lₖ; dₘ) + t_comm(lⱼ, m)
A[b][m] = min₁≤k<j{max{A[k][m-1], T(k+1,b,m)}}
```

#### Algorithm Features
- **Granularity**: Layer-level partition (not block-level)
- **Complexity**: NP-hard problem simplified with constant device sequence assumption
- **Output**: N-1 cut points for optimal LLM partition

### 3.4 Sequence Schedule Algorithm

#### Key Insight
Token processing time increases linearly with position due to attention mechanism.

#### Dynamic Programming Formulation
```
For t_max (maximum slice execution time):
- Enumerate possible t_max values
- Find optimal slicing S* from space S
- Minimize: (M-1)t_max + L[N]
```

#### Implementation
- **Granularity Tradeoff**: Balance between GPU underutilization (fine) and pipeline bubbles (coarse)
- **Optimal Division**: Longer slices at beginning, shorter toward end
- **Caching**: K,V values cached for subsequent calculations

## 4. Evaluation

### 4.1 Experimental Setup

#### Hardware Configuration
- **Cluster**: 2 host machines
  - Machine 1: 4× Pascal P100 GPUs
  - Machine 2: 2× RTX 3090 GPUs
- **Network**: 1000 Mbps wired inter-host, PCIe intra-host

#### Models and Parameters
- **GPT3-2B**: Batch size 12, sequence length 2048 tokens
- **LLaMA-7B**: Batch size 6, sequence length 2048 tokens

#### Baseline Methods
1. **Base**: Uniform distribution, sequential inference
2. **GPipe**: Even distribution with micro-batch pipelining
3. **GP-B**: GPipe with HPipe's balanced distribution
4. **Megatron-LM**: Tensor + pipeline parallelism
5. **Terapipe**: Token-level pipeline with even distribution
6. **TP-T**: Tensor + token-level pipeline

### 4.2 Performance Results

#### Latency and Throughput

**LLaMA-7B Results:**
- Base: 20.3s latency, 0.56k tokens/s throughput
- HPipe: 2.24s latency, 5.03k tokens/s throughput
- Speedup: 9.06× latency reduction, 8.98× throughput improvement

**GPT3-2B Results:**
- Base: Out of memory on P100 devices
- HPipe: Successful inference with significant speedup
- Maximum: 2.28× speedup in both latency and throughput

#### Energy Consumption
- **Reduction**: 68.2% lower energy consumption vs. baseline
- **Reason**: High resource utilization through optimal sequence selection

#### Memory Footprint (MB)

| Model | Method | P@1 | P@2 | P@3 | P@4 | R@1 | R@2 |
|-------|--------|-----|-----|-----|-----|-----|-----|
| LLaMA-7B | HPipe | 1873 | 2977 | 3143 | 1991 | 8713 | 10087 |
| GPT3-2B | HPipe | 4693 | 4651 | 3153 | 295 | 3975 | 7985 |

### 4.3 Resource Utilization

#### Key Findings
- **Uniform Distribution**: RTX 3090 idle time due to computational imbalance
- **HPipe Schedule**: Balanced execution across heterogeneous devices
- **Pipeline Efficiency**: Minimized bubbles through optimal slicing

## 5. Conclusion

HPipe successfully enables LLM deployment on heterogeneous commodity hardware through:
- **Token-level pipeline parallelism** for long-context analysis
- **Dynamic programming algorithms** for optimal workload distribution and sequence slicing
- **Significant performance gains**: 2.28× speedup and 68.2% energy reduction
- **Practical deployment**: Makes private LLM deployment feasible for micro-enterprises

The framework demonstrates the potential to accelerate LLM inference with long sequence inputs on cost-effective hardware environments.