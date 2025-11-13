# HPipe: Large Language Model Pipeline Parallelism for Long Context on Heterogeneous Cost-effective Devices

## Abstract

Micro-enterprises and individual developers emerge long context analysis demands with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques cannot fully perform in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the inference through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for long context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.

## 1 Introduction

The emergence of LLMs has significantly enhanced automated content comprehension, yet their deployment faces three major challenges for micro-enterprises: (1) Extended text creates computational pressure as context windows expand beyond 2048 tokens; (2) Communication discrepancy between PCIe intra-device and network inter-device connections; (3) Heterogeneous devices must be integrated to maximize available resources while managing expensive communication costs.

## 2 Background and Motivation

### 2.1 Parallelism Analysis

Pipeline and tensor parallelism are popular methods for LLM acceleration. Pipeline parallelism distributes LLMs across devices with each handling a computation stage, while tensor parallelism divides individual weight matrices. For constrained environments, pipeline parallelism on the token dimension is advantageous due to its communication-lightweight nature and compatibility with decoder-based transformers.

### 2.2 Device Utilization Insight

FLOPs utilization analysis reveals optimal sequence lengths for different hardware. As sequence length increases, utilization initially improves then decreases due to I/O bottlenecks. Matrix multiplication achieves maximum GPU utilization when dimensions align with tile sizes, necessitating careful sequence length selection.

## 3 Method

### 3.1 HPipe Workflow

HPipe operates in two phases:

**Prepare Phase**: Analyze device specifications and network conditions, determine optimal workload distribution and sequence slicing using dynamic programming, precompute slicing schemes for all sequence lengths.

**Runtime Phase**: Divide input sequence into subsequences, execute pipeline on token dimension with KV cache persistence, handle intermediate activations between stages.

### 3.2 Mathematical Formulation

Let LLM have n layers {l₁,...,lₙ} divided into N blocks across N devices. Input sequence segmented into M subsequences in token dimension.

**Execution Time Model**:
```
tᵢⱼ = tᶜ(∑ₘ₌₁ⁱ⁻¹ sₘ; dⱼ) + tᵗ(lⱼ, sᵢ, B)
```

**Optimization Objective**: Minimize total latency T*:
```
T* ≤ maxᵢ∈N(∑ⱼ₌₀ᴹ tᵢⱼ) + (N-1)max{tᵢⱼ}
```

### 3.3 Distribution Balance

**Problem**: NP-hard device placement for heterogeneous environment
**Solution**: Dynamic programming with layer-level granularity

**Algorithm**:
```
T(a,b,m) = ∑ₖ₌ₐᵇ t_comp(lₖ; dₘ) + t_comm(lⱼ, m)
A[b][m] = min₁≤k<j {max{A[k][m-1], T(k+1,b,m)}}
```

### 3.4 Sequence Schedule

Use dynamic programming to optimize sequence slicing considering:
- Linear relationship between token position and execution time
- Fine-grained vs coarse-grained granularity trade-offs
- Execution time constraint tₘ for balanced pipeline stages

**Algorithm**:
```
T* ≤ min_{tₘ} {max computation time} + (N-1)tₘ
```

## 4 Evaluation

### 4.1 Experimental Setup

**Hardware**: 2 host machines (4×P100 + 2×RTX3090), 1000Mbps inter-host, PCIe intra-host
**Models**: LLaMA-7B (32 layers, 7B params), GPT3-2B (24 layers, 2.7B params)
**Input**: 2048 tokens, batch sizes: GPT3-2B=12, LLaMA-7B=6

### 4.2 Results

**Performance Improvements**:
- LLaMA-7B: 2.24s vs 20.3s baseline (9.06× speedup), 5.03k vs 0.56k tokens/s
- GPT3-2B: 2.28× speedup achieved
- Energy: 68.2% reduction compared to other methods

**Memory Management**:
- Avoids OOM conditions through layer-level distribution
- P@4 and R@1 have lower memory due to fewer layers for communication overhead compensation

**Resource Utilization**:
- RTX3090 shows 10× faster execution than P100
- Balanced workload minimizes idle time
- Dynamic slicing outperforms uniform configurations

## 5 Conclusion

HPipe enables efficient LLM deployment on heterogeneous commodity hardware through token-dimension pipeline parallelism, achieving significant speedups for long-context analysis while maintaining energy efficiency and memory optimization.

## Appendix: Algorithm Specifications

**Algorithm 1: Workload Distribution**
- Input: Layer computation/communication times per device
- Output: Minimal slowest execution time and layer distribution
- Complexity: O(N²M)

**Algorithm 2: Sequence Slicing**
- Input: Sequence length, execution time matrix, device times
- Output: Optimal slicing scheme and expected latency
- Complexity: O(N²)

**Deployment Configuration**:
- LLaMA-7B: 32 layers → [3,4,5,6,7,7] distribution across [P@1,P@2,P@3,P@4,R@1,R@2]
- GPT3-2B: 24 layers → [4,4,4,4,4,4] distribution across same devices
- Sequence slicing: Dynamic programming with tₘ=240ms (LLaMA), tₘ=200ms (GPT3)