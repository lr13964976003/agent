# HPipe: Key Points Summary

## Original Abstract
Micro-enterprises and individual developers emerge long context analysis demands with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques cannot fully perform in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the inference through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for long context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.

## Key Problem Statement
1. **Resource Constraints**: Micro-enterprises only have commodity devices with unreliable interconnections (1GB/s vs 900GB/s NV-link)
2. **Long Context Challenges**: Extended sequences (2048+ tokens) create computational pressure
3. **Parallelism Limitations**: Existing methods (tensor+pipeline) don't work well in constrained environments

## Primary Innovations
1. **Token-Dimension Pipeline**: Pipelining on token dimension instead of micro-batch dimension
2. **Heterogeneous Device Utilization**: Dynamic programming to balance workload across different device types
3. **Sequence Scheduling**: Optimal token segmentation using dynamic programming
4. **Layer-level Partitioning**: Finer granularity than transformer block-level partitioning

## Technical Approach
- **Two-phase workflow**: 
  - Prepare phase: Dynamic programming for optimal workload distribution and sequence slicing
  - Runtime phase: Pipeline inference on token dimension
- **Dynamic Programming Algorithms**:
  - Algorithm 1: Workload distribution across heterogeneous devices
  - Algorithm 2: Optimal sequence slicing for pipeline efficiency

## Performance Results
- **Latency**: 2.24s vs 20.3s baseline (9.06× speedup)
- **Throughput**: 5.03k vs 0.56k tokens/s (9× improvement)
- **Energy**: 68.2% reduction compared to other methods
- **Models Tested**: LLaMA-7B, GPT3-2B on P100 + RTX3090 heterogeneous cluster

## Deployment Context
- **Hardware**: 2 host machines (4×P100 + 2×RTX3090)
- **Network**: 1000Mbps inter-host, PCIe intra-host
- **Sequence Length**: 2048 tokens
- **Batch Sizes**: GPT3-2B: 12, LLaMA-7B: 6