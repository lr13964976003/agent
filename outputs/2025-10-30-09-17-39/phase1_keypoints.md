# HPipe Paper - Key Points Extraction

## Abstract (Preserved in full)
Micro-enterprises and individual developers emerge long context analysis demands with powerful Large Language Models (LLMs). They try to deploy the LLMs at local, but only possess various commodity devices and the unreliable interconnection between devices. Existing parallel techniques cannot fully perform in limited environment. The heterogeneity of devices, coupled with their limited capacity and expensive communication, brings challenges to private deployment for maximized utilization of available devices while masking latency. Hence, we introduce HPipe, a pipeline inference framework that successfully mitigates LLMs from high-performance clusters to heterogeneous commodity devices. By ensuring a balanced distribution of workloads, HPipe facilitates the inference through pipelining the sequences on the token dimension. The evaluation conducted on LLaMA-7B and GPT3-2B demonstrates that HPipe holds the potential for long context analysis on LLM with heterogeneity devices, achieving an impressive speedup in latency and throughput up to 2.28 times.

## Key Problem Statement
1. **Extended text challenge**: Longer inputs create higher arithmetic pressure, making micro-batch pipeline inefficient
2. **Communication discrepancy**: Devices have different communication capabilities (PCIe vs network)
3. **Heterogeneous devices**: Need to integrate various commodity devices with different computing capabilities

## Core Innovation - HPipe Framework
- **Pipeline parallelism on token dimension** instead of traditional batch dimension
- **Dynamic workload distribution** using dynamic programming to handle device heterogeneity
- **Optimal sequence slicing** to maximize resource utilization
- **Two-phase workflow**: Prepare phase (optimization) + Runtime phase (pipeline execution)

## Technical Contributions
1. **Distribution Balance**: NP-hard problem solved via dynamic programming to partition layers across heterogeneous devices
2. **Sequence Schedule**: Dynamic programming algorithm to find optimal token sequence slicing
3. **Token-level pipeline**: Processes sequences in segments across devices while maintaining causal attention

## Performance Results
- **Latency reduction**: Up to 9.06× speedup vs baseline on LLaMA-7B
- **Throughput improvement**: Up to 5.03k tokens/s (from 0.56k)
- **Energy efficiency**: 68.2% reduction in energy consumption
- **Resource utilization**: Better device utilization through balanced workload distribution

## Experimental Setup
- **Devices**: 4× P100 + 2× RTX 3090 across two hosts
- **Models**: LLaMA-7B, GPT3-2B
- **Sequence length**: 2048 tokens
- **Batch sizes**: 12 (GPT3-2B), 6 (LLaMA-7B)
- **Network**: 1000Mbps between hosts, PCIe within hosts