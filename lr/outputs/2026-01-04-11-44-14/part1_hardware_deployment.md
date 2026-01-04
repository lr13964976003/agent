# Hardware Deployment Environment

## GPU Configuration
- **GPU Model**: NVIDIA H100 GPUs (adequate number available)
- **Deployment Strategy**: One expert per GPU per layer
- **Memory Configuration**: Each GPU hosts exactly one expert per layer
- **Compute Capability**: Full GPU compute utilization per expert

## Network Infrastructure
- **Interconnect**: High-performance HPC networking (NVLink, InfiniBand, H100-class NVSwitch)
- **Bandwidth Requirements**: High bandwidth, low latency across nodes
- **Topology**: Topology-aware placement considering node-to-node bandwidth and latency

## CUDA Stream Configuration
- **Overlapping Strategy**: CUDA streams used for overlapping compute and communication
- **Asynchronous Operations**: Non-blocking data transfer during GPU computation
- **Stream Management**: Separate streams for token transfer and expert computation

## Communication Libraries
- **Primary Library**: NCCL (NVIDIA Collective Communications Library)
- **Alternative**: MPI for cross-node communication
- **Implementation**: Asynchronous communication with token batching

## Memory Requirements
- **Per-GPU Memory**: Sufficient for single expert per layer (4096 token dim, 16384 MLP hidden size)
- **Precision**: BF16 (Brain Floating Point 16-bit)
- **Buffer Management**: Token batching buffers for asynchronous routing

## Large EP Regime Specifications
- **Definition**: EP ≥ 16 (Large Expert Parallelism)
- **Deployment**: 16 GPUs minimum for full expert parallelism
- **Scaling**: Near-linear scaling with adequate H100 GPUs
- **Placement**: Cross-node distribution to minimize hotspotting

## Topology-Aware Placement
- **Algorithm Considerations**:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
  - Minimization of maximum tokens per link
- **Objective**: Balance network load while maintaining one-expert-per-GPU principle