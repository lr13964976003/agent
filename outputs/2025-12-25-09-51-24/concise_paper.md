# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures scale large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert parallelism.

We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methodology

### Core Strategy: Large Expert Parallelism (EP ≥ 16)
- **Definition**: Configurations with 16 or more experts per parallel group
- **Principle**: Deploy at most one expert per GPU
- **Constraint**: For E experts and G GPUs, each expert assigned to distinct GPU if E ≤ G
- **Replication**: If E > G, experts replicated across GPUs to maximize concurrency

### Expert Placement Strategy
- **Single-expert-per-GPU**: Eliminates intra-GPU expert contention
- **Topology-aware distribution**: Considers node-to-node bandwidth, latency, and GPU memory
- **Objective**: Minimize maximum tokens sent across any single link

### Routing and Load Balancing
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous routing**: Send token batches while overlapping expert computation
- **Dynamic adjustment**: Monitor per-expert load and adjust gating probabilities

### Communication Overlap and Scheduling
- **Interleaving**: Process one batch while transferring next batch from other nodes
- **Implementation**: CUDA streams or NCCL for non-blocking transfers
- **Pipeline scheduling**: Token outputs immediately routed to next layer's experts

## Experimental Evaluation

### Model Configuration
- **Architecture**: 4-layer MoE, 16 experts per layer
- **Expert type**: MLP with hidden dimension 32768
- **Token dimension**: 8192
- **Attention**: 16 heads × 512 dimensions = 8192 total
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens

### Deployment Configurations

#### Baseline (TP=8, PP=2)
- **GPUs**: 16 H100s
- **Tensor Parallelism**: 8-way splitting
- **Pipeline Parallelism**: 2 stages, 8 GPUs per stage
- **Expert placement**: 8 experts per GPU per layer
- **Performance**: TPS = 120,000, TPOT = 8.3ms

#### Proposed (EP=16)
- **GPUs**: 16 H100s
- **Expert Parallelism**: 16 (one expert per GPU)
- **Expert placement**: 1 expert per GPU per layer
- **Communication**: Asynchronous token routing with batching
- **Performance**: TPS = 450,000, TPOT = 2.2ms

### Results
| Method | GPUs | Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|------|------------|----------------|-----------|
| Baseline | 16 | 8 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed | 16 | 1 expert per GPU | 450,000 | 2.2 |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## Key Innovations

1. **Large EP Definition**: EP ≥ 16 qualifies as "large EP" regime
2. **One-expert-per-GPU**: Eliminates intra-GPU contention
3. **Topology-aware placement**: Minimizes network hotspots
4. **Communication-compute overlap**: Asynchronous routing with CUDA streams
5. **Dynamic load balancing**: Prevents expert overload

## Deployment Requirements

### Hardware Specifications
- **GPUs**: 16× H100 (minimum for EP=16)
- **Interconnect**: NVLink + InfiniBand for cross-node communication
- **Memory**: Sufficient for single expert per GPU (32768 × 8192 × 2 bytes per expert)

### Software Stack
- **Communication libraries**: NCCL, MPI
- **CUDA streams**: For asynchronous operations
- **Load balancing**: Dynamic gating probability adjustment
- **Token routing**: Top-K gating with batching by destination

## Scalability Considerations

### Large EP Regime (EP ≥ 16)
- **Network bandwidth**: Primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Compute utilization**: One-expert-per-GPU ensures full utilization

### Integration with Other Parallelisms
- **Tensor Model Parallelism**: Within individual experts if needed
- **Data Parallelism**: Across MoE network replicas
- **Compatibility**: Maintains high expert-level parallelism

## Performance Characteristics

### Throughput Optimization
- **Maximal expert parallelism**: All 16 experts compute simultaneously
- **No resource contention**: Dedicated GPU per expert
- **Efficient communication**: Amortized across large batches

### Latency Optimization
- **Minimal queuing**: Direct expert access
- **Overlapped transfers**: Communication hidden by computation
- **Balanced load**: Prevents straggler experts

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. With EP ≥ 16, we achieve 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. The method provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, particularly effective for HPC clusters with advanced networking capabilities.

## Implementation Notes

### Critical Dimensions
- Expert hidden size: 32768
- Token dimension: 8192
- Batch size: 1024 sequences
- Sequence length: 10000 tokens

### Deployment Configuration
Complete deployment specifications are provided in `deployment_config.json`, including:
- Device mappings for both baseline and proposed methods
- Parallel strategy parameters
- Communication configurations
- Performance metrics

This approach enables near-linear scaling for large EP regimes while maintaining compatibility with existing model parallelism techniques for handling models exceeding single-GPU memory capacity.