# Phase 2: Methodology Extraction

## Core Methodology: Large-Scale Cross-Node Expert Parallelism

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Mathematical constraint**: For E experts and G GPUs, each expert assigned to distinct GPU if E ≤ G
- **Replication strategy**: If E > G, experts replicated across GPUs to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU expert contention

#### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link
- **Constraint**: Maintain one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation
- Dynamic adjustment of gating probabilities to prevent expert overload

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust routing

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving strategy**: While one batch processes on GPU, next batch transfers from other nodes
- **Implementation**: CUDA streams or asynchronous libraries (NCCL/MPI) for non-blocking transfers

#### 3.2 Pipeline Scheduling
- **Multi-layer optimization**: Token outputs immediately routed to next layer's experts
- **Partial batch processing**: Experts start processing as soon as partial batch arrives
- **Benefit**: Increases throughput, reduces expert idle time

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Network bandwidth**: Primary limiting factor in large EP
- **Mitigation**: Topology-aware routing and token batching
- **Compute utilization**: One-expert-per-GPU ensures full GPU utilization

#### 4.2 Integration with Other Parallelisms
- **Tensor Model Parallelism (TP)**: Applied within individual experts if they exceed single-GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Maintains high expert-level parallelism while supporting large models

## Implementation Details

### Model Architecture Parameters
- Layers: 4 MoE layers
- Experts per layer: 16
- Expert type: MLP with hidden dimension 32768
- Token dimension: 8192
- Attention: 16 heads × 512 dimensions = 8192 total

### Deployment Configuration
- **Baseline**: TP=8, PP=2, 16 GPUs total
  - Each GPU: 1/8 tensor-parallel shard + 8 experts per layer
  - Pipeline stages: 2 stages, 8 GPUs each
- **Proposed**: EP=16, 16 GPUs total
  - Each GPU: 1 expert per layer
  - No tensor or pipeline parallelism within MoE layers

### Performance Optimization
- **Throughput focus**: Maximize TPS (Tokens per Second)
- **Latency focus**: Minimize TPOT (Time per Output Token)
- **Communication cost**: Amortized across many tokens in large batches
- **Memory efficiency**: Balanced expert placement prevents memory hotspots