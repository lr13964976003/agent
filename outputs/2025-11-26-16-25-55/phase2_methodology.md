# Phase 2: Methodology Extraction

## Core Methodology Components

### 1. Expert Placement Strategy

#### Single-Expert-Per-GPU Deployment
- Each GPU hosts at most one expert per layer
- For E experts and G GPUs: if E ≤ G, assign each expert to distinct GPU
- If E > G, replicate experts across GPUs to maximize concurrency while balancing memory
- Ensures no intra-GPU expert contention

#### Cross-Node Distribution
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimizes maximum tokens sent across any single link

### 2. Routing and Load Balancing

#### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

#### Gating Mechanism
- Standard MoE top-K gating determines expert activation per token
- Dynamic adjustment to prevent overloading specific experts

### 3. Communication Overlap and Scheduling

#### Overlapping Compute and Communication
- Interleave expert computation with cross-node token transfers
- While one batch processes, next batch transfers from other nodes
- CUDA streams and asynchronous libraries (NCCL/MPI) prevent blocking

#### Pipeline Scheduling
- Token outputs immediately routed to next layer
- Experts start processing partial batches rather than waiting for full batch
- Fine-grained pipeline increases throughput, reduces idle time

### 4. Large EP Regime Optimization

#### Large Expert Parallelism (EP ≥ 16)
- Network bandwidth becomes primary limiting factor
- Topology-aware routing and token batching mitigate communication overhead
- One-expert-per-GPU policy ensures full GPU utilization

#### Integration with Other Parallelisms
- **Tensor Model Parallelism (TP)**: Partition experts within GPU if needed
- **Data Parallelism (DP)**: Applied across MoE replicas for synchronized updates
- Maintain high expert-level parallelism while handling large models

### 5. Implementation Parameters

#### Model Specifications
- 61 total layers (3 dense + 58 MoE layers)
- Token dimension: 7168
- MHA: 128 heads × 128 dimensions = 16,384 total attention dimension
- MLP hidden size: 2048
- Precision: BF16

#### Deployment Configurations
- **Proposed method**: 16 experts per MoE layer, 928 total GPUs
- **Baseline method**: 4 experts per GPU, 232 total GPUs
- Both use H100 GPUs with 400 TFLOPS compute, 64GB VRAM, 1.8TB/s bandwidth

#### Communication Settings
- Asynchronous token routing
- CUDA streams for GPU-GPU transfers
- NCCL/MPI for cross-node communication
- 80% bandwidth utilization target
- 60% MFU (Model FLOPS Utilization)