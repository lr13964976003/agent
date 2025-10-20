# Phase 2: Complete Methodology of Cross-Node Expert Parallelism

## Background: Mixture-of-Experts Architecture

### MoE in Large-Scale Models
MoE models replace transformer FFN layers with multiple "experts," each trained to specialize in different input patterns. A gating mechanism determines which subset of experts is activated for each token, leading to sparse computation and improved efficiency.

### Parallelism Strategies for MoE
Scaling MoE involves combination of:
- **Data Parallelism (DP)**: Replicas across nodes
- **Tensor Model Parallelism (TP)**: Splits layers across devices
- **Pipeline Parallelism (PP)**: Splits model depth-wise
- **Expert Parallelism (EP)**: Partitions experts across devices

Traditional implementations use moderate EP degree, placing multiple experts per GPU to limit communication.

### Large Expert Parallelism (Large EP) Definition
**Large EP** = configurations where EP ≥ 16. In this regime:
- Network interconnects (NVLink, InfiniBand, H100 NVSwitch) make communication cost less dominant
- Distributing experts across devices maximizes compute concurrency
- Challenge: coordinate cross-node communication and balance routing

## Methods

### 1. Overview
Three key components for maximizing expert-level parallelism:
1. **Expert Placement Strategy** - Assigning experts across GPUs/nodes
2. **Routing and Load Balancing** - Balanced input distribution
3. **Communication Overlap and Scheduling** - Minimize cross-node transfer impact

### 2. Expert Placement Strategy

#### 2.1 Single-Expert-Per-GPU Deployment
**Constraint**: At most one expert per GPU
**Mathematical Formulation**:
- For MoE layer with E experts and cluster of G GPUs
- Ensure each expert assigned to distinct GPU if E ≤ G
- If E > G, replicate experts to maximize concurrency while balancing memory

**Benefit**: Eliminates intra-GPU expert contention, fully utilizes GPU compute units

#### 2.2 Cross-Node Distribution
**Topology-aware placement** considers:
- Node-to-node bandwidth and latency
- GPU memory capacity per node
- Expected token routing patterns

**Placement algorithm objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

### 3. Routing and Load Balancing

#### 3.1 Gating Mechanism
Standard MoE routing: top-K gating scores determine activated experts per token

#### 3.2 Token Sharding Across Nodes
**Efficient cross-node routing includes**:
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously, overlapping with computation
3. **Dynamic Load Balancing**: Monitor per-expert load, adjust gating probabilities to prevent overloading

### 4. Communication Overlap and Scheduling

#### 4.1 Overlapping Compute and Communication
**Interleaving strategy**:
- Process one token batch on GPU while simultaneously transferring next batch
- CUDA streams or asynchronous communication (NCCL/MPI) ensure data transfer doesn't block computation

#### 4.2 Pipeline Scheduling
**Multi-layer MoE optimization**:
- Token outputs from previous MoE layer immediately routed to next layer's experts
- Subsequent layer experts start processing partial batches instead of waiting for full batch
- Fine-grained pipeline increases throughput, reduces idle time

### 5. Scalability Considerations

#### 5.1 Large EP Regime (EP ≥ 16)
**Optimization characteristics**:
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU ensures full GPU utilization while amortizing communication costs

#### 5.2 Memory and Model Parallelism Integration
**For models exceeding single-GPU memory**:
- Each expert can be partitioned using TP within its GPU
- DP applied across MoE network replicas, synchronized weight updates maintained
- High expert-level parallelism preserved

### 6. Summary of Technical Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load**: Topology-aware placement and dynamic gating prevent bottlenecks
3. **Scalable Overlap**: Asynchronous routing enables near-linear scaling for EP ≥ 16
4. **Model Compatibility**: Seamless integration with TP and DP for large models