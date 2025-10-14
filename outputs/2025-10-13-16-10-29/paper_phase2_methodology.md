# Phase 2: Methodology Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Detailed Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Mathematical formulation**: For E experts and G GPUs, ensure each expert assigned to distinct GPU if E ≤ G
- **Replication strategy**: If E > G, replicate experts to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU expert contention, full utilization of GPU compute units

#### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link
- **Constraint**: Maintain one-expert-per-GPU principle

### 2. Routing and Load Balancing Mechanism

#### 2.1 Gating Network
- Standard MoE top-K gating scores determine expert activation
- Dynamic adjustment of gating probabilities to prevent expert overloading

#### 2.2 Token Sharding Strategy
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust routing

#### 2.3 Cross-Node Token Transfer
- Efficient transfer of tokens to experts on different nodes
- Reduced network congestion through careful sharding
- Prevention of stragglers that degrade throughput

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Interleaving
- **Parallel execution**: Process current batch while transferring next batch
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Objective**: Prevent data transfer from blocking GPU computation

#### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Immediate routing**: Token outputs routed to next layer's experts without delay
- **Partial batch processing**: Experts start processing as soon as partial batch arrives
- **Benefit**: Fine-grained pipeline increases throughput, reduces idle time

### 4. Large EP Regime Optimization (EP ≥ 16)

#### 4.1 Network Bandwidth Management
- **Primary limiting factor**: Network bandwidth in large EP setups
- **Mitigation strategies**:
  - Topology-aware routing
  - Token batching optimization
  - Communication-computation overlap

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Partition experts within GPU if necessary
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Synchronization**: Maintained weight updates while preserving expert-level parallelism

### 5. Implementation Details

#### 5.1 Hardware Requirements
- **Minimum EP**: 16 (large EP regime)
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand, H100-class NVSwitch)
- **GPU**: Sufficient memory for single expert + overhead

#### 5.2 Software Stack
- **Communication**: NCCL or MPI for asynchronous transfers
- **Scheduling**: CUDA streams for compute-communication overlap
- **Monitoring**: Per-expert load tracking for dynamic balancing

### 6. Mathematical Formulations

#### 6.1 Expert Placement
- Let E = number of experts, G = number of GPUs
- If E ≤ G: one-to-one mapping from experts to GPUs
- If E > G: replication factor r = ceil(E/G), ensuring balanced distribution

#### 6.2 Throughput Optimization
- **Objective**: Maximize TPS (Tokens per Second)
- **Constraint**: TPOT (Time per Output Token) ≤ threshold
- **Variables**: Expert placement, routing probability, batch size

#### 6.3 Load Balancing
- **Metric**: Variance in per-expert token count
- **Optimization**: Minimize variance through dynamic gating adjustment
- **Constraint**: Maintain routing accuracy and model quality