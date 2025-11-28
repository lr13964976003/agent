# Phase 2: Methodology Extraction

## **Methods - Detailed Methodology**

### **1. Core Method Overview**
The proposed method maximizes expert-level parallelism by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. This shifts the optimization focus from reducing communication overhead to maximizing compute concurrency.

### **2. Expert Placement Strategy**

#### **2.1 Single-Expert-Per-GPU Deployment**
- **Constraint**: Exactly one expert per GPU
- **Mathematical Formulation**: For E experts and G GPUs:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize independent expert concurrency
- **Memory Balance**: Ensures balanced memory usage during replication

#### **2.2 Cross-Node Distribution**
- **Topology-Aware Placement Algorithm** considers:
  - Node-to-node bandwidth and latency metrics
  - GPU memory capacity per node
  - Expected token routing patterns from historical data
- **Optimization Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU constraint

### **3. Routing and Load Balancing**

#### **3.1 Gating Mechanism**
- **Standard MoE Gating**: Top-K gating scores determine expert activation
- **K Value**: Not explicitly specified, but typically K=2 for MoE models
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities to prevent overload

#### **3.2 Token Sharding Across Nodes**
- **Token Batching**: Group tokens by destination expert to minimize network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Real-time monitoring with dynamic gating probability adjustment

### **4. Communication Overlap and Scheduling**

#### **4.1 Overlapping Compute and Communication**
- **Interleaving Strategy**: 
  - While current batch processes, next batch transfers simultaneously
  - CUDA streams or NCCL/MPI for asynchronous communication
- **GPU Utilization**: Data transfer never blocks GPU computation

#### **4.2 Pipeline Scheduling**
- **Layer-to-Layer Flow**: Token outputs immediately routed to next layer
- **Partial Batch Processing**: Subsequent layer experts start processing as soon as partial batch arrives
- **Throughput Optimization**: Fine-grained pipeline reduces idle time per expert

### **5. Scalability Considerations**

#### **5.1 Large EP Regime**
- **Definition**: EP ≥ 16 (256 experts per layer in experiments)
- **Communication Bottleneck**: Network bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **GPU Utilization**: One-expert-per-GPU ensures full compute utilization

#### **5.2 Memory and Model Parallelism Integration**
- **Tensor Parallelism (TP)**: Partition experts within GPU if model exceeds single-GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Integration**: Seamless combination with TP and DP while maintaining high expert-level parallelism

### **6. Multi-Head Latent Attention (MLA) - Technical Details**

#### **6.1 Memory Reduction Mechanism**
- **Standard Attention**: Q/K/V all have hidden_dim (7168) size
- **MLA Approach**: Compress X → K_latent dimension << hidden_dim
- **Per-Head Projection**: Each head projects from latent space to individual K_head

#### **6.2 Dimension Reduction**
- **Original**: KV cache size scales with hidden_dim × num_heads
- **MLA**: Shared latent representation reduces memory by factor related to latent_dim/hidden_dim ratio
- **Efficiency**: Heaviest computation moved outside heads and shared among them

### **7. Implementation Parameters**
- **GPU Allocation**: 256 GPUs per MoE layer (one per expert)
- **Communication Library**: NCCL or MPI for cross-node communication
- **Scheduling**: CUDA streams for asynchronous operations
- **Precision**: FP8 for memory efficiency and speed
- **Batch Processing**: Variable batch size optimized for sequence length

### **8. Mathematical Formulation**
- **Expert Assignment**: Map E experts to G GPUs: f: {1..E} → {1..G} with |f-1(g)| ≤ 1 for all g ∈ G
- **Load Balancing**: Minimize max_g ∑_e tokens routed to expert e on GPU g
- **Communication Cost**: Minimize ∑_i,j tokens transferred between node i and node j