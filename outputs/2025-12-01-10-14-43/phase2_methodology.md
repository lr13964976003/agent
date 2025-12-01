# Phase 2: Methodology Extraction

## **Expert Placement Strategy**

### **Single-Expert-Per-GPU Deployment**
- Each GPU hosts at most one expert per layer
- For E experts and G GPUs: if E ≤ G, each expert assigned to distinct GPU
- If E > G, experts replicated to maximize concurrency while balancing memory usage
- Ensures each expert processes tokens without contention from other experts on same device

### **Cross-Node Distribution**
Topology-aware placement strategy considering:
- Node-to-node bandwidth and latency
- GPU memory capacity per node  
- Expected token routing patterns
- Minimizes maximum number of tokens sent across any single link
- Maintains one-expert-per-GPU principle

## **Routing and Load Balancing**

### **Gating Mechanism**
Standard MoE top-K gating scores determine which subset of experts is activated for each token

### **Token Sharding Across Nodes**
1. **Token Batching**: Group tokens by destination expert to reduce network messages from O(B×S×K) to O(E)
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## **Communication Overlap and Scheduling**

### **Overlapping Compute and Communication**
- Interleave expert computation and communication
- While one batch processes on GPU, next batch simultaneously transferred from other nodes
- CUDA streams and asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block GPU computation

### **Pipeline Scheduling**
- Token outputs from previous MoE layer immediately routed to next layer's experts
- Experts in subsequent layers start processing partial batches rather than waiting for complete batches
- Fine-grained pipeline increases throughput and reduces idle time

## **Scalability Considerations**

### **Large EP Regime (EP ≥ 16)**
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU policy ensures all GPUs fully utilized for compute
- Communication costs amortized across many tokens

### **Memory and Model Parallelism Integration**
- Each expert can be further partitioned using tensor model parallelism (TP) within its GPU if necessary
- Data parallelism (DP) applied across replicas of MoE network
- Allows synchronized weight updates while maintaining high expert-level parallelism