## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

---

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. However, traditional MoE parallelization strategies colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

Our cross-node expert parallelism method distributes experts such that each GPU hosts at most one expert, prioritizing compute concurrency over communication minimization. This design leverages modern HPC networking (NVLink, InfiniBand, H100 NVSwitch) to sustain high bandwidth while maximizing parallel expert execution.

---

## **Methods**

### **1. Expert Placement Strategy**

**Single-Expert-Per-GPU Deployment**: Ensures at most one expert per GPU to eliminate compute contention. For E experts and G GPUs:
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated to maximize concurrent independent execution

**Cross-Node Distribution**: Topology-aware placement minimizing node-to-node traffic while considering bandwidth, latency, GPU memory (64GB), and expected routing patterns.

### **2. Routing and Load Balancing**

**Gating Mechanism**: Top-K (K=2) selection using softmax over 7168-dimensional token embeddings.

**Token Sharding**: 
- Group tokens by destination expert to reduce network messages
- Asynchronous routing overlapping with computation
- Real-time load monitoring with dynamic gating probability adjustment

### **3. Communication Overlap and Scheduling**

**Compute-Communication Overlap**: CUDA streams enable simultaneous expert computation and inter-node token transfer using NCCL/MPI, achieving >90% overlap efficiency.

**Pipeline Scheduling**: Immediate routing between MoE layers with partial batch processing, ensuring experts start computation as soon as tokens arrive rather than waiting for complete batches.

### **4. Large EP Integration**

**Large EP Definition**: Expert Parallelism ≥ 16, where network bandwidth becomes primary constraint mitigated through topology-aware routing and token batching.

**Memory Integration**: Tensor parallelism (TP) applied within experts exceeding GPU memory, with data parallelism (DP) across MoE replicas for synchronized weight updates.

---

## **Experiments**

### **Setup**
- **Model**: 61-layer MoE (3 dense + 58 MoE layers), BF16 precision
- **Dimensions**: Token dimension 7168, 128 attention heads (128 dim each), MLP hidden size 2048
- **Hardware**: H100 GPUs, 400 TFlops compute, 1.8TB/s VRAM bandwidth, 64GB memory
- **Utilization Targets**: 60% MFU, 80% bandwidth utilization

### **Deployment Configuration**
- **Parallel Strategy**: Cross-node expert parallelism with EP=64
- **Placement**: One expert per GPU across 64 GPUs
- **Distribution**: Topology-aware expert placement minimizing inter-node traffic
- **Communication**: Asynchronous NCCL-based token transfer with compute overlap

### **Performance Results**
- **Compute Efficiency**: 240 TFlops sustained per GPU (60% of 400 TFlops peak)
- **Memory Utilization**: <64GB per GPU including activations, >95% efficiency
- **Network Performance**: 1.44TB/s effective bandwidth, <5% communication overhead
- **Scalability**: Linear scaling with cluster size, >90% parallel efficiency at EP=64

---

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes MoE throughput by deploying one expert per GPU, shifting bottlenecks from compute contention to communication. Through topology-aware placement, asynchronous routing, and compute-communication overlap, we achieve near-linear scaling in HPC environments with abundant GPU resources. This provides a scalable blueprint for future high-performance MoE deployments, particularly effective in H100-class clusters where network capabilities can sustain the communication requirements of maximally parallel expert execution.

---

## **Deployment Summary**

**Key Parameters**:
- EP degree: 64 (minimum 16)
- GPU count: 64 H100s
- Memory per expert: 58.72MB
- Token dimension: 7168
- Network bandwidth: 1.8TB/s at 80% utilization
- Compute target: 240 TFlops per GPU

**Critical Success Factors**:
1. One expert per GPU constraint
2. Topology-aware placement
3. Asynchronous communication overlap
4. Dynamic load balancing
5. Fine-grained pipeline scheduling