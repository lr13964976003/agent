# Condensed Paper: Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization. Traditional strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert parallelism.

Our work presents a cross-node expert parallelism method prioritizing expert distribution such that each GPU hosts at most one expert, pushing Expert Parallelism (EP) to large numbers. This shifts the optimization focus from communication reduction to compute concurrency maximization.

## **Methods**

### **1. Expert Placement Strategy**

**Single-Expert-Per-GPU Deployment:**
- Each GPU hosts at most one expert per layer
- For E experts and G GPUs: E ≤ G ensures full distribution
- When E > G: replicate experts across GPUs balancing memory usage

**Cross-Node Distribution:**
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Objective: Minimize maximum tokens sent across any link while maintaining one-expert-per-GPU

### **2. Routing and Load Balancing**

- **Gating Mechanism**: Top-K gating scores determine expert activation per token
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping computation
- **Dynamic Load Balancing**: Monitor per-expert load, adjust gating probabilities to prevent overloading

### **3. Communication Overlap and Scheduling**

- **Compute-Communication Interleaving**: Process current batch while transferring next batch
- **CUDA Streams**: Use separate streams for compute and communication
- **Pipeline Scheduling**: Immediate routing between layers, process partial batches as they arrive

### **4. Scalability Framework**

**Large EP Regime (EP ≥ 16):**
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU ensures full GPU utilization
- Communication costs masked by overlapped calculation

**Memory and Parallelism Integration:**
- Tensor parallelism (TP) within each expert when model exceeds GPU memory
- Data parallelism (DP) across MoE replicas for synchronized weight updates
- Hierarchical deployment: DP → EP → TP layering

## **Experiments**

### **Model Configuration**
- **Architecture**: 61-layer MoE (3 dense layers + 58 MoE layers)
- **Expert Type**: MLP-based experts
- **Token Dimension**: 7168
- **MHA**: 128 heads × 128 dimensions per head
- **MLP Hidden Size**: 2048
- **Precision**: BF16
- **Processing**: Variable batch size and sequence length

### **Hardware Environment**
- **GPUs**: H100 (adequate supply, no limits)
- **Single-card Computing Power**: 400TFlops
- **MFU**: 60%
- **VRAM Bandwidth**: 1.8TBps
- **Bandwidth Utilization**: 80%
- **Single-card Memory**: 64GB

### **Parallel Deployment Details**

**Proposed Cross-Node Expert Parallelism:**
- **GPU Allocation**: One GPU per expert per layer
- **Routing**: Dynamic routing to GPU holding corresponding expert
- **Transfer**: Asynchronous token batches to minimize idle time
- **Processing**: All experts per layer compute in parallel

**Performance Characteristics:**
- Maximized expert independence with dedicated GPU per expert
- Balanced load distribution across nodes
- Near-linear scaling through communication-computation overlap
- High GPU utilization (60% MFU) with 80% bandwidth utilization

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. Through asynchronous token routing, topology-aware placement, and overlapping computation with communication, we provide a scalable blueprint for high-performance MoE inference in GPU-rich environments like H100 clusters.