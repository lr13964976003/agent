## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **1. Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. However, scaling MoE across large GPU clusters introduces significant challenges in expert placement and parallelization. Traditional approaches assign multiple experts per GPU to reduce inter-node communication, creating computational bottlenecks that limit true expert parallelism. This work presents a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, achieving EP ≥ 16 for large-scale inference.

## **2. Background**

### **2.1 Mixture-of-Experts**
MoE models replace transformer FFN layers with multiple specialized experts, using a gating mechanism to activate subsets of experts per token, achieving sparse computation and improved efficiency.

### **2.2 Parallelism Strategies**
Scaling MoE involves combinations of data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Traditional implementations use moderate EP degrees with multiple experts per GPU, limiting network traffic but creating computational contention.

### **2.3 Large Expert Parallelism**
With advanced network interconnects (NVLink, InfiniBand, NVSwitch), communication costs become less dominant than compute concurrency gains. Large EP (≥16) distributes experts across devices with one per GPU, maximizing parallel execution while leveraging network capabilities.

### **2.4 Multi-Head Latent Attention**
MLA reduces attention memory overhead through low-dimensional latent representations for K/V, significantly reducing KV cache size compared to standard attention.

## **3. Methods**

### **3.1 Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity, and routing patterns
- **Assignment**: For E experts and G GPUs, ensure distinct assignment when E ≤ G, replicated placement when E > G

### **3.2 Routing and Load Balancing**
- **Gating**: Top-K expert selection with dynamic load balancing
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, real-time load monitoring
- **Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **3.3 Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Interleave expert computation with token transfers using CUDA streams/NCCL
- **Pipeline Scheduling**: Layer-to-layer immediate routing, partial batch processing for increased throughput
- **Asynchronous Operations**: Ensure data transfer never blocks GPU computation

### **3.4 Scalability**
- **Large EP Regime**: EP ≥ 256 in experiments, network bandwidth as primary limiting factor
- **Integration**: Compatible with tensor parallelism within experts and data parallelism across replicas

## **4. Experiments**

### **4.1 Setup**
- **Model**: 61-layer MoE (3 dense + 58 MoE layers)
- **Experts**: 256 per MoE layer, MLP with hidden size 18432
- **Attention**: MLA with 128 heads, 56 dimensions per head, total dimension 7168
- **Precision**: FP8
- **Hardware**: H100 GPUs (400TFlops, 64GB VRAM, 1.8TBps bandwidth)

### **4.2 Deployment**
- **GPUs Used**: 256 per MoE layer (14,848 total for all MoE layers)
- **Allocation**: One expert per GPU per layer
- **Performance**: All experts compute in parallel with minimal contention

### **4.3 Results**
- **Scalability**: Near-linear scaling with large EP
- **Utilization**: 60% MFU, 80% VRAM bandwidth utilization
- **Contention**: Minimal inter-expert interference
- **Load Balance**: Topology-aware placement prevents hotspots

## **5. Conclusion**
The proposed large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. This approach shifts the computational bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous token routing, topology-aware placement, and compute-communication overlap. The method provides a scalable blueprint for high-performance MoE inference in resource-rich environments like H100 clusters.