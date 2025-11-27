# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Condensed Version

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

---

## **Key Technical Approach**

### **Problem Statement**
Traditional MoE implementations colocate multiple experts on a single GPU to reduce communication overhead, creating computational bottlenecks and limiting true expert parallelism as model and cluster sizes grow.

### **Proposed Solution**
Deploy **at most one expert per GPU** across nodes, pushing Expert Parallelism (EP) to 16 or more experts per parallel group, shifting the bottleneck from computation to communication which is mitigated through sophisticated scheduling.

### **Method Components**

#### **1. Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Each GPU hosts exactly one expert per layer
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity
- **Scalability**: When E ≤ G, distinct assignment; when E > G, expert replication with concurrency maximization

#### **2. Routing & Load Balancing**
- **Dynamic Gating**: Top-K selection with probability adjustment to prevent overloading
- **Token Batching**: Group tokens by destination expert to minimize network messages
- **Asynchronous Routing**: Overlap token transfer with computation using CUDA streams/NCCL

#### **3. Communication Overlap**
- **Compute-Communication Interleaving**: Process current batch while transferring next
- **Pipeline Scheduling**: Route outputs immediately to next layer, process partial batches
- **Fine-Grained Synchronization**: Minimize idle time through careful scheduling

### **Model Architecture**
- **Layers**: 61 total (3 dense + 58 MoE)
- **Expert Type**: MLP-based with 18432 hidden dimensions
- **Token Dimensions**: 7168 per token
- **Attention**: Multi-Head Latent Attention (MLA) with 128 heads × 56 dimensions
- **Precision**: BF16

### **Hardware Configuration**
- **GPUs**: H100 cluster with "adequate resources"
- **Compute**: 400 TFlops per card at 60% MFU
- **Memory**: 64GB VRAM, 1.8TBps bandwidth at 80% utilization
- **Network**: HPC-class interconnects (NVLink/InfiniBand)

### **Deployment Configuration**
- **GPU Allocation**: Number of experts × MoE layers (exact count depends on expert configuration)
- **Parallel Strategy**: Large-scale expert parallelism (EP ≥ 16)
- **Communication**: Asynchronous token routing with topology optimization
- **Resource Utilization**: Full GPU compute utilization, communication overlapped with computation

### **Performance Benefits**
- **Maximized Expert Parallelism**: All experts compute simultaneously without contention
- **Scalability**: Near-linear scaling in large GPU clusters
- **Latency Reduction**: Elimination of intra-GPU expert competition
- **Throughput Optimization**: Full utilization of distributed compute resources

### **Conclusion**
This large-scale cross-node expert parallelism method provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, achieving maximum compute concurrency through careful expert placement and communication overlapping.