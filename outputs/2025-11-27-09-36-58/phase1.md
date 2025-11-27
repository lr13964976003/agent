## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as “large EP” in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

---

## **Key Points**

1. **Problem**: Traditional MoE parallelization colocates multiple experts per GPU, creating computational bottlenecks and limiting expert-level parallelism.

2. **Solution**: Large-scale cross-node expert parallelism ensuring at most one expert per GPU, with Expert Parallelism (EP) ≥ 16.

3. **Core Innovation**: Shifts bottleneck from compute contention to communication, leveraging HPC networking capabilities.

4. **Key Components**:
   - **Expert Placement**: One expert per GPU, topology-aware distribution across nodes
   - **Routing & Load Balancing**: Token batching, asynchronous routing, dynamic gating adjustment
   - **Communication Overlap**: Interleaved compute and communication using CUDA streams/NCCL

5. **Advantages**:
   - Maximized expert parallelism
   - Balanced load across nodes
   - Scalable communication overlap
   - Compatibility with large models (TP/DP integration)

---

## **Experimental Setup**

### Model Configuration
- **Architecture**: 61-layer MoE (first 3 layers dense, remaining MoE)
- **Precision**: BF16
- **Token Dimension**: 7168
- **MHA**: 128 heads, 128 dim per head
- **MLP Hidden Size**: 2048

### Hardware Configuration
- **GPUs**: Adequate H100 GPUs (one GPU per expert)
- **Single-card compute**: 400TFlops, 60% MFU
- **VRAM Bandwidth**: 1.8TBps, 80% utilization
- **Single-card memory**: 64GB

### Deployment Strategy
- **Parallel Strategy**: Cross-node expert parallelism with EP≥16
- **Per-GPU Allocation**: Exactly one expert per GPU per layer
- **Routing**: Dynamic token routing with asynchronous batch transfer