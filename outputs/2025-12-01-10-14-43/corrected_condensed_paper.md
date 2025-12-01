# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **1. Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism.

Our cross-node expert parallelism method distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts optimization from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **2. Methods**

### **2.1 Expert Placement Strategy**

**Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert per layer. For E experts and G GPUs:
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated to maximize concurrency

**Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and expected token routing patterns to minimize cross-link traffic while maintaining the one-expert-per-GPU principle.

### **2.2 Routing and Load Balancing**

**Token Sharding Strategy**:
- **Token Batching**: Group tokens by destination expert, reducing network messages from O(B×S×K) to O(E)
- **Asynchronous Routing**: Non-blocking token transfer overlapping computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **2.3 Communication Overlap and Scheduling**

**Compute-Communication Overlap**: Interleave expert computation with token transfers using CUDA streams and asynchronous communication libraries (NCCL/MPI).

**Pipeline Scheduling**: Multi-layer MoE networks immediately route token outputs, enabling experts to start processing partial batches rather than waiting for complete batches.

## **3. Experiments**

### **3.1 Experimental Setup**
- **Model**: 16-layer MoE, 64 experts per layer
- **Precision**: FP8
- **Batch**: 128 sequences, 128 tokens per sequence
- **Dimensions**: Token dim=1024, MHA=16 heads×64 dim, MoE hidden=2048

### **3.2 Deployment Configurations**

| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | TP shards, colocated experts | 120,000 | 8.3ms |
| **Proposed Cross-Node EP** | 16 | **1 expert per GPU** | **450,000** | **2.2ms** |

**Results**: 3.75× higher throughput and 3.8× lower latency achieved by enabling all 16 experts per layer to compute in parallel without intra-GPU contention.

## **4. Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. Achieving ~3.75× higher throughput and ~3.8× lower latency compared to baseline configurations, this approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments. The method effectively shifts the computational bottleneck from intra-GPU contention to manageable communication overhead through asynchronous routing and topology-aware placement.