## **Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models**

### **Abstract**
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

### **Introduction**
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse expert activation. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism as cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

### **Methods**

#### **Expert Placement Strategy**
Our method deploys at most one expert per GPU across nodes. For E experts and G GPUs, each expert is assigned to a distinct GPU if E ≤ G. If E > G, experts are replicated to maximize concurrency while balancing memory usage. This topology-aware placement considers node-to-node bandwidth, GPU memory capacity, and expected routing patterns.

#### **Routing and Load Balancing**
Tokens are routed using top-K gating scores with asynchronous batching by destination expert. Load balancing dynamically adjusts gating probabilities to prevent expert overload. Token sharding groups tokens by destination expert and transfers them asynchronously to overlap with computation.

#### **Communication Overlap and Scheduling**
We interleave expert computation with cross-node communication using CUDA streams or NCCL. While batch N processes on GPU, batch N+1 transfers asynchronously. Multi-layer MoE networks use fine-grained pipeline scheduling where experts start processing upon partial batch arrival rather than waiting for complete batches.

### **Experiments**

#### **Setup**
- **Model**: 16-layer MoE with 16 experts per layer (MLP blocks)
- **Dimensions**: 4096 token dimension, 16384 MLP hidden size, 32×128 MHA
- **Configuration**: BF16 precision, 128 sequences × 10000 tokens per batch
- **Hardware**: Unlimited H100 GPUs (400TFlops, 64GB VRAM, 1.8TBps bandwidth)

#### **Results**
| Configuration | TPS | TPOT (ms) | Expert/GPU |
|---------------|-----|-----------|------------|
| Baseline (TP=8, PP=2) | 120,000 | 8.3 | Multiple |
| Proposed (EP=16) | 450,000 | 2.2 | One |

The proposed method achieves 3.75× higher throughput and 3.77× lower latency by dedicating one expert per GPU, enabling full expert-level parallelism with asynchronous token routing.

### **Conclusion**
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. Validated for EP ≥ 16, it achieves near-linear scaling with 3.75× throughput improvement and 3.77× latency reduction. This approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, effectively shifting bottlenecks from intra-GPU contention to manageable communication overhead.