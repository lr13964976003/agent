## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation.

## **Methods**

###s a variant of transformer architectures where the feed-forward network (FFN) layers are replaced by multiple "experts," each trained to specialize in different input patterns. A gating mechanism determines which subset of experts is activated for each token, leading to sparse computation and improved efficiency in large-scale training.

### **Our Approach: Large-Scale Cross-Node Expert Parallelism**

Our method focuses on maximizing expert-level parallelism through three key components:

1. **Expert Placement Strategy**: Deploy at most one expert per GPU
2. **Routing and Load Balancing**: Ensure balanced input distribution
3. **Communication Overlap and Scheduling**: Minimize cross-node transfer impact

#### **Single-Expert-Per-GPU Deployment**
- **Constraint**: At most one expert per GPU
- **Allocation**: Each expert assigned to distinct GPU when possible
- **Benefit**: Eliminates intra-GPU expert contention

#### **Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns

#### **Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert
- **Asynchronous Routing**: Overlap communication with computation
- **Dynamic Load Balancing**: Adjust gating probabilities to prevent overloading

#### **Communication Overlap and Scheduling**
- **Interleaving strategy**: Process one batch while transferring next
- **Pipeline Scheduling**: Subsequent layers start with partial batches
- **Technical implementation**: CUDA streams or NCCL/MPI for asynchronous communication

## **Experiments**

###istical Setup**
- **Model**: 4-layer MoE, 16 experts per layer (MLP)
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens
- **Dimensions**: Token=8192, MLP hidden=32768
- **Hardware**: 16 H100 GPUs (inference-only)

### **Deployment Configurations**

#### **Baseline (TP=8, PP=2)**
- **GPUs**: 16 total
- **Parallelism**: TP=8, PP=2
- **Expert Placement**: 8 experts per layer per GPU (shared resources)
- **Processing**: Sequential pipeline stages

#### **Proposed Cross-Node Expert Parallelism**
- **GPUs**: 16 total
- **Expert Parallelism**: EP=16 (maximum)
- **Expert Placement**: 1 expert per layer per GPU (dedicated)
- **Processing**: All 16 experts compute in parallel

### **Results**

| Method | GPUs | Expert per GPU | TPS | TPOT (ms) |
|--------|------|----------------|-----|-----------|
| Baseline | 16 | 8 (shared) | 120,000 | 8.3 |
| Proposed | 16 | 1 (dedicated) | 450,000 | 2.2 |

**Performance Improvements:**
- **Throughput**: 3.75× higher (450K vs 120K TPS)
- **Latency**: 3.8× lower (2.2ms vs 8.3ms TPOT)

## **Conclusion**

We proposed a large-scale cross-node expert parallelism method that achieves **3.75× higher throughput** and **3.8× lower latency** by dedicating one expert per GPU and leveraging EP ≥ 16. This approach shifts the bottleneck from compute contention to efficiently managed communication, providing a scalable blueprint for high-performance MoE inference in HPC environments.