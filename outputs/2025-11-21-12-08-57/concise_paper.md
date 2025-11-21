# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models: Concise Version

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Key Problem and Solution**

Traditional MoE implementations place multiple experts on the same GPU to reduce communication overhead, creating computational bottlenecks and limiting expert-level parallelism. Our solution prioritizes distributing experts across nodes such that each GPU hosts at most one expert, pushing EP ≥ 16 to maximize concurrent computation.

## **Methodology**

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert per layer
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Memory Optimization**: Expert weights fit within single GPU (128 MB per expert)

### 2. Routing and Load Balancing
- **Gating Mechanism**: Top-2 expert selection per token
- **Token Batching**: Group tokens by destination expert
- **Dynamic Load Balancing**: Monitor and adjust gating probabilities to prevent overloading

### 3. Communication Overlap
- **Asynchronous Routing**: Non-blocking token transfers
- **CUDA Streams**: Separate streams for compute and communication
- **Pipeline Scheduling**: Token-level granularity for maximum overlap

## **Experimental Setup**

### Model Configuration
- **Architecture**: 16-layer MoE with 16 experts per layer
- **Expert Type**: MLP with hidden dimension 16384
- **Precision**: BF16
- **Token Dimension**: 4096
- **MHA**: 32 heads × 128 dimensions per head

### Input Configuration
- **Batch**: 128 sequences × 10,000 tokens per sequence
- **Total Tokens**: 1,280,000 tokens per batch

### Hardware
- **GPUs**: 16 × H100 (80 GB HBM3 each)
- **Interconnect**: NVLink + NVSwitch + InfiniBand

## **Results**

| Method | GPUs | Configuration | TPS | TPOT |
|--------|------|---------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 8 experts/GPU + TP shards | 120,000 | 8.3 ms |
| Proposed (EP=16) | 16 | 1 expert/GPU | 450,000 | 2.2 ms |

**Improvements**: 3.75× higher throughput, 3.77× lower latency

## **Conclusion**
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by ensuring one expert per GPU, achieving near-linear scaling in the large EP regime (EP ≥ 16). This approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.