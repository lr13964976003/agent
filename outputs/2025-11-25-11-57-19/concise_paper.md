# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Core Problem and Solution**

Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism. Our solution distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## **Methodology**

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Allocation Logic**: E ≤ G → distinct GPUs; E > G → replicated experts for concurrency

### 2. Routing and Load Balancing
- **Gating Mechanism**: Top-K expert selection per token
- **Token Batching**: Groups tokens by destination expert
- **Asynchronous Routing**: Overlaps communication with computation
- **Dynamic Load Balancing**: Real-time gating probability adjustment

### 3. Communication Overlap and Scheduling
- **Dual Stream Architecture**: Separate compute and communication streams
- **Pipeline Scheduling**: Immediate routing between layers, partial batch processing
- **Overlap Strategy**: CUDA streams/NCCL for non-blocking transfers

### 4. Large EP Regime (EP ≥ 16)
- **Network Optimization**: Topology-aware routing and token batching
- **Resource Integration**: Combines with TP within experts and DP across replicas

## **Experiments**

### Model Configuration
- **Architecture**: 16-layer MoE, 16 experts/layer
- **Precision**: BF16
- **Input**: 128 sequences × 10,000 tokens, 4096-dimensional
- **MHA**: 32 heads × 128 dimensions
- **MLP**: 16,384 hidden dimension

### Performance Comparison

| Method | Parallelism | GPU Allocation | TPS (Tokens/s) | TPOT (ms) |
|--------|-------------|----------------|----------------|-----------|
| Baseline | TP=8, PP=2 | Multiple experts/GPU | 120,000 | 8.3 |
| Proposed | EP=16 | One expert/GPU/layer | 450,000 | 2.2 |

### Results
- **3.75× throughput improvement** (450k vs 120k TPS)
- **3.8× latency reduction** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 16+ GPUs per expert layer

## **Conclusion**
Our large-scale cross-node expert parallelism method maximizes MoE performance by dedicating one expert per GPU, achieving significant throughput and latency improvements in HPC environments with EP ≥ 16.