# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points Summary

### 1. Core Innovation
- **Large Expert Parallelism (EP ≥ 16)**: Distributes experts across nodes with at most one expert per GPU
- **Cross-node deployment**: Maximizes compute concurrency vs traditional colocation approaches
- **Resource exploitation**: Fully utilizes distributed GPU resources

### 2. Technical Approach
- **Expert Placement**: One expert per GPU principle
- **Topology-aware routing**: Considers bandwidth, latency, and memory capacity
- **Asynchronous communication**: Overlaps computation and communication
- **Load balancing**: Dynamic gating probability adjustment

### 3. Performance Gains
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2 vs 8.3 ms TPOT)
- **Near-linear scaling** in large EP regime

### 4. Model Configuration
- **Architecture**: 4-layer MoE, 16 experts per layer
- **Precision**: FP16
- **Batch**: 1024 sequences, 10,000 tokens per sequence
- **Dimensions**: 8192 token dimension, 32768 MLP hidden size
- **MHA**: 16 heads, 512 dimension per head

### 5. Deployment Comparison
- **Baseline**: TP=8, PP=2, 16 GPUs, 8 experts per GPU
- **Proposed**: EP=16, 16 GPUs, 1 expert per GPU
- **Environment**: H100 GPU cluster, inference-only setting

### 6. Critical Constraints
- **Cross-node communication**: Must be managed via asynchronous routing
- **Memory limits**: Each GPU hosts exactly one expert
- **Network topology**: Must be considered for optimal placement
- **Load balancing**: Prevents expert overloading

### 7. Scalability Features
- **Integrates with**: Tensor parallelism (TP) and Data parallelism (DP)
- **HPC optimized**: Designed for high-bandwidth, low-latency networks
- **Future extensibility**: Training scenario adaptation potential