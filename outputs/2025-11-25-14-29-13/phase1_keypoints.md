# Phase 1: Key Points Extraction

## Paper Title: Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Core Problem
- Traditional MoE parallelization assigns multiple experts to the same GPU, creating computational bottlenecks
- Need to scale MoE models across large GPU clusters while maintaining efficiency

### Key Innovation
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- Prioritize distributing experts across nodes to maximize compute concurrency
- Shift focus from communication reduction to compute maximization

### Method Components
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic token routing with load balancing
3. **Communication Overlap**: Asynchronous token routing to overlap computation and communication

### Architecture Details
- 16-layer Mixture-of-Experts (MoE) model
- Each layer has 16 experts (MLP-based)
- Token dimension: 4096
- MLP hidden size: 16384
- Precision: BF16
- Batch size: 128 sequences of 10000 tokens each
- MHA: 32 heads, 128 dimensions per head

### Performance Gains
- **3.75× higher throughput** (450,000 vs 120,000 tokens/second)
- **3.8× lower latency** (2.2ms vs 8.3ms per token)
- Achieved through full GPU utilization with one expert per GPU

### Scalability Features
- Topology-aware expert placement
- Asynchronous token routing
- Fine-grained pipeline scheduling
- Integration with tensor and pipeline parallelism when needed