# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse activation. However, scaling MoE models across GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This maximizes concurrent computation and leverages modern HPC networking capabilities to sustain high bandwidth and low latency.

## **Methods**

### 3. **Methods**

### 3.1 **Expert Placement Strategy**

**Single-Expert-Per-GPU Deployment:**
- Each GPU hosts at most one expert per layer
- For E experts and G GPUs: ensure distinct GPU assignment when E ≤ G
- When E > G: replicate experts to maximize concurrency while balancing memory

**Cross-Node Distribution:**
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Objective: Minimize maximum tokens across any single link

### 3.2 **Routing and Load Balancing**

**Gating Mechanism:**
- Top-K routing (K=2 standard) with dynamic probability adjustment
- Real-time expert utilization monitoring

**Token Sharding:**
1. Token batching by destination expert to reduce network messages
2. Asynchronous routing overlapping with computation
3. Dynamic load balancing to prevent expert overload

### 3.3 **Communication Overlap and Scheduling**

**Compute-Communication Overlap:**
- CUDA streams for separate compute/communication
- NCCL/MPI for asynchronous transfers
- Double/triple buffering for seamless overlap

**Pipeline Scheduling:**
- Layer-wise pipeline: 16 experts per layer on 16 GPUs (parallel)
- Fine-grained pipeline across 16 MoE layers
- Immediate token forwarding between layers

### 3.4 **Large EP Regime (EP ≥ 16)**

**Network Optimization:**
- Bandwidth as primary limiting factor for EP ≥ 16
- Topology-aware routing and token batching mitigation
- Compute saturation through expert independence

**Memory Integration:**
- Tensor parallelism (TP) within expert if needed
- Data parallelism (DP) across model replicas
- Expert parameters + activations within single GPU memory

## **Experiments**

### 4.1 **Experimental Setup**

**Model Configuration:**
- Architecture: 16-layer MoE transformer
- Experts: 16 experts per layer (MLP-based)
- Precision: BF16
- Dimensions:
  - Token: 4096
  - MLP Hidden: 16384
  - MHA: 32 heads × 128 = 4096

**Runtime Configuration:**
- Batch: 128 sequences
- Sequence Length: 10,000 tokens
- Setting: Inference-only
- Hardware: H100 GPUs

**Metrics:**
- TPS (Tokens per Second)
- TPOT (Time per Output Token)

### 4.2 **Baseline vs Proposed**

**Baseline (TP=8, PP=2):**
- GPUs: 16 H100
- Deployment: 8 experts per GPU + 1/8 tensor shard
- Performance: TPS = 120,000, TPOT = 8.3ms

**Proposed (Large EP):**
- GPUs: 16 H100
- Deployment: 1 expert per GPU per layer
- Performance: TPS = 450,000, TPOT = 2.2ms
- Improvement: 3.75× throughput, 3.77× latency reduction

### 4.3 **Deployment Details**

**GPU Mapping:**
```
Proposed Configuration:
Layer 1: Expert 1-16 → GPU 1-16 (parallel)
Layer 2: Expert 1-16 → GPU 1-16 (parallel)
...
Layer 16: Expert 1-16 → GPU 1-16 (parallel)
```

**Parallel Strategy:**
- Expert Parallelism (EP) = 16
- Tensor Parallelism (TP) = 1 (within expert)
- Pipeline Parallelism (PP) = 1 (across layers)

## **Conclusion**

Our large-scale cross-node expert parallelism method achieves 3.75× throughput improvement and 3.77× latency reduction by maximizing expert-level parallelism with one-expert-per-GPU deployment. This approach provides a scalable blueprint for high-performance MoE inference in HPC environments.

## **Appendix: Deployment Configuration Details**

**Hardware Requirements:**
- 16 H100 GPUs minimum
- High-bandwidth interconnects (NVLink/NVSwitch/InfiniBand)
- Homogeneous GPU memory (80GB+ per GPU recommended)

**Software Stack:**
- CUDA streams for overlap
- NCCL for collective communication
- Custom routing layer for token distribution
- Dynamic load balancing algorithms

**Memory Specifications:**
- Expert parameters: ~1GB per expert (16 experts × 1GB = 16GB total)
- Activations: Batch size dependent (128 × 10000 × 4096 × 2 bytes ≈ 10GB)
- Total per GPU: ~30GB (including buffers and overhead)