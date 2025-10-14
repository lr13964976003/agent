# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Concise Version)

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert-level parallelism as model and cluster sizes grow. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## 2. Background

### 2.1 Mixture-of-Experts in Large-Scale Models
MoE models replace FFN layers with multiple "experts," each specializing in different input patterns. A gating mechanism determines which subset of experts is activated for each token, leading to sparse computation and improved efficiency.

### 2.2 Parallelism Strategies for MoE
Scaling MoE models involves data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Standard implementations often use moderate EP degrees, placing multiple experts per GPU to limit communication. However, as network interconnects advance, communication cost becomes less dominant compared to gains from maximizing compute concurrency.

### 2.3 Large Expert Parallelism (Large EP)
We define large EP as configurations where EP ≥ 16. In such regimes, distributing experts across as many devices as possible—ideally one per GPU—minimizes resource contention and maximizes expert-level parallel execution.

## 3. Methods

### 3.1 Expert Placement Strategy

#### Single-Expert-Per-GPU Deployment
- Each GPU hosts at most one expert per layer
- If E ≤ G (experts ≤ GPUs): each expert assigned to distinct GPU
- If E > G: replicate experts to maximize concurrency while balancing memory
- Ensures experts process tokens without contention from other experts on same device

#### Cross-Node Distribution
Experts are distributed across nodes using topology-aware placement considering:
- Node-to-node bandwidth and latency
- GPU memory capacity per node
- Expected token routing patterns

### 3.2 Routing and Load Balancing

#### Gating Mechanism
Top-K gating scores determine expert activation, with tokens routed to experts based on gating network outputs.

#### Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### 3.3 Communication Overlap and Scheduling

#### Overlapping Compute and Communication
- Interleave expert computation and communication using CUDA streams
- While one batch processes, next batch transfers simultaneously
- Leverage NCCL/MPI for asynchronous communication

#### Pipeline Scheduling
In multi-layer MoE networks:
- Token outputs immediately routed to next layer's experts
- Experts start processing as soon as partial batch arrives
- Fine-grained pipeline increases throughput and reduces idle time

### 3.4 Scalability Considerations

#### Large EP Regime (EP ≥ 16)
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU policy ensures full GPU utilization

#### Memory and Model Parallelism Integration
- Experts can use tensor model parallelism within GPU if needed
- Data parallelism applied across MoE network replicas
- Maintains high expert-level parallelism while handling large models

## 4. Experiments

### 4.1 Experimental Setup
- **Model**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32,768
- **Hardware**: 16 H100 GPUs
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

### 4.2 Baseline Deployment (TP=8, PP=2)
- **GPUs**: 16 H100
- **Per-GPU**: 1/8 tensor-parallel shard for all layers, 2 pipeline stages, 8 experts per layer per GPU
- **Results**: 120,000 TPS, 8.3ms TPOT

### 4.3 Proposed Cross-Node Expert Parallelism
- **GPUs**: 16 H100 (one GPU per expert per layer)
- **Per-GPU**: Exactly one expert per layer
- **Results**: 450,000 TPS, 2.2ms TPOT

### 4.4 Performance Comparison
| Method | GPUs | Deployment | TPS | TPOT | Improvement |
|--------|------|------------|-----|------|-------------|
| Baseline | 16 | 8 experts/layer/GPU + TP | 120,000 | 8.3ms | 1.0× |
| Proposed | 16 | 1 expert/layer/GPU | 450,000 | 2.2ms | 3.75× TPS, 3.8× latency |

## 5. Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By ensuring EP ≥ 16 and using asynchronous token routing with topology-aware placement, we achieved ~3.75× higher throughput and ~3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

The method successfully shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth while achieving near-linear scaling in large MoE deployments.