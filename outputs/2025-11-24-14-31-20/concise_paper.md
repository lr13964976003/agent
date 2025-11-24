# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Concise Version)

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures scale large language models while maintaining computational efficiency by activating only a subset of experts per input token. Traditional MoE implementations colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert parallelism. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This maximizes compute concurrency by leveraging modern HPC networking capabilities.

## Background

### Mixture-of-Experts Models
MoE models replace transformer FFN layers with multiple experts, each specializing in different input patterns. A gating mechanism activates a subset of experts per token, creating sparse computation for improved efficiency in large-scale training.

### Parallelism Strategies for MoE
Standard MoE scaling combines data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Traditional implementations use moderate EP degrees with multiple experts per GPU to limit communication. However, as network interconnects advance (NVLink, InfiniBand, H100-class NVSwitch), communication costs become less dominant than compute concurrency gains.

### Large Expert Parallelism (Large EP)
We define large EP as configurations where EP ≥ 16. In this regime, distributing experts across as many devices as possible (one per GPU) minimizes resource contention and maximizes expert-level parallel execution, shifting the optimization focus from reducing communication to maximizing compute concurrency.

## Methods

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Deployment rule**: For E experts and G GPUs, each expert assigned to distinct GPU if E ≤ G
- **Memory handling**: If E > G, experts replicated across GPUs to maximize concurrency while balancing memory

#### 1.2 Cross-Node Distribution
Topology-aware placement considering:
- Node-to-node bandwidth and latency
- GPU memory capacity per node
- Expected token routing patterns

### 2. Routing and Load Balancing

#### 2.1 Token Sharding Process
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Overlap
- **Mechanism**: Interleave expert computation with communication using CUDA streams and NCCL/MPI
- **Pipeline**: Current batch processes while next batch transfers in parallel

#### 3.2 Multi-layer Pipeline Scheduling
- Token outputs immediately routed to next layer's experts
- Subsequent layer experts process partial batches as they arrive

### 4. Scalability Considerations
- **Large EP regime optimization**: EP ≥ 16 with network bandwidth as primary limiting factor
- **Memory integration**: TP within GPU for memory constraints, DP across replicas for synchronized updates

## Experiments

### 1. Experimental Setup
- **Model**: 16-layer MoE, 16 experts per layer (MLP experts)
- **Precision**: BF16
- **Batch**: 128 sequences, 10,000 tokens per sequence
- **Dimensions**: Token dimension 4096, MLP hidden size 16384, MHA: 32 heads × 128 dim = 4096
- **Environment**: Inference-only on H100 GPUs
- **Metrics**: TPS (Tokens/Second), TPOT (Time per Output Token)

### 2. Configurations
| Method | GPUs | Configuration | TPS | TPOT |
|--------|------|---------------|-----|------|
| Baseline | 16 H100 | TP=8, PP=2, 8 experts/GPU | 120,000 | 8.3ms |
| Proposed | 16 H100 | EP=16, 1 expert/GPU | 450,000 | 2.2ms |

### 3. Results
- **Throughput improvement**: 3.75× higher (450k vs 120k TPS)
- **Latency reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling in large EP regime with sufficient network bandwidth

## Conclusion
Our large-scale cross-node expert parallelism method achieves 3.75× higher throughput and 3.8× lower latency by deploying one expert per GPU with EP ≥ 16. This approach provides a scalable blueprint for high-performance MoE inference in HPC environments, shifting optimization from communication reduction to compute concurrency maximization.

## Technical Specifications for Deployment

### Model Architecture Summary
- **Layers**: 16 MoE layers
- **Experts per layer**: 16 experts
- **Expert type**: MLP with hidden size 16384
- **Token dimension**: 4096
- **Precision**: BF16
- **Sequence length**: 10,000 tokens
- **Batch size**: 128 sequences

### Deployment Requirements
- **Minimum GPUs**: 16 (for EP=16 configuration)
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch
- **Parallel strategy**: Expert Parallelism (EP) degree 16
- **Expert placement**: One expert per GPU per layer