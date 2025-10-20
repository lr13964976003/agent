# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Refined Version

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach achieves 3.75× higher throughput (450k vs 120k TPS) and 3.8× lower latency (2.2ms vs 8.3ms TPOT) compared to traditional methods, particularly effective in high-performance computing environments with 16+ GPUs.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency through sparse computation. Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it creates computational bottlenecks and limits expert parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities (400 GB/s NVLink + 200 GB/s InfiniBand) to sustain high bandwidth and low latency across nodes.

## Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
**Mathematical Formulation:**
- Let E = number of experts per layer (E = 16)
- Let G = number of available GPUs (G = 16)
- Placement constraint: ∀g ∈ [1,G], expert_count(g) ≤ 1

**Topology-Aware Placement Algorithm:**
```
function place_experts(E, G, topology):
    # topology contains node_bandwidth[node_i][node_j] and gpu_memory[node]
    placement = {}
    
    # Step 1: Create expert-GPU mapping
    for expert_id in range(E):
        gpu_id = expert_id % G
        placement[expert_id] = gpu_id
    
    # Step 2: Topology-aware optimization
    # Objective: minimize max(tokens sent across any single link)
    link_loads = calculate_link_loads(placement, expected_token_routing)
    
    while max(link_loads) > threshold:
        expert_to_swap = find_expert_for_swap(link_loads)
        new_placement = swap_experts(placement, expert_to_swap)
        if max(calculate_link_loads(new_placement)) < max(link_loads):
            placement = new_placement
            link_loads = calculate_link_loads(placement)
    
    return placement
```

#### 1.2 Cross-Node Distribution
**Network Topology Parameters:**
- NVLink bandwidth: 400 GB/s bidirectional
- InfiniBand bandwidth: 200 GB/s per link
- Intra-node latency: 5-10 μs
- Inter-node latency: 1-5 μs

**Memory Requirements per GPU:**
- Expert parameters: 32,768 × 8,192 × 2 bytes (FP16) = 537 MB
- Expert output: 8,192 × 8,192 × 2 bytes = 131 MB
- Activation buffers: 1024 × 10,000 × 8,192 × 2 bytes = 160 GB (shared)
- Total per GPU: ~668 MB per expert + shared buffers

### 2. Routing and Load Balancing

#### 2.1 Gating and Token Sharding
**Mathematical Formulation:**
```
for each token t in batch:
    # Top-K gating scores
    g_i(t) = softmax(W_gate · t)[i] for i ∈ {1,...,E}
    
    # Dynamic load balancing
    load_e = (tokens routed to expert e) / expert_capacity
    adjustment_factor = max(0.1, 1.0 - load_e * 0.5)
    adjusted_g_i(t) = g_i(t) * adjustment_factor
```

#### 2.2 Asynchronous Communication
**CUDA Stream Configuration:**
- Stream 0: Compute (expert computation)
- Stream 1: Communication (token sending)
- Stream 2: Communication (token receiving)

**NCCL Operations:**
```
# Send tokens to expert on GPU i
ncclSend(token_batch, count, ncclFloat16, dest_gpu, ncclComm, compute_stream)

# Receive tokens from other GPUs
ncclRecv(token_batch, count, ncclFloat16, src_gpu, ncclComm, compute_stream)
```

### 3. Communication Overlap and Scheduling

#### 3.1 Pipeline Scheduling Details
**Layer-wise Processing Timeline:**
```
Communication Timeline per Layer:
├── Token distribution: 26.2 ms (NVLink)
├── Expert computation: 5.0 ms per expert
├── Result aggregation: 26.2 ms (NVLink)
└── Overlap efficiency: 80% communication hidden
```

#### 3.2 Load Balancing Parameters
- **Update frequency**: Every 100 tokens
- **Threshold for rebalancing**: When load_e > 1.2 × average_load
- **Gating probability adjustment**: ±10% per update cycle
- **Expert capacity**: 1.2 × average_load

### 4. Scalability Framework

#### 4.1 Large EP Regime (EP ≥ 16)
**Network Bandwidth Calculations:**
- Total communication per layer: 1024 × 10,000 × 8,192 × 2 bytes = 167.8 GB
- With 16 experts: 167.8 GB ÷ 16 = 10.49 GB per expert per layer
- Communication time: 10.49 GB ÷ 400 GB/s = 26.2 ms

#### 4.2 Memory and Model Parallelism Integration
**Tensor Parallelism within Experts:**
- Column-parallel first linear: Split hidden dimension (32,768) across 2 GPUs
- Row-parallel second linear: Split hidden dimension (32,768) across 2 GPUs

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer MoE with 16 experts per layer
- **Expert Type**: MLP with hidden size 32,768
- **Precision**: FP16
- **Input**: 1,024 sequences × 10,000 tokens × 8,192 dimensions
- **MHA**: 16 heads × 512 dimensions per head = 8,192 total
- **Hardware**: 16 H100 GPUs with NVLink + InfiniBand interconnects

### Evaluation Metrics
- **TPS**: Tokens per second (throughput)
- **TPOT**: Time per output token (latency in ms)

## Experimental Results

### Performance Comparison
| Method | GPUs | Deployment | TPS | TPOT | Improvement |
|--------|------|------------|-----|------|-------------|
| Baseline (TP=8, PP=2) | 16 | 8 experts per layer per GPU + TP shards | 120,000 | 8.3ms | 1.0× |
| Proposed (EP=16) | 16 | 1 expert per layer per GPU | 450,000 | 2.2ms | 3.75× TPS, 3.8× latency |

### Detailed Analysis
**Baseline Configuration:**
- 2 pipeline stages × 8 tensor-parallel GPUs = 16 total GPUs
- Each GPU: 1/8 tensor shard for 8 experts per layer
- Shared compute resources among colocated experts
- Resource contention limiting GPU utilization to 60-70%

**Proposed Cross-Node Expert Parallelism:**
- Expert Parallelism EP=16 across 16 GPUs
- Each GPU: exactly one expert per layer (16 experts total)
- All experts compute in parallel with dedicated resources
- GPU utilization: 95-98%
- Asynchronous token routing with topology-aware placement
- Communication overlap achieving 80% efficiency

### Scalability Results
- **Throughput scaling**: Linear scaling validated for EP=16
- **Memory efficiency**: Memory per expert decreases with EP
- **Network utilization**: NVLink bandwidth not saturated at EP=16
- **Compute efficiency**: Maximum GPU utilization achieved

## Conclusion

The large-scale cross-node expert parallelism method achieves significant performance improvements by maximizing expert-level parallelism through one-expert-per-GPU deployment. This approach shifts optimization from communication reduction to compute concurrency, demonstrating ~3.75× higher throughput and ~3.8× lower latency compared to traditional methods in HPC environments with 16+ GPUs. The method provides a scalable blueprint for future high-performance MoE inference, with validated linear scaling characteristics and full GPU utilization.