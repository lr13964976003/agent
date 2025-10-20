# Phase 1: Key Points Extraction - Large-Scale Cross-Node Expert Parallelism

## Core Problem
Traditional MoE parallelization colocates multiple experts on the same GPU, creating computational bottlenecks that limit expert parallelism as model/cluster sizes grow.

## Proposed Solution
Deploy at most one expert per GPU across nodes to maximize expert-level parallelism (Large EP: EP ≥ 16), trading communication overhead for compute concurrency.

## Key Technical Innovations

### 1. Expert Placement Strategy
- **One-expert-per-GPU principle**: Each GPU hosts at most one expert per layer
- **Cross-node distribution**: Topology-aware placement considering node bandwidth, latency, and GPU memory capacity
- **Optimization objective**: Minimize max(tokens sent across any single link)

### 2. Routing and Load Balancing
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous routing**: Send batches asynchronously to overlap computation
- **Dynamic load balancing**: Adjust gating probabilities based on expert load using equation: load_e = (tokens routed to e)/(expert capacity)

### 3. Communication Overlap
- **CUDA streams**: Multiple streams for asynchronous communication
- **NCCL operations**: Send/recv patterns for cross-node transfers
- **Pipeline scheduling**: Tokens immediately routed to next layer's experts without waiting

### 4. Scalability Framework
- **Large EP regime**: EP ≥ 16 ensures network bandwidth becomes primary limiting factor
- **Integration**: Compatible with tensor parallelism (TP) within experts and data parallelism (DP) across replicas
- **Memory management**: Each expert can be partitioned using TP if exceeding single-GPU memory

## Performance Claims
- **3.75× throughput improvement**: 120k → 450k TPS
- **3.8× latency reduction**: 8.3ms → 2.2ms TPOT
- **Near-linear scaling**: Validated for EP ≥ 16 regime

## Critical Technical Specifications
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Dimensions**: Input=8192, Hidden=32768, MHA=16×512=8192
- **Precision**: FP16
- **Batch**: 1024 sequences × 10,000 tokens
- **Hardware**: 16 H100 GPUs with NVLink + InfiniBand

## Deployment Constraints
- Maximum one expert per GPU
- Topology-aware placement algorithm required
- Asynchronous token routing with load balancing
- Full GPU utilization via dedicated expert resources