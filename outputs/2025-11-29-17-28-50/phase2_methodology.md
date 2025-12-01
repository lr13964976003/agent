# Phase 2: Methodology Extraction

## Expert Placement Strategy

### Single-Expert-Per-GPU Deployment
- **Core Principle**: Deploy at most one expert per GPU
- **Mathematical Constraint**: For E experts and G GPUs, ensure each expert assigned to distinct GPU if E ≤ G
- **Replication Strategy**: If E > G, replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU expert contention, fully utilizes GPU compute units

### Cross-Node Distribution
- **Topology-Aware Placement**: Considers node-to-node bandwidth, latency, GPU memory capacity, expected token routing patterns
- **Optimization Goal**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle
- **Load Distribution**: Prevents hotspotting on any single node

## Routing and Load Balancing

### Gating Mechanism
- **Standard Approach**: Uses top-K gating scores to determine expert activation per input token
- **Implementation**: Follows standard MoE architectures for token routing decisions

### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## Communication Overlap and Scheduling

### Overlapping Compute and Communication
- **Interleaving Strategy**: Process one token batch while transferring next batch simultaneously
- **Technical Implementation**: CUDA streams or asynchronous libraries (NCCL/MPI) to prevent blocking
- **Benefit**: Data transfer does not block GPU computation

### Pipeline Scheduling
- **Multi-layer Coordination**: Token outputs from previous layer immediately routed to next layer's experts
- **Partial Batch Processing**: Subsequent layers start processing as soon as partial batches arrive
- **Throughput Impact**: Increases throughput and reduces expert idle time

## Large EP Regime (EP ≥ 16)

### Network Bandwidth Considerations
- **Primary Limiting Factor**: Network bandwidth becomes bottleneck in large EP setups
- **Mitigation Strategies**: Topology-aware routing and token batching
- **Compute Utilization**: One-expert-per-GPU ensures all GPUs fully utilized for compute

### Memory and Model Parallelism Integration
- **Tensor Parallelism**: Each expert can be partitioned using TP within GPU if necessary
- **Data Parallelism**: Applied across MoE network replicas for synchronized weight updates
- **Scalability**: Maintains high expert-level parallelism while handling large models

## Technical Specifications

### Model Configuration
- **Architecture**: 16-layer MoE, each expert is MLP
- **Precision**: BF16
- **Batch Size**: 128 sequences
- **Sequence Length**: 10,000 tokens
- **Token Dimension**: 4096
- **MHA Configuration**: 32 heads, 128 dimensions per head
- **MLP Hidden Size**: 16384

### Performance Metrics
- **Throughput Metric**: TPS (Tokens per Second)
- **Latency Metric**: TPOT (Time per Output Token)
- **Baseline Comparison**: TP=8, PP=2 configuration vs proposed cross-node expert parallelism