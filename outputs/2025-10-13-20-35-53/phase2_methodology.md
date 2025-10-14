# Phase 2: Methodology Extraction

## Method Overview
Our approach maximizes expert-level parallelism in large-scale MoE models through three key components:

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Mathematical constraint**: For E experts and G GPUs, ensure each expert assigned to distinct GPU if E ≤ G
- **Replication strategy**: If E > G, replicate experts to maximize independent expert concurrency while balancing memory
- **Resource isolation**: Each expert processes tokens without contention from other experts on same device

#### 1.2 Cross-Node Distribution
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency characteristics
  - GPU memory capacity per node
  - Expected token routing patterns
- **Optimization objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE routing**: Top-K gating scores determine expert activation per token
- **Input**: Each token produces gating scores across all experts
- **Output**: Select top-K experts for token processing

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to minimize network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving strategy**: Process current batch while transferring next batch
- **Technical implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Non-blocking**: Ensure data transfer doesn't block GPU computation

#### 3.2 Pipeline Scheduling
- **Multi-layer coordination**: Route token outputs immediately to next layer's experts
- **Partial batch processing**: Subsequent layer experts start processing as soon as partial batch arrives
- **Fine-grained pipeline**: Increase throughput and reduce expert idle time

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Primary limiting factor**: Network bandwidth (mitigated via topology-aware routing and token batching)
- **Compute focus**: One-expert-per-GPU ensures full GPU utilization

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Partition individual experts across GPUs when single GPU insufficient
- **Data Parallelism (DP)**: Apply across MoE network replicas for synchronized weight updates
- **Memory constraints**: Handle very large models exceeding single-GPU memory

### 5. Implementation Details

#### 5.1 Model Architecture Parameters
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Token dimension**: 8192
- **MLP hidden size**: 32768
- **Precision**: FP16

#### 5.2 Batch Configuration
- **Batch size**: 1024 sequences
- **Sequence length**: 10,000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens

#### 5.3 Multi-Head Attention Parameters
- **Number of heads**: 16
- **Head dimension**: 512
- **Total MHA dimension**: 8192 (matches token dimension)

### 6. Communication Patterns

#### 6.1 Token Routing Flow
1. **Input tokens** arrive at routing nodes
2. **Gating computation** determines expert destinations
3. **Token grouping** by destination expert
4. **Asynchronous transfer** to expert GPUs
5. **Expert computation** on assigned tokens
6. **Result aggregation** back to original sequence order

#### 6.2 Network Optimization
- **Topology awareness**: Map experts based on physical network topology
- **Bandwidth utilization**: Balance traffic across available links
- **Latency hiding**: Overlap communication with computation
- **Congestion avoidance**: Dynamic routing adjustments based on load