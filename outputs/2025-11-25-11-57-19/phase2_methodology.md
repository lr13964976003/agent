# Phase 2: Methodology Extraction

## Methods Section - Detailed Technical Implementation

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Allocation Logic**:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize concurrency
- **Memory Optimization**: Balances memory usage during replication

#### 1.2 Cross-Node Distribution
- **Topology Awareness Factors**:
  - Node-to-node bandwidth measurements
  - Inter-node latency characteristics
  - GPU memory capacity per node
  - Historical token routing patterns
- **Optimization Objective**: Minimize maximum tokens sent across any single link
- **Placement Algorithm**: Ensures one-expert-per-GPU while maintaining load balance

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE Architecture**: Top-K gating scores determine expert activation
- **Input Token Routing**: Each token routed to top-K experts based on gating scores

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: 
  - Groups tokens by destination expert
  - Reduces network message count
  - Batch size optimization based on network topology
- **Asynchronous Routing**:
  - Token batches sent asynchronously
  - Expert computation overlaps with communication
  - Non-blocking send/receive operations
- **Dynamic Load Balancing**:
  - Monitors per-expert load in real-time
  - Adjusts gating probabilities dynamically
  - Prevents expert overloading and stragglers

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Dual Stream Architecture**:
  - Stream 1: Token processing on GPU
  - Stream 2: Token transfer between nodes
- **CUDA Streams**: Separate streams for compute and communication
- **NCCL/MPI**: Asynchronous communication libraries for non-blocking transfers

#### 3.2 Pipeline Scheduling
- **Multi-Layer Coordination**:
  - Token outputs immediately routed to next layer's experts
  - Partial batch processing (no full batch waiting)
  - Fine-grained pipeline increases throughput
- **Stall Prevention**: Ensures experts don't wait for full token batches

### 4. Large EP Regime (EP ≥ 16)

#### 4.1 Network Bandwidth Optimization
- **Primary Limiting Factor**: Network bandwidth in EP ≥ 16
- **Mitigation Strategies**:
  - Topology-aware routing
  - Token batching optimization
  - Communication-computation overlap

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within each expert on single GPU if needed
- **Data Parallelism (DP)**: Replicas of MoE network with synchronized updates
- **Combined Strategy**: DP × EP × TP for extreme-scale models

### 5. Implementation Details

#### 5.1 Hardware Requirements
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand, H100 NVSwitch)
- **GPUs**: Sufficient quantity to maintain one-expert-per-GPU constraint
- **Memory**: Adequate per-GPU memory for single expert storage

#### 5.2 Software Components
- **Communication Layer**: NCCL or MPI for cross-node communication
- **Scheduling Engine**: Dynamic token routing and load balancing
- **Monitoring System**: Real-time load tracking and gating adjustment