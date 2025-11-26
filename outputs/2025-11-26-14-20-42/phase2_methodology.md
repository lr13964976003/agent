# Phase 2: Methodology Extraction

## Core Methodology Overview
The proposed method maximizes expert-level parallelism in MoE models through three key components:

### 1. Expert Placement Strategy
**Single-Expert-Per-GPU Principle:**
- Each GPU hosts at most one expert per layer
- For E experts and G GPUs: if E ≤ G, each expert assigned to distinct GPU
- If E > G, experts are replicated to maximize independent computation
- Ensures zero intra-GPU expert contention

**Cross-Node Distribution Algorithm:**
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimizes maximum tokens sent across any single link
- Maintains one-expert-per-GPU constraint

### 2. Routing and Load Balancing Mechanism
**Gating Network:**
- Standard top-K gating scores determine expert activation per token
- Dynamic probability adjustment for load balancing

**Token Sharding Strategy:**
- **Token Batching:** Group tokens by destination expert to reduce network messages
- **Asynchronous Routing:** Send token batches asynchronously while overlapping computation
- **Load Monitoring:** Real-time per-expert load tracking
- **Dynamic Rebalancing:** Adjust gating probabilities to prevent expert overload

### 3. Communication Overlap and Scheduling
**Compute-Communication Overlap:**
- Interleave expert computation with cross-node token transfers
- While current batch processes, next batch transfers simultaneously
- CUDA streams and asynchronous libraries (NCCL/MPI) for non-blocking transfers

**Pipeline Scheduling:**
- Immediate token output routing to next MoE layer
- Partial batch processing (no waiting for full batch)
- Fine-grained pipeline reduces expert idle time

## Technical Implementation Details

### Memory Management
- Each expert fully utilizes its dedicated GPU memory
- No sharing between experts on same device
- TP integration within expert if model exceeds single-GPU capacity

### Network Optimization
- Large EP regime (≥16 experts per parallel group)
- Network bandwidth as primary optimization target
- Topology-aware routing minimizes cross-node traffic
- Token batching reduces message count

### Integration with Other Parallelism
- **Tensor Parallelism (TP):** Applied within individual experts if needed
- **Data Parallelism (DP):** Applied across MoE network replicas
- **Pipeline Parallelism (PP):** Coordinated with expert placement

## Performance Characteristics
### Compute Efficiency
- 100% expert compute utilization (no intra-GPU contention)
- MFU: 60% target utilization
- Single-card compute: 400TFlops (H100)

### Memory Specifications
- VRAM capacity: 64GB per GPU
- Bandwidth: 1.8TBps
- Utilization: 80%

### Communication Pattern
- Asynchronous token routing
- Batch-based message reduction
- Overlap with computation (60% MFU target)

## Model Architecture Integration
### Layer Configuration
- 61 total layers
- First 3 layers: dense (non-MoE)
- Remaining 58 layers: MoE with expert placement

### Dimension Specifications
- Token dimension: 7168
- MHA heads: 128, head dimension: 128
- MLP hidden size: 2048
- Precision: BF16

### Deployment Configuration
- One expert per GPU per layer
- Dynamic routing based on gating scores
- Variable batch and sequence length support
- Inference-only optimization