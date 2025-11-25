## **Methodology Extraction - Phase 2**

### **Model Architecture**
- **Type**: 16-layer Mixture-of-Experts (MoE) transformer
- **Experts per layer**: 16 experts (MLP blocks)
- **Token dimension**: 4096
- **Hidden size of MLP**: 16384
- **Multi-head attention**: 32 heads, 128 dimensions per head
- **Precision**: BF16
- **Batch configuration**: 128 sequences, 10000 tokens per sequence

### **Expert Placement Strategy**

#### **Single-Expert-Per-GPU Deployment**
- **Principle**: Each GPU hosts at most one expert per layer
- **Assignment logic**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs maximizing concurrency
- **Memory isolation**: No intra-GPU expert contention

#### **Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

### **Routing and Load Balancing**

#### **Gating Mechanism**
- **Type**: Top-K gating scores per input token
- **K value**: 2 (typical for MoE)
- **Dynamic routing**: Tokens routed to expert-specific GPUs

#### **Token Sharding Across Nodes**
1. **Token Batching**: Group tokens by destination expert
2. **Asynchronous Routing**: Overlap token transfer with computation
3. **Load Balancing**: Dynamic gating probability adjustment

### **Communication Overlap and Scheduling**

#### **Overlapping Compute and Communication**
- **Mechanism**: CUDA streams/NCCL for asynchronous operations
- **Strategy**: Process batch N while transferring batch N+1
- **Synchronization**: Non-blocking communication primitives

#### **Pipeline Scheduling**
- **Multi-layer coordination**: Immediate routing between layers
- **Partial batch processing**: Start processing upon partial arrival
- **Fine-grained pipeline**: Expert-level scheduling granularity

### **Scalability Parameters**
- **Large EP regime**: EP ≥ 16
- **Primary constraint**: Network bandwidth
- **Secondary constraints**: 
  - Synchronization overhead
  - Load balancing accuracy
- **Integration**: Compatible with TP and DP for memory scaling

### **Hardware Specifications**
- **GPU**: H100 (unlimited availability)
- **Compute**: 400TFlops per GPU
- **Memory**: 64GB VRAM per GPU
- **Bandwidth**: 1.8TBps VRAM bandwidth
- **Utilization targets**: 60% MFU, 80% bandwidth utilization