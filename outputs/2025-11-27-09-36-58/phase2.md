## **Detailed Methodology**

### **1. Expert Placement Strategy**

#### **1.1 Single-Expert-Per-GPU Deployment**
- **Constraint**: At most one expert per GPU
- **Assignment Rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs while maximizing independent expert concurrency
- **Memory Consideration**: 64GB VRAM per GPU for expert storage and intermediate activations

#### **1.2 Cross-Node Distribution Algorithm**
- **Topology Parameters**:
  - Node-to-node bandwidth: 1.8TBps (80% utilization)
  - GPU memory capacity: 64GB per GPU
  - Expected token routing patterns: Dynamic based on gating scores

- **Placement Algorithm**:
  1. Calculate optimal expert distribution across nodes
  2. Minimize maximum tokens per link
  3. Ensure one-expert-per-GPU constraint
  4. Balance memory usage across nodes

### **2. Routing and Load Balancing**

#### **2.1 Gating Mechanism**
- **Top-K Selection**: K=2 (standard MoE practice)
- **Gating Network**: Softmax over expert scores
- **Input**: Token embeddings (dimension 7168)
- **Output**: Expert indices and weights

#### **2.2 Token Sharding Process**
1. **Token Batching**:
   - Group tokens by destination expert
   - Batch size optimization based on network latency
   - Token dimension: 7168 per token

2. **Asynchronous Routing**:
   - Send token batches asynchronously
   - Overlap with expert computation
   - Use NCCL/MPI for communication

3. **Load Balancing**:
   - Monitor per-expert load in real-time
   - Adjust gating probabilities dynamically
   - Prevent expert overloading

### **3. Communication Overlap and Scheduling**

#### **3.1 Compute-Communication Overlap**
- **Mechanism**: 
  - CUDA streams for parallel compute/communication
  - NCCL for inter-node communication
  - Batch processing while transferring next batch

- **Timing Parameters**:
  - Compute time per expert: ~400TFlops * 60% MFU
  - Communication latency: 1.8TBps * 80% utilization
  - Overlap efficiency: >90%

#### **3.2 Pipeline Scheduling**
- **Layer-wise Processing**:
  - Immediate routing between MoE layers
  - Partial batch processing capability
  - Fine-grained pipeline stages

- **Sequence Flow**:
  1. Tokens arrive at layer N
  2. Route to experts on respective GPUs
  3. Compute in parallel
  4. Route outputs to layer N+1
  5. Repeat for all 61 layers

### **4. Large EP Integration**

#### **4.1 EP≥16 Configuration**
- **Definition**: Expert Parallelism ≥ 16
- **Implementation**: 
  - Minimum 16 GPUs per parallel group
  - Each GPU hosts exactly one expert
  - Network bandwidth becomes primary constraint

#### **4.2 Memory and Parallelism Integration**
- **Tensor Parallelism (TP)**:
  - Applied within each expert if exceeds GPU memory
  - Row/column partitioning for MLP layers
  - 2048 hidden dimension split across devices

- **Data Parallelism (DP)**:
  - Replicas of MoE network
  - Synchronized weight updates
  - Maintains expert-level parallelism

### **5. Implementation Parameters**

#### **5.1 Model Specifications**
- **Layers**: 61 total (3 dense + 58 MoE)
- **Experts**: Variable per layer (E total)
- **Expert Type**: MLP with hidden size 2048
- **Precision**: BF16 (2 bytes per parameter)

#### **5.2 Compute Requirements**
- **Per-expert Parameters**: 
  - Input projection: 7168 × 2048 × 2 = 29.36MB
  - Output projection: 2048 × 7168 × 2 = 29.36MB
  - Total per expert: ~58.72MB

- **Activation Memory**: 
  - Input tokens: batch_size × seq_len × 7168 × 2
  - Expert outputs: batch_size × seq_len × 7168 × 2
  - Total: ≤64GB per GPU

#### **5.3 Network Requirements**
- **Inter-node Bandwidth**: 1.8TBps per link
- **Utilization Target**: 80% sustained
- **Latency Tolerance**: <100μs for token transfers
- **Topology**: Fat-tree or direct-connect based on cluster size