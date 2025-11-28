# Phase 3: Experiments Extraction

## **Experimental Setup - Detailed Configuration**

### **1. Hardware Environment**
- **GPU Type**: H100 with 64GB VRAM per card
- **Single-card Computing Power**: 400TFlops
- **Model FLOPs Utilization (MFU)**: 60%
- **VRAM Bandwidth**: 1.8TBps
- **VRAM Bandwidth Utilization**: 80%
- **Cluster Scale**: Adequate H100 resources with no limits (256+ GPUs)

### **2. Model Architecture**
- **Model Type**: 61-layer Mixture-of-Experts (MoE) Transformer
- **Layer Distribution**:
  - First 3 layers: Dense attention layers
  - Remaining 58 layers: MoE layers
- **Expert Configuration**:
  - **Number of Experts per Layer**: 256
  - **Expert Type**: MLP (Multi-Layer Perceptron)
  - **MLP Hidden Size**: 18432
  - **Activation Function**: GELU (implied from MLP structure)

### **3. Attention Configuration**
- **Mechanism**: Multi-Head Latent Attention (MLA)
  - **Number of Heads**: 128
  - **Dimension per Head**: 56
  - **Total Attention Dimension**: 128 × 56 = 7168
- **KV Cache Optimization**: Latent projection for memory reduction

### **4. Precision and Memory**
- **Numerical Precision**: FP8
- **Token Dimension**: 7168
- **Variable Parameters**:
  - **Batch Size**: Variable, optimized for sequence length
  - **Sequence Length**: Variable based on workload

### **5. Parallel Deployment Details**

#### **5.1 Proposed Cross-Node Expert Parallelism**
- **GPU Allocation**: 256 GPUs per MoE layer
- **Per-GPU Assignment**: Exactly one expert per GPU
- **Total GPU Requirement**: 256 × 58 = 14,848 GPUs for all MoE layers
- **Expert Distribution**: Each expert has dedicated GPU resources

#### **5.2 Routing Configuration**
- **Token Routing**: Dynamic routing based on gating network
- **Asynchronous Communication**: Token batches sent asynchronously
- **Network Topology**: Cross-node distribution with topology awareness
- **Load Balancing**: Real-time monitoring and adjustment

### **6. Performance Metrics**
- **Throughput Maximization**: All experts compute in parallel
- **Latency Minimization**: Minimal token latency through overlap
- **Resource Utilization**: 
  - GPU compute: 60% MFU
  - GPU memory: 64GB VRAM per expert
  - Network bandwidth: 80% utilization

### **7. Baseline Comparison**
- **Traditional Approach**: Multiple experts per GPU
- **New Approach**: One expert per GPU, communication handled through scheduling
- **Performance Gain**: Near-linear scaling through expert-level parallelism

### **8. Experimental Validation**
- **Setting**: Inference-only evaluation
- **Scale**: Large cluster environment (HPC-class)
- **Network**: Modern interconnects (NVLink, InfiniBand, NVSwitch)
- **Communication**: NCCL or MPI for cross-node transfers

### **9. Key Experimental Results**
- **Scalability**: Demonstrated near-linear scaling with 256 experts per layer
- **Resource Efficiency**: Full GPU utilization through dedicated expert assignment
- **Network Optimization**: Communication overhead masked by compute overlap
- **Load Balance**: Topology-aware placement prevents node hotspots

### **10. Measurement Parameters**
- **Throughput**: Tokens per second across all experts
- **Latency**: End-to-end token processing time
- **GPU Utilization**: Percentage of GPU compute capacity used
- **Network Efficiency**: Ratio of communication to computation time
- **Load Balance**: Standard deviation of tokens per expert