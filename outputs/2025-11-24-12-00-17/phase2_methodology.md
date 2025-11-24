# Phase 2: Methodology Extraction - Large-Scale Cross-Node Expert Parallelism

## **1. Expert Placement Strategy**

### **1.1 Single-Expert-Per-GPU Deployment**
- **Principle**: At most one expert per GPU, ensuring maximal expert-level parallelism
- **Implementation**: 
  - For MoE layer with E experts and cluster of G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs while maximizing concurrency of independent experts
  - Memory usage balancing across replicas

### **1.2 Cross-Node Distribution**
- **Topology-Aware Placement Algorithm** considers:
  - Node-to-node bandwidth and latency measurements
  - GPU memory capacity per node (H100 specifications)
  - Expected token routing patterns from gating network
- **Objective Function**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU constraint

## **2. Routing and Load Balancing**

### **2.1 Gating Mechanism**
- **Standard MoE Architecture**: Top-K gating scores determine expert activation per token
- **K Value**: Not explicitly stated, but typically K=2 for MoE models
- **Gating Function**: Softmax-based routing probabilities

### **2.2 Token Sharding Across Nodes**
- **Token Batching Strategy**:
  1. Group tokens by destination expert to reduce network messages
  2. Batch size optimization for H100 NVLink/InfiniBand bandwidth
  3. Minimize message count while maintaining compute efficiency

- **Asynchronous Routing Pipeline**:
  - Send token batches asynchronously to overlapping expert computation
  - Use CUDA streams for non-blocking transfers
  - NCCL or MPI for inter-node communication

- **Dynamic Load Balancing**:
  - Monitor per-expert load using token counters
  - Adjust gating probabilities based on real-time load metrics
  - Prevent expert overloading that causes stragglers

## **3. Communication Overlap and Scheduling**

### **3.1 Overlapping Compute and Communication**
- **Interleaved Processing**:
  - While current batch processes on GPU, next batch transfers simultaneously
  - Double-buffering technique for token batches
  - CUDA streams for concurrent kernel execution and data transfer

- **Implementation Details**:
  - Stream 1: Expert computation
  - Stream 2: Token transfer from other nodes
  - Synchronization barriers between computation phases

### **3.2 Pipeline Scheduling**
- **Multi-Layer MoE Pipeline**:
  - Token outputs from layer n immediately routed to layer n+1 experts
  - Partial batch processing: experts start computation as soon as partial tokens arrive
  - Fine-grained pipeline stages prevent full-batch waiting

- **Layer Dependencies**:
  - MoE layer 1 → Expert selection → Token transfer → MoE layer 2 → ... → MoE layer 16
  - Overlap computation across consecutive layers

## **4. Memory and Model Parallelism Integration**

### **4.1 Tensor Model Parallelism (TP) within Experts**
- **When to Apply**: When individual experts exceed single-GPU memory
- **TP Configuration**: 
  - Column-parallel for first linear layer (hidden_size → ffn_hidden_size)
  - Row-parallel for second linear layer (ffn_hidden_size → hidden_size)
  - TP degree: Determined by expert memory requirements

### **4.2 Data Parallelism (DP) Integration**
- **DP for Weight Updates**: Applied across replicas of MoE network
- **Synchronization**: All-reduce operations for gradient synchronization
- **DP Degree**: Number of complete MoE replicas

### **4.3 Memory Layout per GPU**
- **Single Expert Storage**: All parameters for one expert (MLP weights, biases)
- **Token Buffer**: Temporary storage for incoming tokens
- **Activation Buffer**: Intermediate activations during computation
- **Communication Buffer**: Asynchronous token transfer staging area

## **5. System Architecture Details**

### **5.1 Hardware Specifications**
- **GPU**: H100 with specific memory capacity (exact GB not specified)
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch fabric
- **Topology**: 16 GPUs distributed across nodes

### **5.2 Software Stack**
- **Communication Libraries**: NCCL for NVIDIA GPUs, MPI for cross-node
- **Framework**: Likely PyTorch or JAX with custom MoE implementation
- **CUDA Streams**: Multiple streams for concurrent operations
- **Memory Management**: Unified memory for cross-node access

## **6. Expert MLP Structure**
- **Input Dimension**: 4096 (token embedding size)
- **Hidden Layer**: 16384 dimensions (4× expansion ratio)
- **Activation Function**: GELU
- **Output Dimension**: 4096 (back to token embedding size)
- **Parameters per Expert**:
  - First linear: 4096 × 16384 = 67,108,864 parameters
  - Second linear: 16384 × 4096 = 67,108,864 parameters
  - Total: ~134M parameters per expert

## **7. Load Balancing Algorithm**
- **Monitoring**: Real-time token distribution tracking
- **Adjustment**: Dynamic gating probability modification
- **Threshold**: Load imbalance tolerance before triggering rebalancing
- **Frequency**: Continuous monitoring with threshold-based triggers