# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Type**: 61-layer Mixture-of-Experts (MoE) model
- **Layer Distribution**: First 3 layers are dense, remaining 58 layers are MoE
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: BF16 floating point format
- **Variable Parameters**: 
  - Batch size: Variable
  - Sequence length: Variable

### 1.2 Dimensional Specifications
- **Token Dimension**: 7168 dimensions per token
- **MLA Configuration**:
  - Number of heads: 128
  - Dimension per head: 56
  - Total MLA dimension: 128 × 56 = 7168
- **MLP Hidden Size**: 18432 dimensions

### 1.3 Hardware Environment
- **GPUs**: H100 GPUs with "adequate resources" (no specific limit given)
- **Compute Specifications**:
  - Single-card computing power: 400 TFlops
  - MFU (Model FLOPs Utilization): 60%
- **Memory Specifications**:
  - VRAM Bandwidth: 1.8 TBps
  - Bandwidth utilization: 80%
  - Single-card video memory capacity: 64GB

## 2. Parallel Deployment Configuration

### 2.1 Proposed Cross-Node Expert Parallelism
- **GPU Allocation**: One GPU per expert per layer
- **Deployment Strategy**: Each GPU hosts exactly one expert per layer
- **Total GPUs Required**: Number of experts × Number of MoE layers
- **Routing Mechanism**: 
  - Dynamic routing of input tokens to GPU holding corresponding expert
  - Asynchronous token batch transfer to minimize idle time

### 2.2 Parallelism Benefits
- **Expert-Level Parallelism**: All experts per layer compute in parallel
- **Resource Utilization**: Maximizes throughput by eliminating intra-GPU expert contention
- **Token Latency**: Minimized through parallel expert processing

## 3. Evaluation Context
- **Setting**: Inference-only evaluation
- **Environment**: High-performance computing (HPC) cluster
- **Network Infrastructure**: Modern HPC networking (NVLink/InfiniBand assumed)
- **Scalability Focus**: Large-scale deployment with abundant GPU resources

## 4. Key Performance Indicators
- **Throughput**: Maximized through parallel expert computation
- **Latency**: Reduced through elimination of expert contention
- **Scalability**: Near-linear scaling in large MoE deployments
- **Resource Efficiency**: Full GPU utilization maintained through one-expert-per-GPU policy