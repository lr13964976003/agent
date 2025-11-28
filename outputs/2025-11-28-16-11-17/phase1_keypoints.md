# Phase 1: Keypoints Extraction

## Key Contributions

### 1. Large-Scale Cross-Node Expert Parallelism
- **Core Innovation**: Deploy at most one expert per GPU to maximize computational parallelism
- **Definition**: EP ≥ 16 qualifies as "large EP" - true expert-level parallelism
- **Paradigm Shift**: From reducing communication to maximizing compute concurrency

### 2. Method Architecture
- **Target**: Mixture-of-Experts (MoE) models in large-scale inference
- **Environment**: High-performance computing (HPC) and large GPU clusters
- **Focus**: Inference-only setting with abundant H100 resources

### 3. Key Technical Components
1. **Expert Placement Strategy** - One expert per GPU across nodes
2. **Routing and Load Balancing** - Dynamic token routing with balanced distribution
3. **Communication Overlap** - Asynchronous token routing with compute-communication overlap

### 4. Model Specifications
- **Layer Count**: 61 layers (first 3 dense, 58 MoE layers)
- **Expert Count**: 256 experts per MoE layer
- **Token Dimension**: 7168
- **MLP Hidden Size**: 18432
- **MLA Configuration**: 128 heads, 56 dimensions per head
- **Precision**: FP8

### 5. Hardware Requirements
- **GPU**: H100 with 64GB VRAM
- **Compute**: 400TFlops per card at 60% MFU
- **Bandwidth**: 1.8TBps VRAM bandwidth at 80% utilization
- **Parallelism**: EP degree = 256 (one expert per GPU per layer)

### 6. Performance Characteristics
- **Scalability**: Near-linear scaling in large clusters
- **Contention**: Minimal inter-expert contention
- **Load Balance**: Topology-aware placement prevents bottlenecks
- **Memory Efficiency**: Compatible with tensor parallelism for large models

### 7. Novel Techniques
- **Multi-Head Latent Attention (MLA)**: Reduces KV cache memory via low-dimensional latent representations
- **Topology-Aware Routing**: Minimizes cross-node communication
- **Pipeline Scheduling**: Overlaps computation across layers

### 8. Baseline Comparison
- **Traditional Approach**: Multiple experts per GPU to reduce communication
- **New Approach**: One expert per GPU to maximize parallelism, with communication overhead managed through careful scheduling