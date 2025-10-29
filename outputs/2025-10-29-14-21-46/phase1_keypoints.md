# Phase 1: Key Points Extraction

## Research Questions Addressed
1. **What types of model parallelism exist?**
   - Intra-operator parallelism: Parallelizing within individual operators
   - Inter-operator parallelism: Parallelizing across multiple operators
   - Hybrid parallelism: Combining both intra- and inter-operator parallelism

2. **What are the challenges of model parallelism?**
   - High communication requirements for intra-operator parallelism
   - Low device utilization for inter-operator parallelism
   - Optimal partitioning strategy selection
   - Hardware limitations (memory, bandwidth)

3. **What is a modern use-case of model parallelism?**
   - Large Language Models (LLMs) with billions of parameters
   - Transformer architectures (GPT, Megatron, PaLM, Gopher)

## Key Technical Concepts

### Model Parallelism Dimensions
- **Intra-operator parallelism**: Splitting individual operators (e.g., matrix multiplications) across devices
- **Inter-operator parallelism**: Distributing different operators/layers to different devices
- **Data parallelism**: Replicating model across devices, splitting data

### Transformer Architecture Components
- **MLP (Multi-Layer Perceptron)**: Two linear layers with GELU activation
- **Self-attention**: Q, K, V matrices computation
- **Layer normalization**: Applied at various points

### Parallel Strategies
- **Tensor parallelism**: Intra-operator approach for transformer layers
- **Pipeline parallelism**: Inter-operator approach across layers
- **Sequence parallelism**: Parallelizing along sequence dimension for activations

### Key Models Discussed
1. **Megatron family** (8.3B, 530B, 1T parameters)
2. **Gopher** (280B parameters)
3. **PaLM** (540B parameters)
4. **GPT-3** (175B parameters)

### Hardware Configurations
- **NVIDIA**: V100 (32GB), A100 (40/80GB)
- **Google**: TPU v3, TPU v4
- **Interconnects**: NVLink (within node), InfiniBand/Ethernet (between nodes)

### Performance Metrics
- **Hardware FLOPs Utilization (HFU)**: Raw hardware utilization
- **Model FLOPs Utilization (MFU)**: Effective utilization considering memory/compute trade-offs
- **Scaling efficiency**: Performance relative to ideal linear scaling

## Critical Insights
- No single parallelism type addresses all challenges for billion-parameter models
- Hybrid approaches combining tensor, pipeline, and data parallelism are essential
- Communication overhead is the primary bottleneck for intra-operator parallelism
- Specialized hardware (TPUs) enables higher parallelism degrees than commodity GPUs