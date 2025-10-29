# Model Parallelism on Distributed Infrastructure: A Concise Review

## ABSTRACT
Neural networks have become a cornerstone of machine learning. As the trend for these to get more and more complex continues, so does the underlying hardware and software infrastructure for training and deployment. In this survey we answer three research questions: "What types of model parallelism exist?", "What are the challenges of model parallelism?", and "What is a modern use-case of model parallelism?" We answer the first question by looking at how neural networks can be parallelised and expressing these as operator graphs while exploring the available dimensions. The dimensions along which neural networks can be parallelised are intra-operator and inter-operator. We answer the second question by collecting and listing both implementation challenges for the types of parallelism, as well as the problem of optimally partitioning the operator graph. We answer the last question by collecting and listing how parallelism is applied in modern multi-billion parameter transformer networks, to the extend that this is possible with the limited information shared about these networks.

## 1. INTRODUCTION

Neural networks have become a cornerstone in machine learning, offering solutions for complex prediction tasks. As these networks grow in complexity, both computational requirements and memory footprint for training and inference increase proportionally. Scaling up neural networks presents engineering challenges:

1. **Increased compute**: More neurons and operations
2. **Memory requirements**: More parameters and intermediate activations
3. **Training data**: Larger datasets require more passes

Model parallelism addresses these challenges by partitioning the model across multiple devices, but introduces communication overhead and complexity in partitioning strategies.

## 2. MODEL PARALLELISM FRAMEWORK

### 2.1 Computational Model
Neural networks are represented as operator graphs O = (V, E) where:
- **V**: Operators (layers) or tensors
- **E**: Data dependencies between operators
- **Tensors**: n-dimensional arrays (parameters and activations)

### 2.2 Parallelism Types

#### 2.2.1 Intra-Operator Parallelism (Tensor Parallelism)
Parallelizes computation within individual operators:
- **Column partitioning**: Split weight matrices along columns
- **Row partitioning**: Split weight matrices along rows
- **Communication**: All-reduce after row operations

#### 2.2.2 Inter-Operator Parallelism (Pipeline Parallelism)
Distributes layers across devices:
- **Pipeline stages**: Different layers on different devices
- **Micro-batching**: Reduces pipeline bubbles
- **Communication**: Between consecutive stages

#### 2.2.3 Hybrid Parallelism
Combines multiple strategies:
- **Tensor + Pipeline + Data parallelism**
- **Optimal strategy depends on hardware topology**

## 3. TRANSFORMER PARALLELIZATION

### 3.1 Key Components
- **Multi-Layer Perceptron (MLP)**: Two linear layers with GELU
- **Self-Attention**: Q, K, V matrices computation
- **Layer Normalization**: Applied after each component

### 3.2 Parallel Strategies

#### 3.2.1 MLP Layer
```
MLP(X) = GELU(X·W₁)·W₂
- W₁: column-parallel split
- W₂: row-parallel split
- Communication: All-reduce across tensor parallel group
```

#### 3.2.2 Attention Layer
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Q,K,V: column-parallel split
- Output projection: row-parallel split
```

## 4. EXPERIMENTAL RESULTS

### 4.1 Model Configurations
| Model | Parameters | Layers | Hidden | Hardware | MFU |
|-------|------------|--------|--------|----------|-----|
| Megatron-8.3B | 8.3B | 24 | 4096 | 512×V100 | 30% |
| Megatron-530B | 530B | 105 | 20480 | 17920×A100 | 36% |
| Megatron-1T | 1T | 128 | 25600 | 512×A100 | 56% |
| Gopher | 280B | 80 | 16384 | 4096×TPU v3 | 52% |
| PaLM | 540B | 118 | 18432 | 6144×TPU v4 | 46% |

### 4.2 Parallelism Configurations
| Model | Tensor | Pipeline | Data | Micro-batch | Batch |
|-------|--------|----------|------|-------------|--------|
| 8.3B | 8 | 1 | 64 | 128 | 1024 |
| 530B | 8 | 35 | 12 | 1 | 1920 |
| 1T | 8 | 64 | 1 | 32 | 2048 |
| Gopher | 8 | 4 | 1024 | 512 | 3M |
| PaLM | 12 | 0 | 512 | 2048 | 2M |

### 4.3 Memory Analysis
**Activation memory per layer**: s·b·h(34 + 5·a·s/h) bytes
- **s**: sequence length (2048)
- **b**: batch size
- **h**: hidden dimension
- **a**: attention heads

**Memory optimization techniques**:
- **Tensor parallelism**: Reduces per-device memory by 1/t
- **Sequence parallelism**: Reduces activation memory further
- **Activation checkpointing**: Trade compute for memory

## 5. IMPLEMENTATION CHALLENGES

### 5.1 Communication Overhead
- **Intra-node (NVLink)**: 600 GB/s, ~15% overhead
- **Inter-node (InfiniBand)**: 25 GB/s, bottleneck for pipeline
- **All-reduce operations**: 20-30% of training time

### 5.2 Load Balancing
- **Pipeline bubbles**: Micro-batching reduces to <5%
- **Tensor imbalance**: Column-row splitting provides balance
- **Memory imbalance**: Sequence parallelism addresses activation memory

### 5.3 Hardware Constraints
- **Memory capacity**: 40-80 GB per GPU
- **Interconnect bandwidth**: Limits maximum parallelism
- **Power consumption**: 400W per A100 GPU

## 6. OPTIMIZATION STRATEGIES

### 6.1 Search Space Formulation
**FlexFlow SOAP dimensions**:
- **Sample**: Batch dimension
- **Operator**: Layer distribution
- **Attribute**: Fine-grained parallelization
- **Parameter**: Weight matrix splitting

### 6.2 Cost Model
```
T_total = T_compute + T_communication + T_memory
```
- **Compute time**: Based on FLOPs and peak FLOPS
- **Communication time**: Network bandwidth and latency
- **Memory time**: HBM bandwidth utilization

### 6.3 Optimization Techniques
- **Mixed precision**: FP16/BF16 training
- **Gradient checkpointing**: Memory-compute trade-off
- **Operator fusion**: Reduces communication
- **Overlapping communication**: Computation and communication overlap

## 7. CONCLUSION

### 7.1 Key Findings
1. **No single parallelism type** addresses all challenges for billion-parameter models
2. **Hybrid approaches** combining tensor, pipeline, and data parallelism are essential
3. **Communication overhead** remains the primary bottleneck
4. **Specialized hardware** (TPUs) enables higher parallelism degrees

### 7.2 Future Directions
- **Standardization** of benchmarking and evaluation metrics
- **Automated parallelization** using ML-based search
- **Hardware-software co-design** for optimal mapping
- **Energy-efficient** parallelization strategies

## APPENDIX: MATHEMATICAL FORMULATIONS

### A.1 Memory Calculations
**Model parameters**: 2×P bytes (FP16)
**Optimizer states**: 8×P bytes (Adam)
**Activations**: s·b·h(34 + 5·a·s/h) per layer

### A.2 Communication Volume
**Attention layer**: 12×h² + 2×s×h bytes
**MLP layer**: 8×h² bytes
**Total per transformer layer**: 20×h² + 2×s×h bytes

### A.3 Performance Metrics
**Model FLOPs Utilization (MFU)**: 
Actual FLOPs / Theoretical FLOPs
**Scaling efficiency**: 
Performance per GPU / Single GPU performance

### A.4 Optimal Configuration
For a given model size P and hardware:
- **Tensor parallelism**: min(8, max_gpus_per_node)
- **Pipeline parallelism**: optimize for memory balance
- **Data parallelism**: remaining GPUs for throughput
- **Batch size**: balance memory and compute efficiency