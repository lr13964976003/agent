# Phase 2: Methodology Extraction

## Core Methodological Framework

### 1. Operator Graph Representation
The paper employs a computational graph representation where:
- **Nodes**: Operators (layers) or tensors
- **Edges**: Data dependencies (tensors flowing between operators)
- **Tensors**: n-dimensional arrays with specific dimensions

#### Tensor Types
- **Parameter tensors**: Static weights (e.g., W matrices)
- **Activation tensors**: Intermediate computed values (e.g., layer outputs)
- **Input tensor X**: Shape (batch_size × sequence_length × hidden_dim)
- **Output tensor Y**: Depends on task

### 2. Parallelism Taxonomy

#### 2.1 Intra-Operator Parallelism (Tensor Parallelism)
**Definition**: Parallelizing computations within a single operator

**Mathematical Formulation**:
- For matrix multiplication Y = XW^T
- **Column partitioning**: Split W along columns → W = [W₁, W₂]
  - Y = [XW₁^T, XW₂^T] (no communication needed)
- **Row partitioning**: Split W along rows → W = [W₁; W₂]
  - Requires communication: Y = X₁W₁^T + X₂W₂^T

**Transformer MLP Implementation**:
```
MLP(X) = GELU(X·W₁)·W₂
- Partition W₁ by columns: W₁ = [W₁₁, W₁₂]
- Partition W₂ by rows: W₂ = [W₂₁; W₂₂]
- Device 0: computes GELU(X·W₁₁)·W₂₁
- Device 1: computes GELU(X·W₁₂)·W₂₂
- Final result: sum across devices
```

#### 2.2 Inter-Operator Parallelism (Pipeline Parallelism)
**Definition**: Distributing different operators/layers across devices

**Pipeline Structure**:
- **Forward pass**: X → Layer₁ → Layer₂ → ... → Layerₙ
- **Backward pass**: Gradients flow in reverse
- **Micro-batching**: Split batch into smaller chunks to reduce pipeline bubbles

**Activation Memory Calculation**:
- **Single layer**: s·b·h(34 + 5·a·s/h) bytes
  - s = sequence length, b = batch size, h = hidden dimension, a = attention heads
- **With tensor parallelism**: Reduced by factor t (parallelism degree)

#### 2.3 Hybrid Parallelism Strategy
**Megatron's Approach** [28, 21, 14]:
1. **Intra-layer (tensor) parallelism**: 8-way within single node
2. **Inter-layer (pipeline) parallelism**: 64-way across nodes
3. **Data parallelism**: Additional scaling beyond model parallelism

**Optimal Strategy Formula** [21]:
- **Tensor parallelism**: Up to g-way for g-GPU servers
- **Pipeline parallelism**: Scale across servers
- **Total model-parallel size**: M = t × p (t = tensor, p = pipeline)
- **Batch size optimization**: B = b × p × d (b = microbatch, d = data parallel)

### 3. Memory Optimization Techniques

#### 3.1 Activation Checkpointing
- **Trade-off**: Recompute activations during backward pass to save memory
- **Memory reduction**: ~√n reduction for n layers

#### 3.2 Sequence Parallelism [14]
- **Concept**: Parallelize along sequence dimension for activations
- **Memory formula**: s·b·h/t(34 + 5·a·s/h)
- **Efficiency**: Reduces activation memory proportional to parallelism degree

#### 3.3 Precision Optimization
- **Mixed precision**: FP16/BF16 for compute, FP32 for accumulation
- **Quantization**: Lower precision weights (INT8/INT4) for inference

### 4. Search Space Formulation

#### 4.1 FlexFlow SOAP Framework [13]
**Four Parallelizable Dimensions**:
1. **Sample**: Batch dimension splitting
2. **Operator**: Layer-level distribution
3. **Attribute**: Fine-grained parallelization within operators
4. **Parameter**: Weight matrix splitting

**Strategy Representation**:
- S = {cᵢ | cᵢ ∈ Z^{|Pᵢ|}}
- cᵢ = degree of parallelism for dimension Pᵢ

#### 4.2 Cost Model
**Execution time estimation**:
```
T_total = T_compute + T_communication + T_memory
```

**Compute time**: Based on FLOPs count and device peak FLOPS
**Communication time**: 
- Intra-node: NVLink bandwidth (~600 GB/s)
- Inter-node: InfiniBand bandwidth (~100 GB/s)

### 5. Implementation Details

#### 5.1 Megatron Tensor Parallel Implementation
**Transformer Layers**:
- **Self-attention**: Split Q, K, V matrices column-wise
- **MLP**: First layer column-parallel, second layer row-parallel
- **LayerNorm**: Replicated across devices

**Communication patterns**:
- **All-reduce**: Required after row-parallel operations
- **All-gather**: Required for tensor reconstruction

#### 5.2 Pipeline Parallel Scheduling
**GPipe approach** [11]:
- **Staged execution**: Forward passes for all micro-batches first
- **Bubble reduction**: Micro-batch size = batch_size / num_stages

**PipeDream approach** [20]:
- **Asynchronous scheduling**: Interleave forward/backward passes
- **Weight staleness**: 1F1B (1 forward 1 backward) scheduling

### 6. Hardware-Specific Optimizations

#### 6.1 NVIDIA A100 Configuration
- **Intra-node**: 8×A100 with NVLink 3.0
- **Peak performance**: 312 TFLOPS (FP16)
- **Memory**: 40/80 GB HBM2e per GPU
- **Bandwidth**: 2 TB/s HBM2e, 600 GB/s NVLink

#### 6.2 Google TPU v4 Configuration
- **Pod structure**: 4096 chips per pod
- **Interconnect**: 600 GB/s chip-to-chip
- **Memory**: 32 GB HBM per chip
- **Peak performance**: 275 TFLOPS (BF16)

### 7. Evaluation Methodology

#### 7.1 Performance Metrics
- **Model FLOPs Utilization (MFU)**: Effective throughput vs theoretical maximum
- **Scaling efficiency**: Performance per GPU vs single GPU
- **Memory efficiency**: Peak memory usage vs available memory

#### 7.2 Benchmark Settings
- **Models**: GPT-2 (8.3B), Megatron-LM (530B), PaLM (540B)
- **Datasets**: C4, The Pile, Common Crawl
- **Training**: Adam optimizer, cosine learning rate schedule
- **Hardware utilization targets**: >50% MFU for optimal configurations