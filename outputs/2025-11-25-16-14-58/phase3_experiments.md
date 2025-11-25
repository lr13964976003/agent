## **Experiments Extraction - Phase 3**

### **Experimental Setup**

#### **Model Configuration**
- **Architecture**: 16-layer MoE transformer
- **Experts per layer**: 16 (MLP experts)
- **Precision**: BF16
- **Token dimension**: 4096
- **MLP hidden size**: 16384
- **MHA**: 32 heads × 128 dimensions/head
- **Batch**: 128 sequences × 10000 tokens/sequence

#### **Hardware Environment**
- **GPUs**: Unlimited H100 resources
- **Per-GPU specifications**:
  - Compute: 400TFlops
  - VRAM: 64GB
  - Bandwidth: 1.8TBps
  - MFU: 60%
  - Bandwidth utilization: 80%

### **Baseline Configuration (TP=8, PP=2)**
- **Parallel strategy**: Tensor Parallelism=8, Pipeline Parallelism=2
- **Expert placement**: Multiple experts colocated per GPU
- **Processing**: Sequential token flow with shared GPU resources
- **Results**:
  - Throughput: 120,000 TPS
  - Latency: 8.3 ms TPOT

### **Proposed Configuration**
- **Parallel strategy**: Expert Parallelism=16 (one expert per GPU per layer)
- **Expert placement**: Single expert per GPU across 16 GPUs
- **Processing**: Full expert parallelism with asynchronous routing
- **Results**:
  - Throughput: 450,000 TPS
  - Latency: 2.2 ms TPOT

### **Performance Comparison**
| Metric | Baseline (TP=8, PP=2) | Proposed (EP=16) | Improvement |
|--------|------------------------|------------------|-------------|
| TPS | 120,000 | 450,000 | 3.75× |
| TPOT (ms) | 8.3 | 2.2 | 3.77× |
| GPUs | adequate | adequate | - |
| Expert/GPU | Multiple | One | - |

### **Key Experimental Findings**
1. **Throughput scaling**: 3.75× improvement from expert-level parallelism
2. **Latency reduction**: 3.77× reduction from dedicated GPU resources
3. **Resource utilization**: Full GPU utilization per expert without contention
4. **Communication overhead**: Mitigated through asynchronous routing and token batching

### **Scalability Validation**
- **EP regime**: Validated for EP ≥ 16
- **Near-linear scaling**: Achieved with topology-aware placement
- **Network efficiency**: Communication costs amortized across tokens