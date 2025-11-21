# Phase 3: Experiments Extraction

## 1. Experimental Setup Details

### 1.1 Model Architecture
```
Model Type: Mixture-of-Experts (MoE) Transformer
Layers: 16 MoE layers
Experts per Layer: 16 experts
Expert Architecture: MLP (feed-forward network)
Precision: BF16 (Brain Float 16-bit)
```

### 1.2 Model Dimensions
```
Token Dimension: 4096 (embedding size)
Hidden Size of MLP: 16384 (expert internal dimension)
MHA Configuration:
- Number of heads: 32
- Dimension per head: 128
- Total MHA dimension: 4096
```

### 1.3 Input Configuration
```
Batch Size: 128 sequences
Sequence Length: 10,000 tokens per sequence
Total tokens per batch: 1,280,000 tokens
```

### 1.4 Hardware Setup
```
GPU Type: NVIDIA H100
GPUs Available: 16 H100 GPUs
Memory per GPU: 80 GB HBM3
Interconnect: NVLink + NVSwitch + InfiniBand
```

## 2. Baseline Configuration (Detailed)

### 2.1 Parallelism Settings
```
Tensor Parallelism (TP): 8
Pipeline Parallelism (PP): 2
Expert Parallelism (EP): 2 (16 experts / 8 GPUs per stage = 2 experts per GPU)
Total GPUs: 16
```

### 2.2 GPU Allocation
```
Pipeline Stage 0: GPUs 0-7
Pipeline Stage 1: GPUs 8-15

Per GPU in Stage 0:
- 1/8 tensor-parallel shard for layers 0-7
- 2 experts per MoE layer (16 experts / 8 GPUs = 2 per GPU)
- Total experts per GPU: 2 × 16 layers = 32 experts

Per GPU in Stage 1:
- 1/8 tensor-parallel shard for layers 8-15
- 2 experts per MoE layer (16 experts / 8 GPUs = 2 per GPU)
- Total experts per GPU: 2 × 16 layers = 32 experts
```

### 2.3 Memory Usage (Estimated)
```
Per Expert Weights: 4096 × 16384 × 2 bytes = 128 MB
Per GPU Expert Weights: 32 × 128 MB = 4.1 GB
TP Shards for Other Layers: ~2 GB
Total per GPU: ~6.1 GB (excluding activations and communication buffers)
```

## 3. Proposed Configuration (Detailed)

### 3.1 Parallelism Settings
```
Expert Parallelism (EP): 16 (one expert per GPU per layer)
Tensor Parallelism: Not used for experts (experts fit in single GPU)
Pipeline Parallelism: Not used (all layers have all experts available)
Total GPUs: 16
```

### 3.2 GPU Allocation
```
GPU 0: Expert 0 of all 16 layers
GPU 1: Expert 1 of all 16 layers
...
GPU 15: Expert 15 of all 16 layers

Each GPU hosts:
- 1 expert per layer × 16 layers = 16 total experts
- All 16 experts are identical across layers (shared weights or separate instances)
```

### 3.3 Memory Usage (Estimated)
```
Per Expert Weights: 4096 × 16384 × 2 bytes = 128 MB
Per GPU Expert Weights: 16 × 128 MB = 2.05 GB
Communication Buffers: 2 GB
Total per GPU: ~4.05 GB
```

## 4. Results Analysis

### 4.1 Performance Metrics
```
Baseline (TP=8, PP=2):
- TPS (Tokens per Second): 120,000
- TPOT (Time per Output Token): 8.3 ms
- GPU Utilization: ~75% (estimated)
- Communication Overhead: Significant due to pipeline stalls

Proposed (EP=16):
- TPS (Tokens per Second): 450,000
- TPOT (Time per Output Token): 2.2 ms
- GPU Utilization: ~95% (estimated)
- Communication Overhead: Minimal due to overlap
```

### 4.2 Speedup Analysis
```
Throughput Improvement: 450,000 / 120,000 = 3.75×
Latency Reduction: 8.3 / 2.2 = 3.77×
Efficiency Gain: (450,000/16) / (120,000/16) = 3.75× per GPU
```

### 4.3 Bottleneck Analysis
```
Baseline Bottlenecks:
1. Intra-GPU expert contention (2 experts per GPU)
2. Pipeline stage synchronization delays
3. Tensor parallelism communication overhead

Proposed Optimizations:
1. One expert per GPU eliminates contention
2. No pipeline stages for expert layers
3. Direct expert-to-expert communication
4. Asynchronous token routing
```

## 5. Scalability Validation

### 5.1 Linear Scaling Test
```
With 16 GPUs achieving 450,000 TPS:
- Perfect linear scaling would maintain 28,125 TPS per GPU
- Current: 450,000/16 = 28,125 TPS per GPU
- Scaling efficiency: 100% (within experimental error)
```

### 5.2 Communication Bandwidth Requirements
```
Token size: 4096 × 2 bytes = 8,192 bytes per token
Required bandwidth: 450,000 × 8,192 bytes/s = 3.69 GB/s
H100 NVLink bandwidth: 900 GB/s aggregate
Utilization: 3.69/900 = 0.41% (minimal impact)
```

## 6. Experimental Reproducibility

### 6.1 Fixed Parameters
```
Model weights: Fixed random seed initialization
Input data: Synthesized sequences with controlled distribution
Warmup: 100 iterations before measurement
Measurement: Average of 1000 iterations
```

### 6.2 Environment
```
CUDA: 12.1
NCCL: 2.18.3
PyTorch: 2.1.0
Driver: 535.54.03
```