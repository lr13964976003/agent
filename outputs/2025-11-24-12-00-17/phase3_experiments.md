# Phase 3: Experiments Extraction - Deployment Configurations and Results

## **1. Experimental Setup**

### **1.1 Model Configuration**
- **Architecture**: 16-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: BF16 (bfloat16)
- **Context**: Inference-only setting

### **1.2 Input Specifications**
- **Batch Size**: 128 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 1,280,000 tokens
- **Token Dimension**: 4096
- **MHA Configuration**:
  - Number of heads: 32
  - Dimension per head: 128
  - Total MHA dimension: 4096
- **MLP Hidden Size**: 16,384 dimensions

### **1.3 Hardware Environment**
- **GPU**: H100 (exact memory not specified)
- **Total GPUs**: 16 H100 GPUs
- **Network**: NVLink/InfiniBand/H100-class NVSwitch fabric

## **2. Baseline Deployment Configuration (TP=8, PP=2)**

### **2.1 Parallel Strategy Parameters**
- **Tensor Parallelism (TP)**: 8-way split
- **Pipeline Parallelism (PP)**: 2 stages
- **Data Parallelism (DP)**: Not explicitly used
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)

### **2.2 GPU Allocation**
- **Total GPUs**: 16 H100 GPUs
- **TP Group**: 8 GPUs (each group handles 1/8 of tensor-parallel shard)
- **PP Stages**: 2 stages × 8 GPUs each
- **Per-GPU Deployment**:
  - Each GPU holds 1/8 of tensor-parallel shard for all 16 layers
  - Multiple experts (8 per layer) colocated on each GPU
  - Shared compute resources among experts

### **2.3 Memory Layout per GPU**
- **Tensor Shards**: 1/8 of all layer parameters
- **Expert Storage**: 8 experts per layer × 16 layers = 128 experts per GPU
- **Token Buffer**: Shared among all experts on GPU
- **Activation Storage**: Shared tensor-parallel activations

## **3. Proposed Cross-Node Expert Parallelism Configuration**

### **3.1 Parallel Strategy Parameters**
- **Expert Parallelism (EP)**: 16 (one expert per GPU per layer)
- **Tensor Parallelism (TP)**: 1 (experts not split within GPU)
- **Pipeline Parallelism (PP)**: 1 (no pipeline within MoE layers)
- **Data Parallelism (DP)**: 1 (single replica)

### **3.2 GPU Allocation**
- **Total GPUs**: 16 H100 GPUs
- **Per-GPU Deployment**: Exactly one expert per layer
- **Layer Distribution**: 16 layers × 16 experts = 256 total expert instances
- **GPU Assignment**: Each of 16 GPUs hosts 16 experts (one per layer)

### **3.3 Expert-to-GPU Mapping**
```
GPU 0: Expert[layer=0, expert_id=0]
GPU 1: Expert[layer=0, expert_id=1]
...
GPU 15: Expert[layer=0, expert_id=15]
GPU 0: Expert[layer=1, expert_id=0]
...
GPU 15: Expert[layer=15, expert_id=15]
```

### **3.4 Memory Layout per GPU**
- **Expert Parameters**: Complete parameters for 16 experts (one per layer)
- **Per Expert Storage**:
  - MLP parameters: ~134M per expert
  - Total per GPU: 16 × 134M = ~2.14B parameters
- **Token Routing Buffer**: Separate buffers for each layer
- **Communication Buffer**: Async token transfer staging

### **3.5 Token Routing Flow**
1. **Input Distribution**: 128 sequences × 10,000 tokens = 1.28M tokens
2. **Gating Network**: Top-K expert selection per token
3. **Token Sharding**: Group tokens by destination expert
4. **Async Transfer**: Send token batches to expert GPUs
5. **Expert Computation**: Parallel processing on all 16 GPUs
6. **Result Collection**: Gather outputs back to routing GPU

## **4. Performance Results**

### **4.1 Throughput Comparison**
| Metric | Baseline (TP=8, PP=2) | Proposed (EP=16) | Improvement |
|--------|----------------------|------------------|-------------|
| TPS (Tokens/s) | 120,000 | 450,000 | 3.75× |
| TPOT (ms) | 8.3 | 2.2 | 3.77× |
| GPU Utilization | Shared experts | Dedicated experts | Higher |
| Communication | Intra-node | Cross-node | Optimized |

### **4.2 Bottleneck Analysis**
- **Baseline**: Intra-GPU contention from 8 experts sharing resources
- **Proposed**: Network communication becomes primary bottleneck
- **Mitigation**: Asynchronous routing and token batching

### **4.3 Scaling Characteristics**
- **Linear Scaling**: Near-linear scaling for EP ≥ 16
- **Network Requirements**: High-bandwidth, low-latency interconnects
- **Memory Efficiency**: No tensor parallelism within experts (unless needed)

## **5. Deployment Configuration Details**

### **5.1 Expert Placement Algorithm**
```python
# Pseudo-algorithm for expert placement
def place_experts(num_layers=16, num_experts=16, num_gpus=16):
    placement = {}
    for layer in range(num_layers):
        for expert_id in range(num_experts):
            gpu_id = expert_id % num_gpus
            placement[(layer, expert_id)] = gpu_id
    return placement
```

### **5.2 Communication Patterns**
- **All-to-All**: Each GPU may send tokens to any other GPU
- **Batch Size**: Optimize based on network bandwidth
- **Latency Hiding**: CUDA streams for concurrent operations

### **5.3 Load Balancing Metrics**
- **Token Distribution**: Monitor tokens per expert in real-time
- **Rebalancing Trigger**: Load imbalance threshold
- **Gating Adjustment**: Dynamic probability modification

## **6. Memory Requirements**

### **6.1 Per Expert Memory**
- **Parameters**: 134M × 2 bytes (BF16) = 268MB per expert
- **Activations**: ~50MB per expert (estimated)
- **Total per Expert**: ~318MB

### **6.2 Per GPU Memory**
- **16 Experts**: 16 × 318MB = ~5.1GB
- **Token Buffers**: ~2GB for 1.28M tokens
- **Communication Buffers**: ~1GB
- **Total per GPU**: ~8.1GB (well within H100 memory)

## **7. Network Bandwidth Requirements**
- **Peak Bandwidth**: All-to-all communication among 16 GPUs
- **Token Transfer**: 1.28M tokens × 4096 bytes = 5.24GB per iteration
- **Bandwidth Utilization**: Optimized through batching and overlapping
- **Latency Tolerance**: Hidden through async computation