# Phase 3: Experiments Extraction

## **Experimental Setup**

### **Model Configuration**
- **Layers**: 16-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 64 experts
- **Precision**: FP8
- **Batch size**: 128 sequences per batch
- **Sequence length**: 128 tokens per sequence
- **Token dimension**: 1024
- **MHA configuration**: 16 heads × 64 dimension per head
- **MoE hidden size**: 2048

### **Hardware Setup**
- **GPUs**: H100 GPUs (adequate number)
- **Setting**: Inference-only
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

## **Parallel Deployment Configurations**

### **Baseline Deployment (TP=8, PP=2)**
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU holds 8 tensor-parallel shards for all layers
  - Experts are colocated on 16 GPUs
- **Processing**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources
- **Performance**: TPS = 120,000, TPOT = 8.3ms

### **Proposed Cross-Node Expert Parallelism**
- **GPUs Used**: 16 H100 GPUs (one GPU per expert per layer)
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Routing**: 
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent, ensuring minimal idle time
- **Performance**: TPS = 450,000, TPOT = 2.2ms

## **Results Summary**

| Method | GPUs Used | Per-GPU Deployment | TPS | TPOT |
|--------|-----------|-------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | TP shard per GPU | 120,000 | 8.3ms |
| Proposed Cross-Node EP | 16 | 1 expert each layer per GPU | 450,000 | 2.2ms |

### **Performance Improvements**
- **Throughput improvement**: ~3.75× higher (120,000 → 450,000 TPS)
- **Latency improvement**: ~3.8× lower (8.3ms → 2.2ms TPOT)
- **Key enabler**: All 16 experts per layer compute in parallel without intra-GPU contention

## **Discussion Points**
- Deploying one expert per GPU allows full utilization of GPU compute and memory
- Asynchronous token routing ensures minimal waiting, even across nodes
- With 16 GPUs, system scales near-linearly in large EP regime (EP ≥ 16)
- Method particularly effective in GPU-rich environments like H100 clusters