# Phase 2: Methodology Extraction (CORRECTED)

## Layer-Wise Deployment Strategy for 4-Layer Model

### 1. Problem Formulation
Given a 4-layer dense network with 30B parameters:
- L = {l₁, l₂, l₃, l₄}
- Goal: Partition into k groups P = {P₁, P₂, ..., Pₖ}
- Memory constraint: Each partition Pᵢ must fit in SRAM/L2 cache capacity C
- Minimize k (use all 16 GPUs optimally)

### 2. Memory Footprint Estimation (Per Layer)
For each layer lⱼ:
- **Weight size**: 30B params / 4 layers = 7.5B parameters per layer × 2 bytes = **15GB**
- **Activation size**: Using batch_size=128, seq_len=10000, hidden_size=4096 (32×128)
  - Activation memory: 128 × 10000 × 4096 × 2 bytes = **10.48GB**
- **Buffer size**: ~200MB for intermediate computations
- **Total per layer**: ~25.7GB

### 3. Partitioning Algorithm Applied to 4-Layer Model
**Greedy Layer Aggregation for 4 layers:**
- Each layer treated as single partition (P₁=l₁, P₂=l₂, P₃=l₃, P₄=l₄)
- 16 GPUs available, 4 layers → Multiple GPUs per layer
- Alternative: Group layers to utilize all 16 GPUs effectively

### 4. Deployment Strategy for 4-Layer Model
**Optimal deployment using 16 GPUs:**
- Deploy each layer across 4 GPUs (16 GPUs / 4 layers = 4 GPUs per layer)
- Within each layer: Use tensor parallelism across 4 GPUs
- Between layers: Pipeline parallelism across the 4 layers

### 5. Handling Edge Cases
- Single layer (25.7GB) exceeds typical SRAM (~50MB) but fits in HBM (~80GB)
- Solution: Use HBM as primary storage, SRAM for caching
- Batch size 128 is optimal for memory constraints

## Summary for 4-Layer Model:
- 4 layers total, 30B parameters
- Each layer: ~25.7GB total memory
- Deployment: 4 GPUs per layer (TP=4 within layer)
- Pipeline: PP=4 across layers
- Baseline comparison: TP=8, PP=2 (16 GPUs total)