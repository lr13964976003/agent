# Phase 2: Detailed Methodology

## Problem Formulation

### Mathematical Model
Given:
- Model with n layers: L = {l₁, l₂, ..., lₙ}
- Cache capacity per device: C
- Goal: Partition into k disjoint groups P = {P₁, P₂, ..., Pₖ}

### Constraints
1. **Memory constraint**: S(Pᵢ) = Σ(lⱼ∈Pᵢ) size(lⱼ) ≤ C
2. **Execution order**: Layers assigned contiguously in original order
3. **Minimize k**: Minimize number of partitions or balance hardware utilization

## Memory Footprint Estimation

### Components per Layer
1. **Weights**: Parameter tensors
   - Calculation: weight_size(lⱼ) = num_parameters × datatype_size
   - BF16 precision: 2 bytes per parameter
   - Total model size: 30B parameters → 60GB total weights

2. **Activations**: Intermediate outputs
   - Depends on: batch_size × sequence_length × hidden_dimension
   - For dense model: batch_size = 128, seq_len = 10,000
   - Hidden dimensions vary by layer type

3. **Temporary Buffers**: Operator workspace
   - Derived from profiling or analytical models
   - Includes computation temporary storage

### Total Memory Calculation
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation

**Algorithm Steps:**
1. Initialize empty partition Pᵢ
2. Start from first layer l₁
3. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
4. **Termination condition**: If S(Pᵢ) + size(lⱼ) > C, finalize Pᵢ with layers {l_start, ..., lⱼ₋₁}
5. Start new partition Pᵢ₊₁ from layer lⱼ
6. Repeat until all layers assigned

**Properties:**
- Simple O(n) complexity
- Guarantees cache fit
- May create imbalanced partitions

### 3.2 Dynamic Programming (Optional)

**Objective:** Minimize maximum partition size while respecting cache constraint
- More balanced load distribution
- Reduces total number of partitions k
- Higher computational complexity O(n²C)

## Deployment Strategy

### Device Assignment
- Each partition Pᵢ assigned to separate accelerator card
- All weights pre-loaded into SRAM/L2 cache
- Activation and buffer memory pre-allocated within cache

### Execution Flow
1. **Intra-card execution**: Sequential layer processing within partition
2. **Inter-card communication**: Transfer intermediate outputs between partitions
3. **Minimize transfers**: Only communicate when crossing partition boundaries

### Edge Cases Handling
- **Single layer too large**: Apply intra-layer partitioning or model compression
- **Techniques**: Quantization, pruning, tensor parallelism within layer
- **Batch size adjustment**: Reduce activation memory footprint
- **Variable layer sizes**: Adjust partitioning heuristics for balanced utilization

## Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory hierarchy**: 
  - SRAM/L2 cache per GPU (capacity C to be specified)
  - HBM3 high-bandwidth memory
- **Interconnect**: NVLink for high-speed GPU-to-GPU communication
- **Precision**: BF16 (16-bit floating point)

## Model Architecture Details
- **Dense model**: 16-layer fully connected network
- **Layer configuration**:
  - Each layer: combination of attention + MLP blocks
  - Hidden dimension: varies by layer (calculated from 30B parameters across 16 layers)
  - MLP expansion: 4× hidden size (16,384 intermediate dimension)
  - Attention heads: 32 heads × 128 dimensions = 4,096 hidden size