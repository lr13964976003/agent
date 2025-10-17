# Phase 2: Methodology Extraction

## 1. Problem Formulation
Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- Memory footprint of each Pᵢ does not exceed cache capacity C
- Layers assigned contiguously in original order
- Minimize number of partitions k for maximum hardware utilization

**Mathematical constraint:**
S(Pᵢ) = Σₗⱼ∈Pᵢ size(lⱼ) ≤ C

## 2. Memory Footprint Estimation
For each layer lⱼ, calculate:
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

### Component Calculations:
- **weight_size**: parameters × datatype size (FP16 = 2 bytes)
- **activation_size**: output feature map dimensions × batch size
- **buffer_size**: operator workspace requirements (profiled or analytically determined)

## 3. Partitioning Algorithms

### 3.1 Greedy Layer Aggregation Algorithm
**Process:**
1. Initialize empty partition Pᵢ
2. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. When S(Pᵢ) + size(lⱼ) > C, finalize Pᵢ with layers {l_start, ..., lⱼ₋₁}
4. Start new partition Pᵢ₊₁ from layer lⱼ
5. Repeat until all layers assigned

**Properties:** Simple, efficient, guarantees cache-fit partitions

### 3.2 Dynamic Programming Approach (Optional)
**Objective:** Minimize maximum partition size while respecting cache capacity C
**Use case:** When balanced load distribution is critical

## 4. Deployment Strategy
**Execution flow:**
1. **Pre-deployment:** Calculate layer sizes using estimation formulas
2. **Partitioning:** Apply chosen algorithm to create layer groups
3. **Loading:** Load entire partition (weights + activations + buffers) into SRAM/L2 cache
4. **Execution:** Process layers sequentially on assigned card
5. **Communication:** Transfer intermediate outputs only between partitions on different cards

## 5. Memory Hierarchy Optimization
- **Target memory:** SRAM or L2 cache of accelerator cards
- **Avoid:** Off-chip DRAM access during layer execution
- **Benefit:** Significant latency reduction and energy efficiency

## 6. Edge Case Handling
- **Single layer exceeds C:** Apply intra-layer partitioning or model compression (quantization/pruning)
- **Variable layer sizes:** Adjust partitioning heuristics to prevent under-utilization
- **Batch size impact:** Tune batch size to reduce activation memory footprint

## 7. Implementation Requirements
- **Static analysis:** Pre-deployment size estimation
- **Dynamic profiling:** Runtime accuracy adjustment
- **Hardware abstraction:** Adaptable to different cache capacities C
- **Contiguous allocation:** Preserve layer execution order