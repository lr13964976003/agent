# Phase 2: Methodology Extraction

## Methodology: Layer-wise Deployment Strategy

### 1. Problem Formulation

**Given:** A large model composed of *n* layers L = {l₁, l₂, ..., lₙ}

**Objective:** Partition these layers into *k* disjoint groups P = {P₁, P₂, ..., Pₖ}, each assigned to a separate hardware accelerator card, such that:

- **Memory constraint:** The memory footprint of each group Pᵢ does not exceed the capacity C of the SRAM or L2 cache available on the corresponding card
- **Execution order preservation:** Layers are assigned contiguously in the original order
- **Hardware utilization:** The number of partitions *k* is minimized or balanced to maximize hardware utilization

**Formal constraint:** For each partition Pᵢ, the size S(Pᵢ) satisfies:
```
S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C
```

### 2. Memory Footprint Estimation

The memory footprint of each layer includes three components:

**Components:**
- **Weights:** Parameter tensors stored for the layer
- **Activations:** Intermediate outputs needed during inference or training
- **Temporary Buffers:** Workspace memory required by operators during computation

**Calculation formula:**
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

**Detailed calculations:**
- **weight_size**: Based on number of parameters × datatype size (FP16 = 2 bytes)
- **activation_size**: Output feature map dimensions × batch size
- **buffer_size**: Derived from profiling or analytical models of operator requirements

**Estimation timing:** Can be performed statically prior to deployment or dynamically profiled for accuracy

### 3. Partitioning Algorithms

#### 3.1 Greedy Layer Aggregation Algorithm

**Process:**
1. Initialize empty partition Pᵢ
2. Iteratively add subsequent layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If adding lⱼ causes S(Pᵢ) > C, finalize Pᵢ with layers {l_start, ..., l_{j-1}}
4. Start new partition P_{i+1} beginning from layer lⱼ
5. Repeat until all layers are assigned

**Properties:** Simple, efficient, guarantees cache fit for each partition

#### 3.2 Dynamic Programming for Balanced Partitions (Optional)

**Purpose:** Achieve more balanced load and minimize number of partitions
**Approach:** DP algorithm optimizes partition boundaries to minimize maximum partition size while respecting cache capacity constraint

### 4. Deployment Strategy

**Post-partitioning steps:**
1. **Memory allocation:** Load all weights and pre-allocate activation and buffer memory within SRAM or L2 cache
2. **Sequential execution:** Execute layers sequentially on assigned card
3. **Communication minimization:** Transfer intermediate outputs only when passing data between partitions on different cards

### 5. Handling Edge Cases

**Single layer exceeding capacity C:**
- Apply intra-layer partitioning
- Use model compression techniques (quantization, pruning)

**Activation memory optimization:**
- Tune batch size to reduce activation memory footprint
- Adjust partitioning heuristics for models with variable layer sizes

### 6. Summary of Advantages

- **Reduced memory access latency:** Minimize off-chip DRAM accesses by fitting partitions in SRAM/L2 cache
- **Improved throughput:** Faster memory access and parallel execution on multiple cards
- **Scalability:** Adaptable to varying model sizes and hardware configurations