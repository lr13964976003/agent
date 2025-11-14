# Phase 1: Keypoints Extraction

## Paper: Layer-wise Deployment Strategy for Large Neural Networks

### Core Problem Addressed
- **Challenge**: Large neural networks exceed on-chip memory (SRAM/L2 cache) capacity, causing expensive off-chip memory accesses
- **Proposed Solution**: Layer-wise partitioning to fit model partitions entirely within fast on-chip memory

### Key Technical Contributions

#### 1. Problem Formulation
- **Goal**: Partition n layers L = {l₁, l₂, ..., lₙ} into k disjoint groups P = {P₁, P₂, ..., Pₖ}
- **Constraint**: Each partition Pᵢ must satisfy S(Pᵢ) ≤ C (cache capacity)
- **Objective**: Minimize k (number of partitions) while preserving execution order

#### 2. Memory Footprint Estimation
- **Formula**: size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
- **Weight Size**: Parameters × datatype size (BF16 = 2 bytes)
- **Activation Size**: batch_size × sequence_length × hidden_dimension × 2
- **Buffer Size**: Operator workspace from profiling

#### 3. Partitioning Strategy
- **Primary Algorithm**: Greedy Layer Aggregation (O(n) complexity)
- **Alternative**: Dynamic Programming for balanced partitions
- **Constraint Satisfaction**: Guarantees each partition fits cache capacity

#### 4. Deployment Method
- **Process**: Load entire partition into SRAM/L2 cache → Execute sequentially → Transfer outputs between partitions
- **Key Benefit**: Minimize inter-device communication by keeping computations local

### Critical Model Specifications (Addressing Inconsistencies)

#### From Original Paper Analysis:
- **Model Type**: Dense fully connected network
- **Layer Count**: 4 layers (per original paper, NOT 16 as in previous submission)
- **Model Size**: 30B parameters total
- **Precision**: BF16 (2 bytes per parameter)
- **Configuration**: 32 heads, 128 head dimension, 16384 MLP hidden size

#### Memory Calculations (Corrected):
- **Total Model**: 30B parameters × 2 bytes = 60 GB total
- **Per Layer**: 60 GB ÷ 4 layers = 15 GB per layer
- **Activation Size**: 128 × 10000 × 16384 × 2 = 39.06 GB per layer
- **Cache Capacity**: 50 MB per H100 GPU L2 cache

### Performance Metrics
- **Baseline**: TP=8, PP=2 using 16 GPUs
- **Results**: 20% TPS improvement (12,800 → 15,360 tokens/s)
- **Latency Reduction**: 17% TPOT improvement (0.078 → 0.065 ms)

### Key Insights for Deployment
1. **Cache-Conscious Design**: Explicit consideration of cache limits during partitioning
2. **Working Set Concept**: Only active layer data needs to be in cache (not entire activations)
3. **Streaming Strategy**: Activations can be processed in chunks to fit cache constraints
4. **Scalability**: Linear scaling with available devices, maintaining high efficiency

### Addressing Previous Errors
- **Layer Count**: Corrected to 4 layers from original paper
- **Memory Math**: Clarified that working set (weights + partial activations) fits cache, not full activations
- **Cache Utilization**: Re-calculated based on actual working set sizes rather than full layer data