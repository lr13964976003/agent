# Phase 1: Key Points Extraction

## Key Innovation Points

### 1. Dual-Level Partitioning Scheme
- **First Level**: Head-wise partitioning - splits total h heads into n groups, each containing h/n heads
- **Second Level**: Intra-head dimension partitioning - splits each head's feature dimension d into m segments, each of size d/m
- **Total Partitions**: m × n partitions enabling deployment on m × n devices

### 2. Mathematical Formulations
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- Weight matrices: W_Q, W_K, W_V ∈ ℝ^(D×D)
- Head dimension: d = D/h (per head)
- Partition parameters: h_g = h/n (heads per group), d_s = d/m (slice dimension per partition)
- Each partition: W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g) for i∈[1,n], j∈[1,m]

### 3. Advantages Over Conventional Methods
- **Scalability**: Supports m×n devices beyond head count limitations
- **Load Balancing**: Even workload distribution across both dimensions
- **Memory Efficiency**: Each device stores only 1/(m×n) of parameters and activations
- **Communication Efficiency**: Hierarchical partitioning reduces synchronization bandwidth

### 4. Aggregation Process
- **Step 1**: Concatenate dimension slices within each head group
- **Step 2**: Concatenate head groups to reconstruct full MHA output
- **Communication**: Only required for intra-group concatenation

### 5. Performance Results
- **Throughput Improvement**: 31.7% (1.2M → 1.58M tokens/sec)
- **Overhead Reduction**: 37.1% (0.35ms → 0.22ms TPOT)
- **Configuration**: 16-layer dense transformer, 16 H100 GPUs, FP16 precision