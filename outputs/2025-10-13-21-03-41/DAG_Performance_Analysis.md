# DAG Performance Analysis: Cross-Node Expert Parallelism vs Baseline MoE

## Executive Summary

This analysis compares the computational runtime of two MoE (Mixture of Experts) model configurations:
1. **Proposed Cross-Node Expert Parallelism (EP=16)** - 16 GPUs with 1 expert per GPU
2. **Baseline MoE (TP=8, PP=2)** - 16 GPUs with tensor parallelism (TP=8) and pipeline parallelism (PP=2)

## Matrix Multiplication Operations Analysis

### 1. Proposed Cross-Node Expert Parallelism (EP=16)

#### Layer Structure per Transformer Block
Each of the 4 layers follows this pattern:
- Multi-Head Attention (MHA) with QKV projection
- Expert routing and parallel expert computation
- Feed-forward networks within experts

#### Matrix Multiplication Dimensions

**MHA QKV Linear Operations:**
- **Q projection**: [batch_size × seq_len, hidden_dim] × [hidden_dim, hidden_dim]
  - Dimensions: m=1024×10000=10,240,000, k=8192, n=8192
  - Get_Time(10240000, 8192, 8192)

- **K projection**: Same as Q projection
  - Get_Time(10240000, 8192, 8192)

- **V projection**: Same as Q projection  
  - Get_Time(10240000, 8192, 8192)

**Attention Computation:**
- **Q×K^T**: [batch_size × seq_len, d_k] × [d_k, seq_len]
  - After head split: d_k = 512, heads = 16
  - Per head: Get_Time(10240000, 512, 10000)
  - Total for all heads: 16 × Get_Time(10240000, 512, 10000)

- **Attention×V**: [batch_size × seq_len, seq_len] × [seq_len, d_k]
  - Per head: Get_Time(10240000, 10000, 512)
  - Total for all heads: 16 × Get_Time(10240000, 10000, 512)

**MHA Output Linear:**
- **Concatenated attention × W_O**: [batch_size × seq_len, hidden_dim] × [hidden_dim, hidden_dim]
  - Get_Time(10240000, 8192, 8192)

**Expert Feed-Forward Networks:**
Each expert contains:
- **Up projection**: [variable_tokens, hidden_dim] × [hidden_dim, 4×hidden_dim]
  - Get_Time(variable_tokens, 8192, 32768)
- **Down projection**: [variable_tokens, 4×hidden_dim] × [4×hidden_dim, hidden_dim]
  - Get_Time(variable_tokens, 32768, 8192)

**Gate Network:**
- **Gate projection**: [batch_size × seq_len, hidden_dim] × [hidden_dim, num_experts]
  - Get_Time(10240000, 8192, 16)

#### Longest Path Analysis
The critical path through the DAG follows:
1. **Input → MHA QKV → Attention → MHA Output → Residual Add → Layer Norm → Gate → Expert Routing → Expert Computation → Aggregation → Residual Add → Layer Norm**

**Sequential Operations per Layer:**
- MHA QKV: 3 × Get_Time(10240000, 8192, 8192)
- Attention: 16 × [Get_Time(10240000, 512, 10000) + Get_Time(10240000, 10000, 512)]
- MHA Output: Get_Time(10240000, 8192, 8192)
- Gate: Get_Time(10240000, 8192, 16)
- Expert computation (parallel across 16 GPUs): Max of expert workloads
- Expert FFN: Get_Time(variable_tokens, 8192, 32768) + Get_Time(variable_tokens, 32768, 8192)

**Total Longest Path (4 layers):**
```
4 × [3×Get_Time(10240000,8192,8192) + 16×(Get_Time(10240000,512,10000)+Get_Time(10240000,10000,512)) + Get_Time(10240000,8192,8192) + Get_Time(10240000,8192,16) + Expert_FFN_Time]
```

### 2. Baseline MoE (TP=8, PP=2)

#### Layer Structure with Tensor and Pipeline Parallelism
- **Stage 0**: Layers 0-1 on GPUs 0-7 (8-way TP)
- **Stage 1**: Layers 2-3 on GPUs 8-15 (8-way TP)
- Pipeline communication between stages

#### Matrix Multiplication Dimensions (Per TP Rank)

**MHA QKV Linear (per TP rank):**
- **Q projection**: [batch_size × seq_len, hidden_dim/8] × [hidden_dim/8, hidden_dim/8]
  - Dimensions: m=10240000, k=1024, n=1024
  - Get_Time(10240000, 1024, 1024)

**Attention Computation (per TP rank):**
- **Q×K^T**: [batch_size × seq_len, d_k/8] × [d_k/8, seq_len]
  - d_k = 512, so per head: d_k/8 = 64
  - Per head: Get_Time(10240000, 64, 10000)
  - Total for 2 heads per rank: 2 × Get_Time(10240000, 64, 10000)

- **Attention×V**: [batch_size × seq_len, seq_len] × [seq_len, d_k/8]
  - Per head: Get_Time(10240000, 10000, 64)
  - Total for 2 heads per rank: 2 × Get_Time(10240000, 10000, 64)

**All-Reduce Operations:**
- **Attention All-Reduce**: Communication overhead across 8 TP ranks
- **Expert Aggregation**: All-reduce across 8 experts per GPU

**Expert Feed-Forward Networks (per GPU):**
Each GPU hosts 8 experts:
- **Up projection**: [variable_tokens, hidden_dim] × [hidden_dim, 4×hidden_dim]
  - Get_Time(variable_tokens, 8192, 32768)
- **Down projection**: [variable_tokens, 4×hidden_dim] × [4×hidden_dim, hidden_dim]
  - Get_Time(variable_tokens, 32768, 8192)

#### Longest Path Analysis

**Critical Path through Pipeline:**
1. **Stage 0**: 2 layers × (MHA + Expert computation)
2. **Pipeline communication**: Between stages
3. **Stage 1**: 2 layers × (MHA + Expert computation)

**Sequential Operations:**
- **Stage 0 (2 layers):**
  - 2 × [3×Get_Time(10240000,1024,1024) + 2×(Get_Time(10240000,64,10000)+Get_Time(10240000,10000,64)) + Get_Time(10240000,1024,1024)]

- **Pipeline communication**: Communication latency between stages

- **Stage 1 (2 layers):**
  - Same computation as Stage 0

**Total Longest Path:**
```
2×[3×Get_Time(10240000,1024,1024) + 2×(Get_Time(10240000,64,10000)+Get_Time(10240000,10000,64)) + Get_Time(10240000,1024,1024)] 
+ Pipeline_Communication 
+ 2×[3×Get_Time(10240000,1024,1024) + 2×(Get_Time(10240000,64,10000)+Get_Time(10240000,10000,64)) + Get_Time(10240000,1024,1024)]
```

## Parallelism Analysis

### Proposed Method (EP=16)
- **Expert Parallelism**: 16 experts distributed across 16 GPUs
- **Parallel Computation**: All 16 experts run simultaneously
- **Load Balancing**: Perfect distribution (1 expert per GPU)
- **Communication**: Minimal - only token routing and aggregation

### Baseline Method (TP=8, PP=2)
- **Tensor Parallelism**: 8-way split within each stage
- **Pipeline Parallelism**: 2 stages (4 layers per stage)
- **Expert Distribution**: 8 experts per GPU (64 total experts)
- **Communication**: 
  - All-reduce for tensor parallelism
  - Pipeline communication between stages
  - Expert aggregation across 8 experts per GPU

## Performance Comparison Summary

### Longest Path Computation

**Proposed EP=16:**
- **Critical Path**: 4 sequential layers
- **Parallel Expert Computation**: 16 experts in parallel
- **Matrix Multiplication Bottleneck**: MHA operations dominate
- **Communication Overhead**: Minimal token routing

**Baseline TP=8, PP=2:**
- **Critical Path**: 2 layers + pipeline communication + 2 layers
- **Parallel Expert Computation**: 8 experts per GPU (but 8-way TP reduces individual matrix sizes)
- **Communication Overhead**: 
  - All-reduce for tensor parallelism
  - Pipeline stage communication
  - Expert aggregation across 8 experts

### Key Insights

1. **Matrix Size Trade-off**: Proposed method uses full matrix dimensions (8192×8192) while baseline uses reduced dimensions (1024×1024) due to tensor parallelism

2. **Parallelism Granularity**: 
   - Proposed: Coarse-grained expert parallelism
   - Baseline: Fine-grained tensor parallelism + pipeline parallelism

3. **Communication Patterns**:
   - Proposed: Minimal communication (token routing)
   - Baseline: Heavy communication (all-reduce + pipeline)

4. **Load Distribution**:
   - Proposed: Perfect load balancing (1 expert per GPU)
   - Baseline: Uneven load (8 experts per GPU, potential contention)

## Final Runtime Representation

### Proposed EP=16 Longest Path:
```
Total_Runtime = 4 × [
    3×Get_Time(10240000, 8192, 8192) +           // QKV projections
    16×(Get_Time(10240000, 512, 10000) +         // Q×K^T per head
         Get_Time(10240000, 10000, 512)) +       // Attention×V per head
    Get_Time(10240000, 8192, 8192) +             // MHA output
    Get_Time(10240000, 8192, 16) +               // Gate network
    Get_Time(avg_tokens_per_expert, 8192, 32768) +  // Expert up-projection
    Get_Time(avg_tokens_per_expert, 32768, 8192)    // Expert down-projection
]
```

### Baseline TP=8, PP=2 Longest Path:
```
Total_Runtime = 2 × [
    3×Get_Time(10240000, 1024, 1024) +           // QKV projections (TP=8)
    2×(Get_Time(10240000, 64, 10000) +           // Q×K^T per head (TP=8)
         Get_Time(10240000, 10000, 64)) +        // Attention×V per head (TP=8)
    Get_Time(10240000, 1024, 1024) +             // MHA output (TP=8)
    8×(Get_Time(avg_tokens_per_expert, 8192, 32768) +  // Expert up-projection
        Get_Time(avg_tokens_per_expert, 32768, 8192))   // Expert down-projection
] + 
Pipeline_Communication_Latency + 
2 × [Same computation as above]
```

Where `avg_tokens_per_expert` = (1024 × 10000) / 16 = 640,000 tokens per expert (assuming uniform distribution).

## Conclusion

The proposed cross-node expert parallelism achieves better load balancing and reduces communication overhead compared to the baseline tensor+pipeline parallelism approach. The longest path computation shows that while individual matrix operations are larger in the proposed method, the parallel execution of experts across 16 GPUs and minimal communication overhead result in potentially better overall performance.