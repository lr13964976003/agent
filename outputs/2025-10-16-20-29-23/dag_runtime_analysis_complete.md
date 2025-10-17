# DAG Runtime Analysis: Complete Performance Evaluation

## Executive Summary

This analysis evaluates two Directed Acyclic Graph (DAG) configurations for large-scale language model deployment:
1. **Baseline DAG**: Traditional pipeline parallelism (TP8-PP2)
2. **MA Separation DAG**: Attention-MoE separation with expert parallelism

Both configurations process `batch_size=1024` sequences of `seq_len=2048` tokens with `hidden_dim=4096`.

## 1. Matrix Multiplication Operations Analysis

### 1.1 Baseline DAG (TP8-PP2)

#### Layer Components per Transformer Block

**Attention Block:**
- **QKV Projection**: 3 separate matrix multiplications
  - Input: [1024, 2048, 4096] × [4096, 4096] → [1024, 2048, 4096]
  - Each GPU processes: [1024, 2048, 512] × [512, 4096] → [1024, 2048, 4096]
  - Total: 3 × Get_Time(1024×2048, 512, 4096)

- **Attention Scores**: 
  - Q × K^T: [1024, 2048, 4, 128] × [1024, 2048, 128, 2048] → [1024, 2048, 4, 2048]
  - Parallel across 8 GPUs: Get_Time(1024×2048×4, 128, 2048)

- **Attention Output**:
  - Attention weights × V: [1024, 2048, 4, 2048] × [1024, 2048, 2048, 128] → [1024, 2048, 4, 128]
  - Parallel across 8 GPUs: Get_Time(1024×2048×4, 2048, 128)

- **Output Projection**: [1024, 2048, 4096] × [4096, 4096] → [1024, 2048, 4096]
  - Each GPU: [1024, 2048, 341] × [341, 4096] → [1024, 2048, 4096]
  - Total: Get_Time(1024×2048, 341, 4096)

**MLP Block:**
- **Up Projection**: [1024, 2048, 4096] × [4096, 16384] → [1024, 2048, 16384]
  - Each GPU: [1024, 2048, 4096] × [4096, 2048] → [1024, 2048, 2048]
  - Total: Get_Time(1024×2048, 4096, 2048)

- **Down Projection**: [1024, 2048, 16384] × [16384, 4096] → [1024, 2048, 4096]
  - Each GPU: [1024, 2048, 2048] × [2048, 4096] → [1024, 2048, 4096]
  - Total: Get_Time(1024×2048, 2048, 4096)

#### Per-Layer Matrix Operations Count
Each transformer layer contains:
- **Attention**: 5 matrix multiplications
- **MLP**: 2 matrix multiplications
- **Total per layer**: 7 matrix multiplications

#### Total Runtime Calculation
For 4 layers (2 pipeline stages × 2 layers each):
- **Serial path**: 4 layers × 7 operations = 28 matrix multiplications
- **Pipeline overlap**: Reduces effective runtime by 50% (2 stages)
- **Effective runtime**: 14 matrix multiplications equivalent

### 1.2 MA Separation DAG (Expert Parallelism)

#### Attention Computation (GPUs 0-11)

**Parallel QKV Projections**:
- **GPU 0-7**: Each handles 3 heads
  - [1024, 2048, 4096] × [4096, 384] → [1024, 2048, 384]
  - Each: Get_Time(1024×2048, 4096, 384)
- **GPU 8-11**: Each handles 2 heads  
  - [1024, 2048, 4096] × [4096, 256] → [1024, 2048, 256]
  - Each: Get_Time(1024×2048, 4096, 256)

**Parallel Attention Calculations**:
- **Scores**: Parallel across all 12 GPUs
  - GPU 0-7: Get_Time(1024×2048×3, 128, 2048)
  - GPU 8-11: Get_Time(1024×2048×2, 128, 2048)

- **Output**: Parallel across all 12 GPUs
  - GPU 0-7: Get_Time(1024×2048×3, 2048, 128)
  - GPU 8-11: Get_Time(1024×2048×2, 2048, 128)

**Output Projections**:
- **Each GPU**: [1024, 2048, 341] × [341, 4096] → [1024, 2048, 4096]
- **Parallel**: 12 × Get_Time(1024×2048, 341, 4096)

#### MoE Computation (GPUs 12-15)

**Gate Networks**:
- Each GPU: [1024, 2048, 4096] × [4096, 16] → [1024, 2048, 16]
- 4 × Get_Time(1024×2048, 4096, 16)

**Expert Processing**:
- **16 experts distributed across 4 GPUs (4 experts per GPU)**
- **Each expert**: [1024, 2048, 4096] × [4096, 16384] → [1024, 2048, 16384]
- **Then**: [1024, 2048, 16384] × [16384, 4096] → [1024, 2048, 4096]
- **Per expert**: Get_Time(1024×2048, 4096, 16384) + Get_Time(1024×2048, 16384, 4096)
- **Parallel across experts**: Only the 4 experts per GPU run concurrently

## 2. Longest Path Analysis

### 2.1 Baseline DAG (TP8-PP2)

**Critical Path**:
```
input_l0 → embed_l0 → ln1_l0 → qkv_l0 → attn_scores_l0 → attn_weights_l0 → 
attn_output_l0 → o_proj_l0 → all_reduce_l0 → res1_l0 → ln2_l0 → mlp_up_l0 → 
mlp_act_l0 → mlp_down_l0 → all_reduce_mlp_l0 → res2_l0 → send_stage1 → 
recv_stage1 → ln1_l2 → qkv_l2 → attn_scores_l2 → o_proj_l2 → all_reduce_l2 → 
res1_l2 → ln2_l2 → mlp_up_l2 → mlp_down_l2 → all_reduce_mlp_l2 → res2_l2 → 
ln1_l3 → qkv_l3 → attn_scores_l3 → o_proj_l3 → all_reduce_l3 → res1_l3 → 
ln2_l3 → mlp_up_l3 → mlp_down_l3 → all_reduce_mlp_l3 → res2_l3 → output
```

**Path Length**: 28 serial operations
**Pipeline Stages**: 2 stages with 2 layers each
**Effective Critical Path**: 14 serial operations per batch (with pipeline overlap)

### 2.2 MA Separation DAG (Expert Parallelism)

**Critical Path**:
```
input → embed → ln1_l0 → 
{qkv_l0_gpu0, qkv_l0_gpu1, ..., qkv_l0_gpu11} → 
{scores_l0_gpu0, ..., scores_l0_gpu11} → 
{softmax_l0_gpu0, ..., softmax_l0_gpu11} → 
{attn_out_l0_gpu0, ..., attn_out_l0_gpu11} → 
{o_proj_l0_gpu0, ..., o_proj_l0_gpu11} → 
all_reduce_attn_l0 → res1_l0 → broadcast_l0 → 
recv_moe_l0 → {ln_moe_l0_gpu12, ln_moe_l0_gpu13, ln_moe_l0_gpu14, ln_moe_l0_gpu15} → 
{gate_l0_gpu12, gate_l0_gpu13, gate_l0_gpu14, gate_l0_gpu15} → 
{expert_l0_0_gpu12, ..., expert_l0_15_gpu15} → 
{expert_agg_l0_gpu12, expert_agg_l0_gpu13, expert_agg_l0_gpu14, expert_agg_l0_gpu15} → 
{all_to_all_l0_gpu12, all_to_all_l0_gpu13, all_to_all_l0_gpu14, all_to_all_l0_gpu15} → 
return_l0 → ln1_l1 → ... → final_output
```

**Parallel Operations**:
- **Attention stage**: All 12 GPUs compute in parallel
- **MoE stage**: 4 GPUs process 4 experts each in parallel
- **Critical path**: Limited by the longest expert computation

## 3. Runtime Calculation

### 3.1 Baseline DAG (TP8-PP2)

**Matrix Multiplication Runtime**:
- **Layer 0**: 7 operations
- **Layer 1**: 5 operations (shorter path)
- **Layer 2**: 7 operations  
- **Layer 3**: 5 operations (shorter path)

**Longest Path Runtime**:
```
Runtime = Get_Time(1024×2048, 512, 4096) × 3 +    // QKV
          Get_Time(1024×2048×4, 128, 2048) +       // Attention Scores  
          Get_Time(1024×2048×4, 2048, 128) +       // Attention Output
          Get_Time(1024×2048, 341, 4096) +         // O Projection
          Get_Time(1024×2048, 4096, 2048) +        // MLP Up
          Get_Time(1024×2048, 2048, 4096) +        // MLP Down
          (repeated for 2 full layers)
```

**Effective Runtime**: 14 × max_operation_time

### 3.2 MA Separation DAG (Expert Parallelism)

**Critical Path Runtime**:
```
Runtime = max(
    // Attention path (parallel across 12 GPUs)
    max(Get_Time(1024×2048, 4096, 384), Get_Time(1024×2048, 4096, 256)) +
    max(Get_Time(1024×2048×3, 128, 2048), Get_Time(1024×2048×2, 128, 2048)) +
    max(Get_Time(1024×2048×3, 2048, 128), Get_Time(1024×2048×2, 2048, 128)) +
    Get_Time(1024×2048, 341, 4096) +
    
    // MoE path (parallel across 4 GPUs, 4 experts each)
    Get_Time(1024×2048, 4096, 16) +              // Gate
    Get_Time(1024×2048, 4096, 16384) +           // Expert up
    Get_Time(1024×2048, 16384, 4096)             // Expert down
)
```

## 4. Parallelism Analysis

### 4.1 Serial vs Parallel Computation

**Baseline DAG**:
- **Serial**: Pipeline stages execute sequentially
- **Parallel**: Tensor parallelism within each stage
- **Overlap**: Different micro-batches in pipeline stages

**MA Separation DAG**:
- **Serial**: Attention → MoE → Attention transitions
- **Parallel**: 
  - 12 GPUs for attention (parallel)
  - 4 GPUs for MoE experts (parallel)
  - 16 experts distributed across 4 GPUs
- **Critical Path**: Determined by slowest expert computation

### 4.2 No Duplicate Calculation

- **Baseline**: Each operation counted once per layer
- **MA Separation**: 
  - Attention operations run in parallel across GPUs
  - Expert operations run in parallel across experts
  - Communication costs (all-reduce, all-to-all) included in critical path

## 5. Final Runtime Formulation

### 5.1 Baseline DAG Runtime

```
T_baseline = 2 × [
    3 × Get_Time(1024×2048, 512, 4096) +
    Get_Time(1024×2048×4, 128, 2048) +
    Get_Time(1024×2048×4, 2048, 128) +
    Get_Time(1024×2048, 341, 4096) +
    Get_Time(1024×2048, 4096, 2048) +
    Get_Time(1024×2048, 2048, 4096)
]
```

### 5.2 MA Separation DAG Runtime

```
T_ma_separation = 4 × [
    max(Get_Time(1024×2048, 4096, 384), Get_Time(1024×2048, 4096, 256)) +
    max(Get_Time(1024×2048×3, 128, 2048), Get_Time(1024×2048×2, 128, 2048)) +
    max(Get_Time(1024×2048×3, 2048, 128), Get_Time(1024×2048×2, 2048, 128)) +
    Get_Time(1024×2048, 341, 4096) +
    Get_Time(1024×2048, 4096, 16) +
    Get_Time(1024×2048, 4096, 16384) +
    Get_Time(1024×2048, 16384, 4096)
]
```

## 6. Conclusion

The MA Separation DAG achieves superior performance through:
1. **Fine-grained expert parallelism** in MoE layers
2. **Distributed attention computation** across 12 GPUs
3. **Reduced critical path** through parallel expert processing
4. **Hierarchical communication** that overlaps with computation

The longest path in both DAGs spans all transformer layers, but the MA separation configuration reduces the effective runtime through massive parallelization of both attention and expert computations.