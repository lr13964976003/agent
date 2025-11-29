# DAG Validation Issues

## Critical Issues Found

### 1. Inconsistent Expert Parallelism Implementation
- **Issue**: Only GPU 4 shows proper expert splitting/aggregation pattern
- **Affected Nodes**: 
  - `gpu4_split_experts`, `gpu4_expert1`, `gpu4_expert2`, `gpu4_aggregate_experts`
- **Problem**: GPUs 0-3 and 5-7 have simple `gpuX_ffn_experts` nodes without the split/aggregate pattern
- **Impact**: Inconsistent parallel strategy across GPUs

### 2. Communication Node GPU Format Inconsistency
- **Issue**: Communication nodes use different GPU format than computation nodes
- **Affected Nodes**: All communication nodes (`comm_gpu0_gpu1`, `comm_gpu1_gpu2`, etc.)
- **Current Format**: "GPU: 0 → 1" 
- **Expected Format**: Should match computation nodes format "GPU: X"
- **Impact**: Format inconsistency could cause parsing issues

### 3. Highly Similar Repeated Modules
- **Issue**: Identical layer structures repeated across GPUs
- **Affected GPUs**: GPUs 0, 1, 2, 3, 5, 6, 7 have nearly identical structures
- **Specific Repeats**:
  - All have identical MHA patterns: `layernorm → mha_qkv → mha_attention → mha_out → residual`
  - All have identical FFN patterns: `layernorm → ffn_gate → ffn_experts → ffn_out → residual`
- **Impact**: Violates "no highly similar repeated modules" requirement

### 4. DOT Style Definition Conflicts
- **Issue**: Multiple conflicting node style definitions
- **Location**: Lines with `node [fillcolor=..., shape=..., style=filled]`
- **Problem**: Later definitions override earlier ones, causing inconsistent styling
- **Impact**: Could cause rendering issues and unclear component identification

### 5. Missing Output Node GPU Specification
- **Issue**: Output node doesn't specify GPU assignment format
- **Affected Node**: `output`
- **Current**: Shows "GPU: Host" 
- **Problem**: Inconsistent with other node formats

## Required Modifications

1. **Standardize Expert Parallelism**: Either implement split/aggregate pattern for all GPUs or remove from GPU 4
2. **Fix Communication Node Format**: Change "GPU: X → Y" to consistent format
3. **Reduce Module Repetition**: Add variations or consolidate repeated patterns
4. **Fix Style Definitions**: Use single consistent node style definition
5. **Standardize Output Node**: Use consistent GPU format specification

## Severity: HIGH
These issues violate multiple requirements and could cause deployment problems.