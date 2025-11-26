# Nodes Requiring Modification

## Critical Issues Found

### 1. Incomplete Expert Coverage
The DAG only includes 16 experts but the gating network specifies 32 experts. Missing experts:
- expert_16_gate through expert_31_gate
- expert_16_expert through expert_31_expert  
- expert_16_multiply through expert_31_multiply

### 2. Incomplete Tensor Specifications
All existing expert nodes have "?" placeholders that need to be replaced with actual dimensions:

#### Expert Gate Nodes (All 16 existing + 16 missing):
- `expert_0_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_1_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_2_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_3_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_4_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_5_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_6_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_7_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_8_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_9_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_10_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_11_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_12_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_13_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_14_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_15_gate`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len

#### Expert Expert Nodes (All 16 existing + 16 missing):
- `expert_0_expert`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- `expert_1_expert`: Input: `[batch_size=?, seq_len=?, heads=2048]` → Need actual batch_size, seq_len
- [Pattern continues for all expert nodes...]

#### Expert Multiply Nodes (All 16 existing + 16 missing):
- `expert_0_multiply`: Input: `[batch_size=?, seq_len=?, heads=7168], [batch_size=?, seq_len=?, heads=2048]` → Need actual dimensions
- `expert_1_multiply`: Input: `[batch_size=?, seq_len=?, heads=7168], [batch_size=?, seq_len=?, heads=2048]` → Need actual dimensions
- [Pattern continues for all multiply nodes...]

### 3. Missing GPU Assignments
Need to add expert nodes for GPUs 16-31 to match the 32-expert requirement.

### 4. Edge Connection Issues
Missing edges for experts 16-31:
- `token_scatter -> expert_16_gate` through `token_scatter -> expert_31_gate`
- `expert_16_multiply -> token_gather` through `expert_31_multiply -> token_gather`
- `routing -> expert_16_gate` through `routing -> expert_31_gate`

## Summary
**Total nodes to modify/add: 96 nodes**
- 48 existing expert nodes need tensor shape completion
- 48 additional expert nodes need to be created (16 experts × 3 node types each)

This DAG is **NOT READY** for production use.