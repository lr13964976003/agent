# Nodes Requiring Modification

## Critical Issues Found

### ❌ Missing Connections (REQUIRED FIXES)

The following nodes violate the DAG requirement that "except for the output node, each node must output to another node":

1. **`layer1_mla_dp0_tp1`**
   - Current: No outgoing edges
   - Required: Must connect to appropriate downstream computation
   - Suggested fix: Connect to `layer1_ffn_dp0_tp1` (following the pattern of `layer1_mla_dp0_tp0 → layer1_ffn_dp0_tp0`)

2. **`layer1_mla_dp1_tp0`**
   - Current: No outgoing edges
   - Required: Must connect to appropriate downstream computation
   - Suggested fix: Connect to `layer1_ffn_dp1_tp0` (following the established pattern)

3. **`layer1_mla_dp1_tp1`**
   - Current: No outgoing edges
   - Required: Must connect to appropriate downstream computation
   - Suggested fix: Connect to `layer1_ffn_dp1_tp1` (following the established pattern)

### ⚠️ Excessive Repetition (OPTIMIZATION SUGGESTION)

While not technically incorrect, the following pattern creates 192 nearly identical sub-graphs:

- **Expert chains** across layers 4 and 61:
  - `layer4_gate_dp*_tp*_ep*` → `layer4_dispatch_dp*_tp*_ep*` → `layer4_expert_mlp_dp*_tp*_ep*` → `layer4_expert_proj_dp*_tp*_ep*` → `layer4_expert_allreduce_dp*_tp*_ep*` → `layer4_aggregate_dp*_tp*_ep*`
  - `layer61_gate_dp*_tp*_ep*` → `layer61_dispatch_dp*_tp*_ep*` → `layer61_expert_mlp_dp*_tp*_ep*` → `layer61_expert_proj_dp*_tp*_ep*` → `layer61_expert_allreduce_dp*_tp*_ep*` → `layer61_aggregate_dp*_tp*_ep*`

**Total**: 64 experts × 2 layers = 128 expert sub-graphs, plus similar patterns in layer 4

### Modification Requirements

**MANDATORY FIXES** (DAG will be incorrect without these):
1. Add missing edges for the three orphaned MLA nodes
2. Ensure proper data flow continuation

**RECOMMENDED OPTIMIZATIONS** (for better graph clarity):
1. Consider using subgraph/cluster notation for repeated expert patterns
2. Evaluate if fewer expert instances with proper scheduling could represent the same computation

### Specific Edge Additions Required

```dot
layer1_mla_dp0_tp1 -> layer1_ffn_dp0_tp1
layer1_mla_dp1_tp0 -> layer1_ffn_dp1_tp0  
layer1_mla_dp1_tp1 -> layer1_ffn_dp1_tp1
```

These additions will complete the data flow and ensure all non-output nodes have proper outgoing connections.