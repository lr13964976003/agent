# Nodes Requiring Modification

## Critical Issue Found

### Node: `gate2_gpu116`

**Problem**: GPU index inconsistency

**Current Definition**:
```
gate2_gpu116 [label="Expert Gate 2\nGPU: 116\nInput: [batch_size=1, seq_len=2048, hidden_dim=7168]\nOutput: [batch_size=1, seq_len=2048, num_experts=16]" fillcolor=lightblue shape=rectangle]
```

**Required Fix**:
- Change `GPU: 116` to `GPU: 0`
- Change node identifier from `gate2_gpu116` to `gate2_gpu0` for consistency

**Reason**: The GPU assignment should follow the same pattern as the first MoE layer where the gate is assigned to GPU 0, not GPU 116. This ensures consistency in the parallel strategy and proper GPU allocation.

**File Location**: `moe_deployment_dag.dot` line 179

## Impact
This modification ensures:
- Consistent GPU numbering scheme
- Proper Expert Parallelism (EP=16) allocation
- Maintains optimal parallel strategy across all layers