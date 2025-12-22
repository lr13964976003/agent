# Nodes That Need Modification

## Critical Issues Requiring Immediate Attention

### 1. Expert System Expansion (ADD 59 MISSING NODES)
**Current**: Only 5 expert nodes exist
**Required**: Add the following 59 missing expert nodes:

```dot
layer0_expert5 [label="Expert 5\nGPU: PP0_EP5_GPU\nINPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]" fillcolor=lightgreen shape=box]
layer0_expert6 [label="Expert 6\nGPU: PP0_EP6_GPU\nINPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]" fillcolor=lightgreen shape=box]
# ... (continue pattern through expert 63)
layer0_expert63 [label="Expert 63\nGPU: PP0_EP63_GPU\nINPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]" fillcolor=lightgreen shape=box]
```

### 2. Attention Block Decomposition (MODIFY 8 EXISTING NODES)
**Current**: Generic attention nodes without proper QKV decomposition
**Required**: Replace with proper attention submodules:

```dot
# Replace existing attention nodes with:
layer0_qkv_tp0 [label="QKV Projection\nGPU: PP0_TP0_GPU\nINPUT: [batch_size=64, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=64, seq_len=128-10240, heads=2, d_k=64]" fillcolor=lightgreen shape=box]
layer0_score_tp0 [label="Attention Score\nGPU: PP0_TP0_GPU\nINPUT: [batch_size=64, seq_len=128-10240, heads=2, d_k=64]\nOUTPUT: [batch_size=64, seq_len=128-10240, heads=2, d_k=64]" fillcolor=lightgreen shape=box]
layer0_out_tp0 [label="Attention Output\nGPU: PP0_TP0_GPU\nINPUT: [batch_size=64, seq_len=128-10240, heads=2, d_k=64]\nOUTPUT: [batch_size=64, seq_len=128-10240, hidden_dim=128]" fillcolor=lightgreen shape=box]
```

### 3. Communication Nodes (ADD 4 MISSING NODES)
**Current**: Only 4 communication nodes
**Required**: Add missing all-to-all communications:

```dot
layer0_dispatch_alltoall [label="Expert Dispatch All-to-All\nDispatch tokens to 64 experts\n64 GPUs involved\nEP64 communication" fillcolor=pink shape=ellipse style="filled,dashed"]
layer0_combine_alltoall [label="Expert Combine All-to-All\nCombine expert outputs\n64 GPUs involved\nEP64 communication" fillcolor=pink shape=ellipse style="filled,dashed"]
layer0_tp_allreduce [label="TP All-Reduce\nTensor Parallel All-Reduce\n8 GPUs involved\nTP8 communication" fillcolor=pink shape=ellipse style="filled,dashed"]
stage1_tp_allreduce [label="Stage1 TP All-Reduce\nStage 1 Tensor Parallel All-Reduce\n8 GPUs involved\nTP8 communication" fillcolor=pink shape=ellipse style="filled,dashed"]
```

### 4. Node Connectivity Fixes (MODIFY CONNECTIONS)
**Current**: Improper input/output connections
**Required**: Fix these specific connection issues:

```dot
# Fix input connections for all TP blocks
input_broadcast -> layer0_qkv_tp0
input_broadcast -> layer0_qkv_tp1
input_broadcast -> layer0_qkv_tp2
input_broadcast -> layer0_qkv_tp3
input_broadcast -> layer0_qkv_tp4
input_broadcast -> layer0_qkv_tp5
input_broadcast -> layer0_qkv_tp6
input_broadcast -> layer0_qkv_tp7

# Fix attention computation flow
layer0_qkv_tp0 -> layer0_score_tp0
layer0_score_tp0 -> layer0_out_tp0
layer0_out_tp0 -> layer0_tp_allreduce

# Fix expert dispatch connections
layer0_router -> layer0_dispatch_alltoall
layer0_dispatch_alltoall -> layer0_expert0
layer0_dispatch_alltoall -> layer0_expert1
# ... (connect to all 64 experts)
layer0_dispatch_alltoall -> layer0_expert63

# Fix expert combine connections
layer0_expert0 -> layer0_combine_alltoall
layer0_expert1 -> layer0_combine_alltoall
# ... (connect all 64 experts)
layer0_expert63 -> layer0_combine_alltoall
layer0_combine_alltoall -> layer0_norm
```

### 5. Remove Non-Operational Nodes (DELETE 2 NODES)
**Current**: Contains non-operational nodes
**Required**: Remove these nodes:

```dot
# DELETE: node [fillcolor=lightblue shape=ellipse style=filled]
# DELETE: edge [fontname=Arial fontsize=9]
```

### 6. Parallel Strategy Alignment (MODIFY ALL NODE LABELS)
**Current**: Generic GPU assignments
**Required**: Specific EP64-TP8-PP2-DP2 assignments:

```dot
# Update all node labels to include:
# - Specific GPU IDs (0-2047 for 2048 total GPUs)
# - Correct parallel strategy notation
# - Proper input/output dimensions
# - Strategy: EP64-TP8-PP2-DP2

# Example corrections:
layer0_qkv_tp0 [label="QKV Projection\nGPU: 0 (EP64-TP8-PP2-DP2)\nTP Rank 0, PP Stage 0\nINPUT: [batch_size=64, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=64, seq_len=128-10240, heads=2, d_k=64]" fillcolor=lightgreen shape=box]

layer0_expert0 [label="Expert 0\nGPU: 0-15 (EP64-TP8-PP2-DP2)\nEP Group 0, PP Stage 0\nINPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]\nOUTPUT: [batch_size=~1, seq_len=128-10240, hidden_dim=1024]" fillcolor=lightgreen shape=box]
```

## Implementation Priority

1. **CRITICAL** (Fix immediately): Node connectivity issues
2. **HIGH** (Fix within 1 day): Expert system expansion
3. **MEDIUM** (Fix within 3 days): Attention block decomposition
4. **LOW** (Fix within 1 week): Communication node enhancement

## Validation Checklist

After modifications, verify:
- [ ] All 64 expert nodes exist and are connected
- [ ] Attention block has proper QKV -> Score -> Output flow
- [ ] All communication nodes are present and connected
- [ ] No nodes have missing input/output connections (except input/output terminals)
- [ ] All nodes have proper EP64-TP8-PP2-DP2 GPU assignments
- [ ] DAG remains acyclic
- [ ] Parallel strategy is fully represented