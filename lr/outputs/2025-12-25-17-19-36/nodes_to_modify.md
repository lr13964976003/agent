# Nodes Requiring Modification

## Critical Issues - Must Fix

### 1. Cycle-Creating Nodes

#### `ep_all2all` (Expert Parallel All-to-All)
**Problem:** Creates cycle with bidirectional edges
**Current edges:**
- `gpu0_l0_moe -> ep_all2all` 
- `gpu1_l0_moe -> ep_all2all`
- `ep_all2all -> gpu0_l0_moe` ❌ (Creates cycle)
- `ep_all2all -> gpu1_l0_moe` ❌ (Creates cycle)

**Required fix:** Remove backward edges. Expert parallel should route tokens forward, not return them to source.

#### `gpu0_l0_moe` (GPU 0 Layer 0 MoE)
**Problem:** Part of cycle due to incoming edge from ep_all2all
**Current problematic edge:**
- `ep_all2all -> gpu0_l0_moe` ❌

**Required fix:** Remove this incoming edge to break the cycle.

#### `gpu1_l0_moe` (GPU 1 Layer 0 MoE)
**Problem:** Part of cycle due to incoming edge from ep_all2all
**Current problematic edge:**
- `ep_all2all -> gpu1_l0_moe` ❌

**Required fix:** Remove this incoming edge to break the cycle.

### 2. Disconnected Nodes

#### `load_balancer` (Load Balancer)
**Problem:** Only has output edges, no input edges
**Current edges:**
- `load_balancer -> gpu0_l0_gate`
- `load_balancer -> gpu0_l1_gate`
- `load_balancer -> gpu1_l0_gate`
- `load_balancer -> gpu1_l1_gate`

**Required fix:** Add input connections from GPU utilization metrics or system monitoring nodes, or document as implicit source node.

## Incomplete Representation - Should Fix

### 3. Aggregate Node

#### `remaining_gpus` (GPUs 2-23 Aggregate)
**Problem:** Represents 22 GPUs as single aggregate node without individual decomposition
**Current representation:** Single node for GPUs 2-23

**Required fix:** Decompose into individual GPU nodes (gpu2 through gpu23) with the same attention decomposition pattern as GPUs 0-1:
- Each GPU needs: q_proj, k_proj, v_proj, attn_scores, out_proj nodes
- Each GPU needs: gate and moe nodes for MoE layers
- Each GPU needs proper input/output connections

## Summary of Required Changes

1. ** Remove cycle-creating edges:**
   - Delete: `ep_all2all -> gpu0_l0_moe`
   - Delete: `ep_all2all -> gpu1_l0_moe`

2. **Fix load balancer connectivity:**
   - Add input edges to `load_balancer` from GPU monitoring nodes
   - Or document as implicit control source

3. **Expand GPU representation:**
   - Replace `remaining_gpus` with individual nodes gpu2 through gpu23
   - Apply same attention decomposition pattern as GPUs 0-1
   - Ensure proper parallel strategy connections for all 24 GPUs

## Impact Assessment

- **Cycle issue:** CRITICAL - Would cause infinite processing loops
- **Load balancer:** MEDIUM - Incomplete data flow representation
- **GPU aggregation:** MEDIUM - Incomplete parallel strategy visualization

These modifications are essential for creating a valid, acyclic DAG that accurately represents the 24-GPU parallel strategy.