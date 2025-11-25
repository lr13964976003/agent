# Nodes Requiring Modification

## Baseline DAG - Critical Issues

### CYCLE-RELATED NODES (CRITICAL PRIORITY)
- **layer_0_add_norm2** - Part of cycle structure
- **layer_7_qkv** - Connected incorrectly creating cycles
- **layer_8_add_norm2** - Part of pipeline communication cycle
- **layer_15_qkv** - Final layer with cyclic dependencies

### MISSING LAYERS (HIGH PRIORITY)
- **layer_1_qkv** through **layer_6_qkv** - Missing entirely
- **layer_9_qkv** through **layer_14_qkv** - Missing entirely
- Corresponding attention, projection, and normalization layers
- Missing gate and expert nodes for all intermediate layers

### GPU ASSIGNMENT ISSUES
- **All intermediate layers** (1-6, 9-14) - Missing GPU specifications
- **Expert nodes** - Need verification of GPU assignments
- **Pipeline communication nodes** - GPU transitions need validation

## Proposed DAG - Structural Issues

### INCOMPLETE LAYERS (CRITICAL)
- **layer_1 through layer_14** - Completely missing
- **All intermediate layer nodes** absent
- **Missing scatter/gather** operations for layers 1-14

### EXPERT ROUTING ISSUES
- **layer_0_scatter** - Incomplete routing implementation
- **layer_0_gather** - Missing proper aggregation
- **layer_15_scatter** - Needs verification
- **layer_15_gather** - Needs verification

### GPU AND SHAPE SPECIFICATIONS
- **All missing layers** need GPU assignments
- **Expert nodes** need shape validation
- **Communication nodes** need GPU specifications

## Specific Modification Requirements

### For Baseline DAG:
1. **BREAK cycles** by removing redundant connections
2. **ADD 11 missing layers** (1-6, 9-14)
3. **VERIFY GPU indices** for all layers (0-15)
4. **VALIDATE tensor shapes** [batch=128,seq=10000,hidden=4096]
5. **ENSURE proper input/output** for each node

### For Proposed DAG:
1. **ADD complete 16-layer structure**
2. **IMPLEMENT proper EP=16 expert parallelism**
3. **VERIFY scatter/gather operations**
4. **VALIDATE GPU assignments**
5. **ENSURE shape consistency**

## Priority Order
1. **Baseline DAG cycles** - Immediate fix required
2. **Missing layers** - Both DAGs need completion
3. **GPU specifications** - All nodes need proper assignment
4. **Shape validation** - Ensure tensor compatibility
5. **Input/output verification** - Each node must have both

**Status: Both DAGs require extensive modifications before they can be considered valid.**