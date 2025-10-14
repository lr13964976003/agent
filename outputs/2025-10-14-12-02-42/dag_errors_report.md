# DAG Validation Errors Report

## Baseline Tensor Pipeline DAG Errors

### Nodes with Missing Outputs (should have outputs but don't):
- `Layer1_QKV_Split` - This node splits input for QKV operations but has no outgoing edges
- `Layer8_QKV_Split` - This node splits input for QKV operations but has no outgoing edges

### Nodes with Missing Inputs (should have inputs but don't):
- `Layer1_Add` - This node performs addition operation but has no incoming edges
- `Layer15_Add` - This node performs addition operation but has no incoming edges

## Proposed Layer-wise DAG Errors

### Nodes with Missing Outputs (should have outputs but don't):
- `Layer2_Q_Linear` - This node performs Q linear transformation but has no outgoing edges

### Nodes with Missing Inputs (should have inputs but don't):
- `Comm14_15` - This communication node should receive input from Layer14 but has no incoming edges

## Required Fixes

### Baseline Tensor Pipeline DAG:
1. Add edges from `Layer1_QKV_Split` to appropriate Q, K, V linear layers for Layer1
2. Add edges from `Layer8_QKV_Split` to appropriate Q, K, V linear layers for Layer8
3. Add incoming edges to `Layer1_Add` from previous layer operations
4. Add incoming edges to `Layer15_Add` from previous layer operations

### Proposed Layer-wise DAG:
1. Add outgoing edges from `Layer2_Q_Linear` to appropriate next operations
2. Add incoming edges to `Comm14_15` from Layer14 operations

## Summary
Both DAGs have structural connectivity issues that violate the requirements that all nodes (except input) must have inputs and all nodes (except output) must have outputs.