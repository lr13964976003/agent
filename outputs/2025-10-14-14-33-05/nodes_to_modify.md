# DAG Validation Report

## Issues Found in DAGs

### Proposed Two-Level Partitioning DAG
- **File**: `./outputs/2025-10-14-14-33-05/proposed_two_level_partitioning.dot`
- **Cycle Check**: ✅ No cycles detected
- **Input Node Validation**: ❌ `broadcast_0` has only outputs (no input nodes)
- **Output Node Validation**: ✅ All non-output nodes have at least one output

### Detailed MHA Partitioning DAG
- **File**: `./outputs/2025-10-14-14-33-05/detailed_mha_partitioning.dot`
- **Cycle Check**: ✅ No cycles detected
- **Input Node Validation**: ❌ `broadcast_detailed` has only outputs (no input nodes)
- **Output Node Validation**: ✅ All non-output nodes have at least one output

## Nodes Requiring Modification

### Proposed Two-Level Partitioning DAG
- **broadcast_0**: Needs to have at least one input connection

### Detailed MHA Partitioning DAG
- **broadcast_detailed**: Needs to have at least one input connection

## Recommended Fixes

Both DAGs have the same issue: the broadcast nodes (`broadcast_0` and `broadcast_detailed`) are missing input connections. These nodes should receive input from a preceding node in the graph to satisfy the DAG requirements.

- For `broadcast_0` in Proposed Two-Level Partitioning DAG: Add an edge from a preceding node to `broadcast_0`
- For `broadcast_detailed` in Detailed MHA Partitioning DAG: Add an edge from a preceding node to `broadcast_detailed`