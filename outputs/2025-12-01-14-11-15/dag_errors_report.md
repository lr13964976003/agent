# DAG Error Analysis Report

## Critical Issues Found

### 1. GPU Mapping DAG - CYCLES DETECTED
**File**: `../outputs/2025-12-01-14-11-15/gpu_mapping_dag.dot`
**Issue**: Contains cycles, violating DAG properties
**Status**: CRITICAL - This makes the entire DAG invalid

### 2. Detailed MoE Layer DAG - Missing Input Connections
**File**: `../outputs/2025-12-01-14-11-15/detailed_moe_layer.dot`
**Issue**: 58 nodes have only outgoing edges (no incoming connections)
**Violation**: All nodes except input must have at least one input node

### Nodes Requiring Modification in Detailed MoE Layer DAG:

The following nodes need incoming edges added:

- expert_55
- expert_14
- expert_37
- expert_35
- expert_62
- expert_22
- expert_54
- expert_5
- expert_23
- expert_39
- expert_6
- expert_26
- expert_17
- expert_19
- expert_3
- expert_59
- expert_7
- expert_47
- expert_53
- expert_51
- expert_41
- expert_61
- expert_29
- expert_9
- expert_18
- expert_21
- expert_46
- expert_10
- expert_49
- expert_43
- expert_2
- expert_50
- expert_15
- expert_42
- expert_58
- expert_11
- expert_30
- expert_12
- expert_27
- expert_28
- expert_20
- expert_1
- expert_57
- expert_45
- expert_31
- expert_36
- expert_63
- expert_4
- expert_33
- expert_34
- expert_44
- expert_60
- expert_25
- expert_38
- expert_52
- expert_13

## Recommended Fixes:

1. **GPU Mapping DAG**: Remove cycles by breaking bidirectional connections or restructure the communication pattern
2. **Detailed MoE Layer DAG**: Add proper incoming connections to all expert nodes from the routing mechanism

## DAGs That Passed Validation:

- llm_hybrid_parallelism_dag.dot ✓
- complete_deployment_dag.dot ✓
- expert_parallelism_detailed.dot ✓
- tensor_parallelism_detailed.dot ✓