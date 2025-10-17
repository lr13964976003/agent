# DAG Error Report

## Analysis Results

### ma_separation_dag.dot
- **Status**: VALID
- **Issues**: None
- **Summary**: This DAG satisfies all requirements:
  - No cycles detected
  - All non-input nodes have at least one input
  - All non-output nodes have at least one output

### baseline_tp8_pp2_dag.dot
- **Status**: INVALID
- **Issues**:
  - `stage0_layer0_residual1` has only in-degree but is not an output node
  - `stage1_layer3_residual2` has only out-degree but is not an input node
  - The DAG appears to be incomplete - missing connections between stage0 and stage1

### baseline_tp8_dag.dot
- **Status**: INVALID
- **Issues**:
  - `layer0_norm1` has only in-degree but is not an output node
  - `layer3_moe_allreduce` has only out-degree but is not an input node
  - The DAG appears to be heavily truncated/incomplete

### baseline_pp2_dag.dot
- **Status**: INVALID
- **Issues**:
  - `s0_embedding` has only in-degree but is not an output node
  - `s1_layer3_residual2` has only out-degree but is not an input node
  - The DAG appears to be heavily truncated/incomplete

## Required Modifications

### For baseline_tp8_pp2_dag.dot:
- **stage0_layer0_residual1**: Needs output connections
- **stage1_layer3_residual2**: Needs input connections from stage0

### For baseline_tp8_dag.dot:
- **layer0_norm1**: Needs output connections to continue the pipeline
- **layer3_moe_allreduce**: Needs input connections from preceding layers

### For baseline_pp2_dag.dot:
- **s0_embedding**: Needs output connections to continue the pipeline  
- **s1_layer3_residual2**: Needs input connections from stage0

## Recommendation
These DAGs appear to be missing critical connections between stages/layers. The structure suggests they should contain full pipeline stages, but are currently showing only fragments of the complete graph.