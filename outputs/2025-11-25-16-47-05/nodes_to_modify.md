# DAG Analysis Report - Nodes Requiring Modification

## Baseline DAG Issues

### Critical Connectivity Issues
- **Missing connections**: Expert nodes l0_exp0 through l0_exp7 and l8_exp0 through l8_exp7 are not properly connected in the edge extraction
- **Incomplete layer representation**: Only layers 0 and 8 are shown, missing layers 1-7 and 9-15
- **Disconnected expert nodes**: 17 expert nodes have only in-degree but no outgoing edges detected

### GPU Assignment Issues
- **Expert GPU conflicts**: Multiple experts assigned to same GPU (e.g., Expert 0 and Expert 4 both on GPU 0)
- **Expert distribution**: Only 4 unique GPUs used per layer (0-3, 8-11) instead of 8 GPUs per stage

### Shape Specification Issues
- **Inconsistent tensor shapes**: Mix of concrete shapes ([128,10000,4096]) and vague "tokens" notation
- **Missing intermediate shapes**: Expert input/output shapes use "tokens" without specification

## Proposed DAG Issues

### Representation Issues
- **Template-based nodes**: Uses "expert_template" instead of individual expert nodes
- **Vague GPU assignments**: Uses variables like "layer_gpu" and "layer×16+N" instead of concrete GPU indices
- **Simplified connections**: Single template connection masks individual expert routing complexity

### Shape Specification Issues
- **Vague routing shapes**: "[tokens,4096] routed" is not a concrete tensor shape
- **Missing expert specifications**: Single template represents all 16 experts without individual details

## Required Modifications

### Baseline DAG
1. **Fix connectivity**: Ensure all expert nodes have proper incoming and outgoing edges
2. **Complete layer representation**: Add layers 1-7 and 9-15 with full specifications
3. **Standardize GPU assignments**: Use consistent 8 GPUs per stage with proper expert distribution
4. **Standardize shapes**: Replace "tokens" with concrete tensor dimensions

### Proposed DAG
1. **Replace templates**: Create individual nodes for each of the 16 experts per layer
2. **Specify concrete GPU indices**: Replace variables with actual GPU assignments (0-255)
3. **Add detailed connections**: Show individual routing to each expert
4. **Standardize tensor shapes**: Use concrete shapes throughout

## Files Status
- **Baseline DAG**: INCORRECT - Multiple connectivity and representation issues
- **Proposed DAG**: INCORRECT - Template-based representation inadequate for detailed analysis