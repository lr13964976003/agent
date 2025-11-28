# Nodes That Need Modification

## Issues Found:

### 1. Nodes with Only In-Degree (No Outputs):
- `comm_0_64`: Only receives input from gate_0, no outgoing edges
- `comm_64_255`: Only receives input from gate_64, no outgoing edges  
- `final_out`: Only receives input from collect_res, no outgoing edges (this is expected for output node)
- `expert_192`: Only receives input from comm_to_expert_192, no outgoing edges
- `expert_128`: Only receives input from comm_to_expert_128, no outgoing edges
- `expert_0`: Only receives input from comm_to_expert_0, no outgoing edges

### 2. Nodes with Only Out-Degree (No Inputs):
- `comm_to_expert_255`: Only outputs to expert_255, no incoming edges
- `comm_to_expert_192`: Only outputs to expert_192, no incoming edges
- `input_tokens`: Only outputs to token_split, no incoming edges (this is expected for input node)
- `comm_to_expert_64`: Only outputs to expert_64, no incoming edges
- `comm_to_expert_128`: Only outputs to expert_128, no incoming edges

### 3. Required Modifications:

#### For Expert Nodes (expert_0, expert_128, expert_192):
These nodes should connect to their respective MLA operations:
- `expert_0` should connect to `mla_0`
- `expert_128` should connect to `mla_128` (missing node)
- `expert_192` should connect to `mla_192` (missing node)

#### For Communication Nodes:
- `comm_0_64` and `comm_64_255` should connect to appropriate destination nodes
- `comm_to_expert_*` nodes should receive inputs from appropriate source nodes

#### Missing Nodes:
- Missing `mla_128` and `mla_192` nodes
- Missing `gate_128` and `gate_192` nodes  
- Missing `residual_add_128` and `residual_add_192` nodes

### 4. Root Cause:
The DAG appears to be incomplete - it only shows partial processing for GPUs 0, 64, and 255, but has isolated expert nodes for GPUs 128 and 192 without their corresponding processing pipeline.

### 5. Recommended Fixes:
1. Add missing MLA, Gate, and Residual Add nodes for GPUs 128 and 192
2. Connect the communication nodes to appropriate destinations
3. Ensure all expert nodes have proper incoming and outgoing edges
4. Verify the complete data flow path from input to output