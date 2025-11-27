#!/usr/bin/env python3
"""
Generate DAG for Large-Scale Cross-Node Expert Parallelism MoE Deployment

This script generates a comprehensive DAG showing the deployment of a 61-layer MoE model
with large-scale cross-node expert parallelism (EP=32) on 1856 H100 GPUs across 232 nodes.
"""

import os
from graphviz import Digraph

def create_moe_dag():
    """Create comprehensive DAG for MoE deployment with single-expert-per-GPU"""
    
    # Create DAG with specific attributes
    dot = Digraph(comment='Large-Scale Cross-Node Expert Parallelism MoE Deployment')
    dot.attr('graph', rankdir='TB', splines='ortho', compound='true', ranksep='0.8', nodesep='0.5')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9', arrowhead='normal')
    
    # Define tensor dimensions
    batch_size = "?"
    seq_len = "?"
    hidden_size = 7168
    num_heads = 128
    head_dim = 128
    ffn_hidden = 2048
    
    # Define GPU and node configuration
    gpus_per_node = 8
    total_nodes = 232
    total_gpus = 1856
    experts_per_layer = 32
    
    # Input node
    input_shape = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
    dot.node('input', f'INPUT\\nInput: [{input_shape}]\\nOutput: [{input_shape}]\\nGPU: 0-1855', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Dense layers (1-3) - using data parallelism across all GPUs
    for layer in range(1, 4):
        # MHA for dense layer
        mha_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        mha_output = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        
        # Create MHA nodes - distributed across multiple GPUs for parallel processing
        for gpu_id in range(0, 32):  # First 32 GPUs for initial processing
            mha_node_id = f'dense_layer{layer}_mha_gpu{gpu_id}'
            dot.node(mha_node_id, 
                     f'DENSE L{layer} MHA\\nInput: [{mha_input}]\\nOutput: [{mha_output}]\\nGPU: {gpu_id}', 
                     fillcolor='lightblue')
            
            if layer == 1:
                dot.edge('input', mha_node_id, style='dashed')
            else:
                # Connect to previous layer
                prev_mha_id = f'dense_layer{layer-1}_ffn_gpu{gpu_id}'
                dot.edge(prev_mha_id, mha_node_id)
        
        # FFN for dense layer
        ffn_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        ffn_output = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        
        for gpu_id in range(0, 32):
            ffn_node_id = f'dense_layer{layer}_ffn_gpu{gpu_id}'
            dot.node(ffn_node_id, 
                     f'DENSE L{layer} FFN\\nInput: [{ffn_input}]\\nOutput: [{ffn_output}]\\nGPU: {gpu_id}', 
                     fillcolor='lightblue')
            
            # Connect MHA to FFN
            mha_node_id = f'dense_layer{layer}_mha_gpu{gpu_id}'
            dot.edge(mha_node_id, ffn_node_id)
    
    # MOE layers (4-61) with expert parallelism
    for layer in range(4, 62):
        layer_offset = (layer - 4) * 32  # Each layer uses 32 different GPUs
        
        # MHA for MOE layer (shared across all experts)
        mha_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        mha_output = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        
        for gpu_id in range(layer_offset, layer_offset + 32):
            actual_gpu_id = gpu_id % total_gpus
            mha_node_id = f'moe_layer{layer}_mha_gpu{actual_gpu_id}'
            dot.node(mha_node_id, 
                     f'MOE L{layer} MHA\\nInput: [{mha_input}]\\nOutput: [{mha_output}]\\nGPU: {actual_gpu_id}', 
                     fillcolor='yellow')
            
            # Connect to previous layer
            if layer == 4:
                prev_ffn_id = f'dense_layer3_ffn_gpu{actual_gpu_id % 32}'
                dot.edge(prev_ffn_id, mha_node_id)
            else:
                prev_agg_id = f'moe_layer{layer-1}_agg_gpu{actual_gpu_id}'
                dot.edge(prev_agg_id, mha_node_id)
        
        # Gating network (distributed)
        gate_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        gate_output = f"batch_size={batch_size}, seq_len={seq_len}, num_experts={experts_per_layer}"
        
        for gpu_id in range(layer_offset, layer_offset + 32):
            actual_gpu_id = gpu_id % total_gpus
            gate_node_id = f'moe_layer{layer}_gate_gpu{actual_gpu_id}'
            dot.node(gate_node_id, 
                     f'MOE L{layer} GATE\\nInput: [{gate_input}]\\nOutput: [{gate_output}]\\nGPU: {actual_gpu_id}', 
                     shape='parallelogram', fillcolor='orange')
            
            # Connect MHA to gate
            mha_node_id = f'moe_layer{layer}_mha_gpu{actual_gpu_id}'
            dot.edge(mha_node_id, gate_node_id)
        
        # Expert MLPs (one per GPU - single expert per GPU principle)
        expert_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        expert_output = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        
        for expert_id in range(experts_per_layer):
            gpu_id = layer_offset + expert_id
            actual_gpu_id = gpu_id % total_gpus
            
            # First linear layer of expert MLP
            expert_linear1_id = f'moe_layer{layer}_expert{expert_id}_linear1_gpu{actual_gpu_id}'
            linear1_input = expert_input
            linear1_output = f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}"
            
            dot.node(expert_linear1_id, 
                     f'MOE L{layer} Expert{expert_id} Linear1\\nInput: [{linear1_input}]\\nOutput: [{linear1_output}]\\nGPU: {actual_gpu_id}', 
                     fillcolor='lightcoral')
            
            # GELU activation
            gelu_id = f'moe_layer{layer}_expert{expert_id}_gelu_gpu{actual_gpu_id}'
            dot.node(gelu_id, 
                     f'MOE L{layer} Expert{expert_id} GELU\\nInput: [{linear1_output}]\\nOutput: [{linear1_output}]\\nGPU: {actual_gpu_id}', 
                     fillcolor='lightcoral')
            
            dot.edge(expert_linear1_id, gelu_id)
            
            # Second linear layer of expert MLP
            expert_linear2_id = f'moe_layer{layer}_expert{expert_id}_linear2_gpu{actual_gpu_id}'
            linear2_output = expert_output
            
            dot.node(expert_linear2_id, 
                     f'MOE L{layer} Expert{expert_id} Linear2\\nInput: [{linear1_output}]\\nOutput: [{linear2_output}]\\nGPU: {actual_gpu_id}', 
                     fillcolor='lightcoral')
            
            dot.edge(gelu_id, expert_linear2_id)
            
            # Routing communication (dashed lines from gate to experts)
            for gate_gpu in range(layer_offset, layer_offset + 32):
                actual_gate_gpu = gate_gpu % total_gpus
                gate_node_id = f'moe_layer{layer}_gate_gpu{actual_gate_gpu}'
                dot.edge(gate_node_id, expert_linear1_id, style='dashed', 
                        label=f'top2 routing from GPU{actual_gate_gpu}')
        
        # Token aggregation after expert processing
        agg_input = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
        agg_output = agg_input
        
        for gpu_id in range(layer_offset, layer_offset + 32):
            actual_gpu_id = gpu_id % total_gpus
            agg_node_id = f'moe_layer{layer}_agg_gpu{actual_gpu_id}'
            dot.node(agg_node_id, 
                     f'MOE L{layer} AGGREGATION\\nInput: [{agg_input}]\\nOutput: [{agg_output}]\\nGPU: {actual_gpu_id}', 
                     shape='parallelogram', fillcolor='lightgreen')
            
            # Connect all experts to aggregation (with communication)
            for expert_id in range(experts_per_layer):
                expert_gpu = (layer_offset + expert_id) % total_gpus
                expert_linear2_id = f'moe_layer{layer}_expert{expert_id}_linear2_gpu{expert_gpu}'
                dot.edge(expert_linear2_id, agg_node_id, style='dashed', 
                        label=f'expert{expert_id} output')
    
    # Output node
    output_shape = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
    dot.node('output', f'OUTPUT\\nInput: [{output_shape}]\\nOutput: [{output_shape}]\\nGPU: 0-1855', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect final layer to output
    final_agg_nodes = []
    for gpu_id in range(1856 - 32, 1856):  # Last layer's aggregation nodes
        final_agg_id = f'moe_layer61_agg_gpu{gpu_id}'
        final_agg_nodes.append(final_agg_id)
        dot.edge(final_agg_id, 'output')
    
    return dot

def create_simplified_moe_dag():
    """Create a simplified DAG showing just one MOE layer for clarity"""
    
    dot = Digraph(comment='Single MOE Layer - Large-Scale Expert Parallelism')
    dot.attr('graph', rankdir='TB', splines='ortho', compound='true', ranksep='1.0')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue', fontname='Arial', fontsize='11')
    dot.attr('edge', fontname='Arial', fontsize='10', arrowhead='normal')
    
    # Dimensions
    batch_size = "?"
    seq_len = "?"
    hidden_size = 7168
    num_heads = 128
    head_dim = 128
    ffn_hidden = 2048
    
    # Input to MOE layer
    input_shape = f"batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
    dot.node('input', f'INPUT\\n[{input_shape}]\\nGPU: ALL', 
             shape='ellipse', fillcolor='lightgreen')
    
    # MHA (shared)
    mha_shape = input_shape
    dot.node('mha', f'MULTI-HEAD ATTENTION\\nInput: [{mha_shape}]\\nOutput: [{mha_shape}]\\nGPU: 0-31', 
             fillcolor='yellow')
    dot.edge('input', 'mha')
    
    # Gating network
    gate_output_shape = f"batch_size={batch_size}, seq_len={seq_len}, num_experts=32"
    dot.node('gate', f'GATING NETWORK\\nInput: [{mha_shape}]\\nOutput: [{gate_output_shape}]\\nGPU: 0-31', 
             shape='parallelogram', fillcolor='orange')
    dot.edge('mha', 'gate')
    
    # Show first 8 experts (GPUs 0-7) as representative sample
    for expert_id in range(8):
        gpu_id = expert_id
        
        # Expert Linear 1
        linear1_output = f"batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}"
        dot.node(f'expert{expert_id}_linear1', 
                 f'Expert{expert_id} Linear1\\nInput: [{mha_shape}]\\nOutput: [{linear1_output}]\\nGPU: {gpu_id}', 
                 fillcolor='lightcoral')
        
        # GELU
        dot.node(f'expert{expert_id}_gelu', 
                 f'Expert{expert_id} GELU\\nInput: [{linear1_output}]\\nOutput: [{linear1_output}]\\nGPU: {gpu_id}', 
                 fillcolor='lightcoral')
        
        # Expert Linear 2
        dot.node(f'expert{expert_id}_linear2', 
                 f'Expert{expert_id} Linear2\\nInput: [{linear1_output}]\\nOutput: [{mha_shape}]\\nGPU: {gpu_id}', 
                 fillcolor='lightcoral')
        
        # Connections within expert
        dot.edge(f'expert{expert_id}_linear1', f'expert{expert_id}_gelu')
        dot.edge(f'expert{expert_id}_gelu', f'expert{expert_id}_linear2')
        
        # Routing from gate (dashed)
        dot.edge('gate', f'expert{expert_id}_linear1', style='dashed', 
                label=f'top2 routing')
    
    # Aggregation node
    dot.node('agg', f'AGGREGATION\\nInput: [{mha_shape}] x 32 experts\\nOutput: [{mha_shape}]\\nGPU: 0-31', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect experts to aggregation (dashed)
    for expert_id in range(8):
        dot.edge(f'expert{expert_id}_linear2', 'agg', style='dashed', 
                label=f'expert{expert_id} output')
    
    # Output
    dot.node('output', f'OUTPUT\\n[{mha_shape}]\\nGPU: ALL', 
             shape='ellipse', fillcolor='lightgreen')
    dot.edge('agg', 'output')
    
    # Add note about remaining experts
    dot.node('note', 'NOTE: Experts 8-31 follow same pattern\\non GPUs 8-31 respectively', 
             shape='note', fillcolor='lightgray', fontname='Arial Italic')
    
    return dot

def main():
    """Generate both comprehensive and simplified DAGs"""
    
    output_dir = "../outputs/2025-11-26-17-49-41"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comprehensive MoE deployment DAG...")
    comprehensive_dag = create_moe_dag()
    
    # Save comprehensive DAG
    comprehensive_dot_path = os.path.join(output_dir, "moe_comprehensive_deployment.dot")
    comprehensive_svg_path = os.path.join(output_dir, "moe_comprehensive_deployment.svg")
    
    with open(comprehensive_dot_path, 'w') as f:
        f.write(comprehensive_dag.source)
    
    comprehensive_dag.render(
        filename=os.path.join(output_dir, "moe_comprehensive_deployment"),
        format='svg', cleanup=True
    )
    
    print("Generating simplified MoE layer DAG...")
    simplified_dag = create_simplified_moe_dag()
    
    # Save simplified DAG
    simplified_dot_path = os.path.join(output_dir, "moe_single_layer.dot")
    simplified_svg_path = os.path.join(output_dir, "moe_single_layer.svg")
    
    with open(simplified_dot_path, 'w') as f:
        f.write(simplified_dag.source)
    
    simplified_dag.render(
        filename=os.path.join(output_dir, "moe_single_layer"),
        format='svg', cleanup=True
    )
    
    # Generate deployment summary
    deployment_summary = {
        "deployment_plan": {
            "strategy": "large_scale_cross_node_expert_parallelism",
            "expert_parallelism_degree": 32,
            "single_expert_per_gpu": True,
            "total_gpus": 1856,
            "total_nodes": 232,
            "gpus_per_node": 8,
            "experts_per_layer": 32,
            "moe_layers": 58,
            "dense_layers": 3,
            "tensor_dimensions": {
                "hidden_size": 7168,
                "num_heads": 128,
                "head_dim": 128,
                "ffn_hidden_size": 2048,
                "precision": "BF16"
            }
        },
        "generated_files": [
            comprehensive_dot_path,
            comprehensive_svg_path,
            simplified_dot_path,
            simplified_svg_path
        ],
        "optimization_targets": {
            "mfu_utilization": "60%",
            "bandwidth_utilization": "80%",
            "latency": "minimized",
            "throughput": "maximized"
        },
        "key_features": [
            "single_expert_per_gpu_deployment",
            "cross_node_expert_distribution",
            "topology_aware_placement",
            "async_compute_communication_overlap",
            "dynamic_load_balancing",
            "fine_grained_pipeline_scheduling"
        ]
    }
    
    import json
    summary_path = os.path.join(output_dir, "deployment_dag_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(deployment_summary, f, indent=2)
    
    print(f"DAG generation complete!")
    print(f"Files saved to: {output_dir}")
    print(f"- Comprehensive DAG: {comprehensive_dot_path}")
    print(f"- Simplified DAG: {simplified_dot_path}")
    print(f"- Deployment summary: {summary_path}")
    
    return {
        "comprehensive_dot": comprehensive_dot_path,
        "comprehensive_svg": comprehensive_svg_path,
        "simplified_dot": simplified_dot_path,
        "simplified_svg": simplified_svg_path,
        "summary": summary_path
    }

if __name__ == "__main__":
    main()