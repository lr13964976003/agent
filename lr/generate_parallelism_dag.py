#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_parallelism_dag():
    """
    Create a complete DAG for the parallelism strategy deployment method.
    
    Configuration:
    - 64 GPUs total
    - Pipeline Parallelism (PP): 4 stages
    - Expert Parallelism (EP): 4-way
    - Data Parallelism (DP): 4-way
    - 16 layers, 16 experts per layer
    """
    
    # Create the DAG
    dot = Digraph(comment='LLM Parallelism Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='20,30', concentrate='true')
    dot.attr('node', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input dimensions
    batch_size = "?"
    seq_len = "512"
    heads = "16"
    d_k = "32"
    hidden_size = "1024"
    
    # Create input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]',
             shape='invhouse', style='filled', fillcolor='lightpink')
    
    # Pipeline stages
    for pp_stage in range(4):
        with dot.subgraph(name=f'cluster_pp_stage_{pp_stage}') as c:
            c.attr(label=f'Pipeline Stage {pp_stage} (Layers {pp_stage*4}-{(pp_stage+1)*4-1})', 
                   style='rounded', bgcolor='lightgray')
            
            # For each DP group (4 groups)
            for dp_group in range(4):
                with c.subgraph(name=f'cluster_dp_{pp_stage}_{dp_group}') as dp_c:
                    dp_c.attr(label=f'DP Group {dp_group}', style='dashed', bgcolor='white')
                    
                    # For each GPU in the EP group (4 GPUs)
                    for ep_gpu in range(4):
                        gpu_id = pp_stage * 16 + dp_group * 4 + ep_gpu
                        
                        # Create computation nodes for this GPU
                        # Layer processing nodes
                        for layer in range(4):
                            layer_id = pp_stage * 4 + layer
                            
                            # Attention computation
                            attn_node = f'attn_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                            dp_c.node(attn_node,
                                     f'Attention L{layer_id}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                     shape='rectangle', style='filled', fillcolor='lightgreen')
                            
                            # MoE routing
                            route_node = f'route_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                            dp_c.node(route_node,
                                     f'MoE Router L{layer_id}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                     shape='parallelogram', style='filled', fillcolor='lightyellow')
                            
                            # Expert computations (4 experts per GPU)
                            for expert in range(4):
                                expert_id = ep_gpu * 4 + expert
                                expert_node = f'expert_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}_exp{expert_id}'
                                dp_c.node(expert_node,
                                         f'Expert {expert_id} L{layer_id}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                         shape='rectangle', style='filled', fillcolor='lightgreen')
                            
                            # Expert aggregation
                            agg_node = f'agg_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                            dp_c.node(agg_node,
                                     f'Expert Agg L{layer_id}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                     shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Add communication nodes and edges
    # Input to first pipeline stage
    for dp_group in range(4):
        for ep_gpu in range(4):
            first_attn = f'attn_pp0_dp{dp_group}_gpu{ep_gpu}_layer0'
            dot.edge('input', first_attn, label='Input Data', style='dashed')
    
    # Connect nodes within each pipeline stage
    for pp_stage in range(4):
        for dp_group in range(4):
            for ep_gpu in range(4):
                for layer in range(4):
                    layer_id = pp_stage * 4 + layer
                    
                    attn_node = f'attn_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                    route_node = f'route_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                    agg_node = f'agg_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                    
                    # Attention -> Router
                    dot.edge(attn_node, route_node)
                    
                    # Router -> Experts (with dashed line for gate selection)
                    for expert in range(4):
                        expert_id = ep_gpu * 4 + expert
                        expert_node = f'expert_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}_exp{expert_id}'
                        dot.edge(route_node, expert_node, style='dashed', label=f'Gated {expert_id}')
                    
                    # Experts -> Aggregation
                    for expert in range(4):
                        expert_id = ep_gpu * 4 + expert
                        expert_node = f'expert_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}_exp{expert_id}'
                        dot.edge(expert_node, agg_node)
                    
                    # Connect to next layer or pipeline stage
                    if layer < 3:  # Next layer in same stage
                        next_attn = f'attn_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id+1}'
                        dot.edge(agg_node, next_attn)
                    else:  # Next pipeline stage or output
                        if pp_stage < 3:  # Send to next pipeline stage
                            # Communication between pipeline stages
                            for next_dp in range(4):
                                for next_gpu in range(4):
                                    next_attn = f'attn_pp{pp_stage+1}_dp{next_dp}_gpu{next_gpu}_layer{(pp_stage+1)*4}'
                                    # Add communication node
                                    comm_node = f'comm_pp{pp_stage}_to_pp{pp_stage+1}_dp{dp_group}_gpu{ep_gpu}'
                                    dot.node(comm_node,
                                             f'P2P Comm\\nStage {pp_stage}->{pp_stage+1}\\nGPU {pp_stage*16+dp_group*4+ep_gpu}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                             shape='ellipse', style='filled', fillcolor='lightblue')
                                    dot.edge(agg_node, comm_node)
                                    dot.edge(comm_node, next_attn)
                        else:  # Final output
                            output_node = f'output_dp{dp_group}_gpu{ep_gpu}'
                            dot.node(output_node,
                                     f'Output\\nGPU {pp_stage*16+dp_group*4+ep_gpu}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                     shape='house', style='filled', fillcolor='lightcoral')
                            dot.edge(agg_node, output_node)
    
    # Add expert-to-expert communication within stages (all-to-all)
    for pp_stage in range(4):
        for layer in range(4):
            layer_id = pp_stage * 4 + layer
            for dp_group in range(4):
                for ep_gpu in range(4):
                    # Expert all-to-all communication
                    for target_gpu in range(4):
                        if ep_gpu != target_gpu:
                            comm_node = f'expert_all2all_pp{pp_stage}_dp{dp_group}_layer{layer_id}_gpu{ep_gpu}_to_{target_gpu}'
                            dot.node(comm_node,
                                     f'Expert A2A\\nGPU {pp_stage*16+dp_group*4+ep_gpu}->{pp_stage*16+dp_group*4+target_gpu}\\nLayer {layer_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]',
                                     shape='ellipse', style='filled', fillcolor='lightblue')
                            
                            # Connect from router to communication
                            route_node = f'route_pp{pp_stage}_dp{dp_group}_gpu{ep_gpu}_layer{layer_id}'
                            dot.edge(route_node, comm_node, style='dotted')
                            
                            # Connect to target expert
                            target_expert_base = f'expert_pp{pp_stage}_dp{dp_group}_gpu{target_gpu}_layer{layer_id}_exp{target_gpu*4}'
                            dot.edge(comm_node, target_expert_base, style='dotted')
    
    # Create output directory if it doesn't exist
    os.makedirs('./outputs/2025-12-26-11-27-33', exist_ok=True)
    
    # Save the DOT file
    dot_file_path = './outputs/2025-12-26-11-27-33/llm_parallelism_dag.dot'
    dot.save(dot_file_path)
    
    # Render to SVG
    svg_file_path = './outputs/2025-12-26-11-27-33/llm_parallelism_dag.svg'
    dot.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")
    
    return dot_file_path, svg_file_path

if __name__ == "__main__":
    create_parallelism_dag()