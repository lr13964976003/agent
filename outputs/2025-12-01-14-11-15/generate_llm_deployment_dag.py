#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def generate_llm_deployment_dag():
    """
    Generate a complete DAG for LLM deployment with hybrid 3D parallelism
    - Expert Parallelism: 8-way (8 experts per GPU group)
    - Tensor Parallelism: 2-way (within each expert)
    - Pipeline Parallelism: 4-way (4 pipeline stages)
    - Total: 64 GPUs (8 × 2 × 4)
    """
    
    dot = Digraph(comment='LLM Deployment DAG with Hybrid 3D Parallelism')
    dot.attr(rankdir='TB', size='300,200')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node shapes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    batch_size = 128
    seq_len = 1024
    token_dim = 1024
    num_heads = 16
    head_dim = 64
    num_experts = 64
    moe_hidden = 2048
    num_layers = 16
    
    # Parallelism configuration
    ep_degree = 8  # Expert parallelism degree
    tp_degree = 2  # Tensor parallelism degree
    pp_degree = 4  # Pipeline parallelism degree
    
    experts_per_gpu = num_experts // ep_degree
    heads_per_gpu = num_heads // tp_degree
    layers_per_stage = num_layers // pp_degree
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded', bgcolor='lightgray')
        c.node('input', 
               f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
               shape='invhouse', style='filled', fillcolor='lightcoral')
    
    # Generate DAG for each pipeline stage
    for pp_stage in range(pp_degree):
        with dot.subgraph(name=f'cluster_pp_stage_{pp_stage}') as c:
            c.attr(label=f'Pipeline Stage {pp_stage} (Layers {pp_stage*layers_per_stage}-{(pp_stage+1)*layers_per_stage-1})', 
                   style='rounded', bgcolor='lightblue')
            
            # For each layer in this pipeline stage
            for layer_idx in range(pp_stage * layers_per_stage, (pp_stage + 1) * layers_per_stage):
                layer_name = f'layer_{layer_idx}'
                
                with c.subgraph(name=f'cluster_{layer_name}') as layer_cluster:
                    layer_cluster.attr(label=f'Layer {layer_idx}', style='rounded', bgcolor='lightgreen')
                    
                    # Layer input (from previous layer or input)
                    if layer_idx == 0:
                        prev_output = 'input'
                    else:
                        prev_output = f'layer_{layer_idx-1}_output'
                    
                    # Layer Norm (before attention)
                    ln1_name = f'{layer_name}_ln1'
                    layer_cluster.node(ln1_name,
                                       f'LayerNorm1_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='rectangle', style='filled', fillcolor='lightgreen')
                    dot.edge(prev_output, ln1_name)
                    
                    # Multi-Head Attention with Tensor Parallelism
                    # Split attention across 2 GPUs per expert group
                    for tp_rank in range(tp_degree):
                        attn_name = f'{layer_name}_attn_tp{tp_rank}'
                        start_gpu = pp_stage * ep_degree * tp_degree + tp_rank
                        end_gpu = pp_stage * ep_degree * tp_degree + tp_degree - 1
                        
                        layer_cluster.node(attn_name,
                                           f'Attention_TP{tp_rank}_GPU{start_gpu}-{end_gpu}\\n'
                                           f'Input: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\n'
                                           f'Output: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                                           shape='rectangle', style='filled', fillcolor='lightgreen')
                        
                        # Connect to layer norm
                        dot.edge(ln1_name, attn_name)
                    
                    # Attention output aggregation
                    attn_agg_name = f'{layer_name}_attn_agg'
                    layer_cluster.node(attn_agg_name,
                                       f'Attention_Aggregate_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    # Connect attention outputs to aggregation
                    for tp_rank in range(tp_degree):
                        attn_name = f'{layer_name}_attn_tp{tp_rank}'
                        dot.edge(attn_name, attn_agg_name)
                    
                    # Residual connection
                    residual_add1_name = f'{layer_name}_residual_add1'
                    layer_cluster.node(residual_add1_name,
                                       f'Residual_Add1_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Input2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    dot.edge(prev_output, residual_add1_name)  # Skip connection
                    dot.edge(attn_agg_name, residual_add1_name)
                    
                    # Layer Norm (before MOE)
                    ln2_name = f'{layer_name}_ln2'
                    layer_cluster.node(ln2_name,
                                       f'LayerNorm2_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='rectangle', style='filled', fillcolor='lightgreen')
                    dot.edge(residual_add1_name, ln2_name)
                    
                    # Expert routing
                    routing_name = f'{layer_name}_routing'
                    layer_cluster.node(routing_name,
                                       f'Expert_Routing_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                    dot.edge(ln2_name, routing_name)
                    
                    # Expert Parallelism - distribute experts across 8 GPUs
                    expert_outputs = []
                    for ep_rank in range(ep_degree):
                        # Each GPU handles 8 experts
                        start_expert = ep_rank * experts_per_gpu
                        end_expert = (ep_rank + 1) * experts_per_gpu - 1
                        
                        # Expert computation with tensor parallelism
                        for tp_rank in range(tp_degree):
                            expert_name = f'{layer_name}_expert_ep{ep_rank}_tp{tp_rank}'
                            gpu_id = pp_stage * ep_degree * tp_degree + ep_rank * tp_degree + tp_rank
                            
                            layer_cluster.node(expert_name,
                                               f'Experts_{start_expert}-{end_expert}_TP{tp_rank}_GPU{gpu_id}\\n'
                                               f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                               f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                               shape='rectangle', style='filled', fillcolor='lightgreen')
                            
                            # Gate connection (dashed line for routing)
                            dot.edge(routing_name, expert_name, style='dashed')
                    
                    # Expert output aggregation
                    expert_agg_name = f'{layer_name}_expert_agg'
                    layer_cluster.node(expert_agg_name,
                                       f'Expert_Aggregate_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    # Connect all experts to aggregation
                    for ep_rank in range(ep_degree):
                        for tp_rank in range(tp_degree):
                            expert_name = f'{layer_name}_expert_ep{ep_rank}_tp{tp_rank}'
                            dot.edge(expert_name, expert_agg_name)
                    
                    # Residual connection (MOE)
                    residual_add2_name = f'{layer_name}_residual_add2'
                    layer_cluster.node(residual_add2_name,
                                       f'Residual_Add2_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Input2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    dot.edge(residual_add1_name, residual_add2_name)  # Skip connection
                    dot.edge(expert_agg_name, residual_add2_name)
                    
                    # Set layer output
                    layer_output_name = f'{layer_name}_output'
                    layer_cluster.node(layer_output_name,
                                       f'Layer_{layer_idx}_Output_GPU{pp_stage*ep_degree*tp_degree}-{((pp_stage+1)*ep_degree*tp_degree)-1}\\n'
                                       f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                                       f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                                       shape='rectangle', style='filled,rounded', fillcolor='lightblue')
                    
                    dot.edge(residual_add2_name, layer_output_name)
    
    # Final output
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='rounded', bgcolor='lightgray')
        c.node('output', 
               f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
               shape='house', style='filled', fillcolor='lightcoral')
        
        # Connect last layer to output
        dot.edge('layer_15_output', 'output')
    
    # Communication edges between pipeline stages
    for pp_stage in range(pp_degree - 1):
        last_layer_curr_stage = (pp_stage + 1) * layers_per_stage - 1
        first_layer_next_stage = (pp_stage + 1) * layers_per_stage
        
        comm_name = f'pp_comm_stage_{pp_stage}_to_{pp_stage+1}'
        dot.node(comm_name,
                 f'Pipeline_Communication_Stage{pp_stage}_to_Stage{pp_stage+1}\\n'
                 f'Input: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\n'
                 f'Output: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='ellipse', style='filled', fillcolor='lightblue')
        
        dot.edge(f'layer_{last_layer_curr_stage}_output', comm_name)
        dot.edge(comm_name, f'layer_{first_layer_next_stage}_ln1')
    
    return dot

if __name__ == '__main__':
    # Generate the DAG
    dag = generate_llm_deployment_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-01-14-11-15/llm_deployment_dag.dot'
    dag.save(dot_file_path)
    
    # Save as SVG image
    svg_file_path = '../outputs/2025-12-01-14-11-15/llm_deployment_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")