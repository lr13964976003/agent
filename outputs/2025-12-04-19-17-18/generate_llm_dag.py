#!/usr/bin/env python3
"""
DAG Generator for LLM EP16_TP4_PP2_Hybrid Strategy
This script generates a comprehensive DAG representing the deployment of a 30B-MoE model
on 128 GPUs with Expert Parallelism (16), Tensor Parallelism (4), and Pipeline Parallelism (2)
"""

import graphviz
from graphviz import Digraph
import os

def create_llm_dag():
    # Create the DAG
    dot = Digraph(comment='LLM EP16_TP4_PP2_Hybrid Strategy DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontsize='10')
    
    # Define node shapes and colors
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters from the strategy
    total_layers = 16
    experts_per_layer = 64
    token_dimension = 1024
    moe_hidden_size = 2048
    batch_size = 128
    sequence_length = 1024
    
    # Parallel dimensions
    ep_size = 16  # Expert Parallel
    tp_size = 4   # Tensor Parallel
    pp_size = 2   # Pipeline Parallel
    
    # GPU assignment: 128 GPUs total
    # EP16_TP4_PP2 = 16 * 4 * 2 = 128 GPUs
    
    # Create input node
    with dot.subgraph(name='cluster_input') as input_cluster:
        input_cluster.attr(label='Input Layer', style='rounded', fillcolor='lightgray')
        input_node = 'input'
        input_cluster.node(input_node, 
                          f'Input Layer\\nGPU: All GPUs\\nInput: [batch_size={batch_size}, seq_len={sequence_length}, hidden={token_dimension}]\\nOutput: [batch_size={batch_size}, seq_len={sequence_length}, hidden={token_dimension}]',
                          shape='rectangle', fillcolor='lightblue')
    
    # Process each pipeline stage
    for pp_stage in range(pp_size):
        with dot.subgraph(name=f'cluster_pp_stage_{pp_stage}') as pp_cluster:
            pp_cluster.attr(label=f'Pipeline Stage {pp_stage}', style='rounded', fillcolor='lightcyan')
            
            # Each pipeline stage has total_layers/2 = 8 layers
            layers_per_stage = total_layers // pp_size
            start_layer = pp_stage * layers_per_stage
            end_layer = start_layer + layers_per_stage
            
            for layer_idx in range(start_layer, end_layer):
                with pp_cluster.subgraph(name=f'cluster_layer_{layer_idx}') as layer_cluster:
                    layer_cluster.attr(label=f'Layer {layer_idx}', style='rounded', fillcolor='lightpink')
                    
                    # Self-Attention part (Tensor Parallel)
                    with layer_cluster.subgraph(name=f'cluster_attention_{layer_idx}') as att_cluster:
                        att_cluster.attr(label=f'Self-Attention Layer {layer_idx}', style='rounded', fillcolor='lightsteelblue')
                        
                        # QKV Linear layers (Column Parallel)
                        for tp_rank in range(tp_size):
                            gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                            qkv_node = f'qkv_linear_{layer_idx}_tp{tp_rank}'
                            att_cluster.node(qkv_node,
                                           f'QKV Linear\\nGPU: {gpu_id}\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size}, seq={sequence_length}, heads=32, d_k={token_dimension//32}]',
                                           shape='rectangle', fillcolor='lightblue')
                        
                        # Attention computation
                        for tp_rank in range(tp_size):
                            gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                            att_comp_node = f'attention_{layer_idx}_tp{tp_rank}'
                            att_cluster.node(att_comp_node,
                                           f'Attention Computation\\nGPU: {gpu_id}\\nInput: [batch={batch_size}, seq={sequence_length}, heads=32, d_k={token_dimension//32}]\\nOutput: [batch={batch_size}, seq={sequence_length}, heads=32, d_k={token_dimension//32}]',
                                           shape='rectangle', fillcolor='lightblue')
                        
                        # Output Linear (Row Parallel)
                        for tp_rank in range(tp_size):
                            gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                            out_linear_node = f'att_out_linear_{layer_idx}_tp{tp_rank}'
                            att_cluster.node(out_linear_node,
                                           f'Attention Output Linear\\nGPU: {gpu_id}\\nInput: [batch={batch_size}, seq={sequence_length}, heads=32, d_k={token_dimension//32}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]',
                                           shape='rectangle', fillcolor='lightblue')
                        
                        # All-reduce for attention output
                        allreduce_att_node = f'allreduce_att_{layer_idx}'
                        att_cluster.node(allreduce_att_node,
                                       f'All-Reduce Attention\\nGPU: {[pp_stage*ep_size*tp_size}-{(pp_stage+1)*ep_size*tp_size-1}]\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]',
                                       shape='ellipse', fillcolor='lightgreen')
                    
                    # MoE part (Expert Parallel)
                    with layer_cluster.subgraph(name=f'cluster_moe_{layer_idx}') as moe_cluster:
                        moe_cluster.attr(label=f'MoE Layer {layer_idx}', style='rounded', fillcolor='lightcoral')
                        
                        # Gate (on first GPU of each pipeline stage)
                        gate_gpu = pp_stage * (ep_size * tp_size)
                        gate_node = f'gate_{layer_idx}'
                        moe_cluster.node(gate_node,
                                       f'Gate Network\\nGPU: {gate_gpu}\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size}, seq={sequence_length}, experts={experts_per_layer}]',
                                       shape='parallelogram', fillcolor='lightyellow')
                        
                        # Expert selection (dashed line process)
                        expert_select_node = f'expert_select_{layer_idx}'
                        moe_cluster.node(expert_select_node,
                                       f'Expert Selection\\nGPU: {gate_gpu}\\nInput: [batch={batch_size}, seq={sequence_length}, experts={experts_per_layer}]\\nOutput: [batch={batch_size}, seq={sequence_length}, selected_experts=8]',
                                       shape='parallelogram', fillcolor='lightyellow')
                        
                        # Expert networks (Expert Parallel)
                        experts_per_ep = experts_per_layer // ep_size  # 4 experts per EP group
                        for ep_rank in range(ep_size):
                            for expert_idx in range(experts_per_ep):
                                expert_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                                expert_node = f'expert_{layer_idx}_ep{ep_rank}_exp{expert_idx}'
                                moe_cluster.node(expert_node,
                                               f'Expert {ep_rank*experts_per_ep + expert_idx}\\nGPU: {expert_gpu}\\nInput: [batch={batch_size//8}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size//8}, seq={sequence_length}, hidden={moe_hidden_size}]',
                                               shape='rectangle', fillcolor='lightblue')
                        
                        # Token routing to experts
                        for ep_rank in range(ep_size):
                            route_node = f'token_route_{layer_idx}_ep{ep_rank}'
                            start_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                            end_gpu = start_gpu + tp_size - 1
                            moe_cluster.node(route_node,
                                           f'Token Routing EP{ep_rank}\\nGPU: [{start_gpu}-{end_gpu}]\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size//8}, seq={sequence_length}, hidden={token_dimension}]',
                                           shape='ellipse', fillcolor='lightgreen')
                        
                        # Expert computation
                        for ep_rank in range(ep_size):
                            for expert_idx in range(experts_per_ep):
                                expert_comp_node = f'expert_comp_{layer_idx}_ep{ep_rank}_exp{expert_idx}'
                                expert_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                                moe_cluster.node(expert_comp_node,
                                               f'Expert FFN {ep_rank*experts_per_ep + expert_idx}\\nGPU: {expert_gpu}\\nInput: [batch={batch_size//8}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size//8}, seq={sequence_length}, hidden={moe_hidden_size}]',
                                               shape='rectangle', fillcolor='lightblue')
                        
                        # Expert output aggregation
                        expert_agg_node = f'expert_agg_{layer_idx}'
                        moe_cluster.node(expert_agg_node,
                                       f'Expert Output Aggregation\\nGPU: [{pp_stage*ep_size*tp_size}-{(pp_stage+1)*ep_size*tp_size-1}]\\nInput: [batch={batch_size//8}, seq={sequence_length}, hidden={moe_hidden_size}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={moe_hidden_size}]',
                                       shape='parallelogram', fillcolor='lightyellow')
                        
                        # Final MoE linear (Tensor Parallel)
                        for tp_rank in range(tp_size):
                            gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                            moe_linear_node = f'moe_out_linear_{layer_idx}_tp{tp_rank}'
                            moe_cluster.node(moe_linear_node,
                                           f'MoE Output Linear\\nGPU: {gpu_id}\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={moe_hidden_size}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]',
                                           shape='rectangle', fillcolor='lightblue')
                        
                        # All-reduce for MoE output
                        allreduce_moe_node = f'allreduce_moe_{layer_idx}'
                        moe_cluster.node(allreduce_moe_node,
                                       f'All-Reduce MoE\\nGPU: [{pp_stage*ep_size*tp_size}-{(pp_stage+1)*ep_size*tp_size-1}]\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]',
                                       shape='ellipse', fillcolor='lightgreen')
                    
                    # Layer normalization
                    layernorm_node = f'layernorm_{layer_idx}'
                    layer_cluster.node(layernorm_node,
                                     f'Layer Normalization\\nGPU: [{pp_stage*ep_size*tp_size}-{(pp_stage+1)*ep_size*tp_size-1}]\\nInput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]\\nOutput: [batch={batch_size}, seq={sequence_length}, hidden={token_dimension}]',
                                     shape='rectangle', fillcolor='lightblue')
    
    # Create output node
    with dot.subgraph(name='cluster_output') as output_cluster:
        output_cluster.attr(label='Output Layer', style='rounded', fillcolor='lightgray')
        output_node = 'output'
        output_cluster.node(output_node,
                          f'Output Layer\\nGPU: All GPUs\\nInput: [batch_size={batch_size}, seq_len={sequence_length}, hidden={token_dimension}]\\nOutput: [batch_size={batch_size}, seq_len={sequence_length}, vocab_size=50000]',
                          shape='rectangle', fillcolor='lightblue')
    
    # Connect the nodes
    # Input to first layer
    dot.edge('input', 'qkv_linear_0_tp0', style='solid')
    
    # Connect within attention
    for layer_idx in range(total_layers):
        for tp_rank in range(tp_size):
            # QKV -> Attention computation
            dot.edge(f'qkv_linear_{layer_idx}_tp{tp_rank}', f'attention_{layer_idx}_tp{tp_rank}')
            # Attention -> Output linear
            dot.edge(f'attention_{layer_idx}_tp{tp_rank}', f'att_out_linear_{layer_idx}_tp{tp_rank}')
            # Output linear -> All-reduce
            dot.edge(f'att_out_linear_{layer_idx}_tp{tp_rank}', f'allreduce_att_{layer_idx}')
    
    # Connect attention to MoE
    for layer_idx in range(total_layers):
        dot.edge(f'allreduce_att_{layer_idx}', f'gate_{layer_idx}', style='dashed')  # Gate selection with dashed line
        dot.edge(f'gate_{layer_idx}', f'expert_select_{layer_idx}')
        
        # Connect token routing
        for ep_rank in range(ep_size):
            dot.edge(f'expert_select_{layer_idx}', f'token_route_{layer_idx}_ep{ep_rank}')
            
            # Connect to experts
            for expert_idx in range(experts_per_ep):
                dot.edge(f'token_route_{layer_idx}_ep{ep_rank}', f'expert_{layer_idx}_ep{ep_rank}_exp{expert_idx}')
                dot.edge(f'expert_{layer_idx}_ep{ep_rank}_exp{expert_idx}', f'expert_comp_{layer_idx}_ep{ep_rank}_exp{expert_idx}')
                dot.edge(f'expert_comp_{layer_idx}_ep{ep_rank}_exp{expert_idx}', f'expert_agg_{layer_idx}')
        
        # Connect MoE output
        dot.edge(f'expert_agg_{layer_idx}', f'moe_out_linear_{layer_idx}_tp0')
        for tp_rank in range(tp_size):
            dot.edge(f'moe_out_linear_{layer_idx}_tp{tp_rank}', f'allreduce_moe_{layer_idx}')
        
        # Connect to layer norm
        dot.edge(f'allreduce_moe_{layer_idx}', f'layernorm_{layer_idx}')
        
        # Connect to next layer (if not last)
        if layer_idx < total_layers - 1:
            next_layer = layer_idx + 1
            dot.edge(f'layernorm_{layer_idx}', f'qkv_linear_{next_layer}_tp0')
        else:
            # Last layer connects to output
            dot.edge(f'layernorm_{layer_idx}', 'output')
    
    return dot

def main():
    # Create output directory
    output_dir = '../outputs/2025-12-04-19-17-18'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    print("Generating LLM DAG...")
    dag = create_llm_dag()
    
    # Save as DOT file
    dot_path = os.path.join(output_dir, 'llm_ep16_tp4_pp2_dag.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    print(f"DOT file saved to: {dot_path}")
    
    # Render as SVG
    svg_path = os.path.join(output_dir, 'llm_ep16_tp4_pp2_dag.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_path}")
    
    # Also render as PNG for easier viewing
    png_path = os.path.join(output_dir, 'llm_ep16_tp4_pp2_dag.png')
    dag.render(png_path.replace('.png', ''), format='png', cleanup=True)
    print(f"PNG file saved to: {png_path}")
    
    # Return the paths
    return {
        'dot_path': dot_path,
        'svg_path': svg_path,
        'png_path': png_path
    }

if __name__ == '__main__':
    paths = main()
    print("DAG generation completed!")
    print(f"Files generated:")
    for key, path in paths.items():
        print(f"  {key}: {path}")