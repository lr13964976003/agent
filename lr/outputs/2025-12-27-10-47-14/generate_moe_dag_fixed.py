#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_moe_dag():
    # Create a new directed graph
    dot = Digraph(comment='MoE Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='20,30', fontsize='12')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input dimensions
    batch_size = 64  # DP splits 128 into 2
    seq_len = 1024   # Variable, using representative value
    hidden_dim = 512
    heads = 16
    d_k = 32
    moe_hidden = 1024
    
    # GPU mapping function
    def gpu_id(dp_rank, pp_rank, ep_rank, tp_rank):
        return (dp_rank * 128) + (pp_rank * 32) + (ep_rank * 2) + tp_rank
    
    # Create input node
    dot.node('input', 
             f'Input\\nGPU Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', fillcolor='lightblue')
    
    # Focus on one pipeline stage for clarity (Stage 0: Layers 0-3)
    pp_rank = 0
    layers_start = 0
    layers_end = 4
    
    with dot.subgraph(name=f'cluster_pipeline_{pp_rank}') as c:
        c.attr(label=f'Pipeline Stage {pp_rank} (Layers {layers_start}-{layers_end-1})\\nGPUs {gpu_id(0, pp_rank, 0, 0)}-{gpu_id(0, pp_rank, 15, 1)}', 
               style='rounded,filled', fillcolor='lightgray', fontname='Arial Bold')
        
        # Process each layer in this pipeline stage
        for layer_idx in range(layers_start, layers_end):
            # MHA computation for this layer
            mha_node = f'mha_layer_{layer_idx}'
            c.node(mha_node,
                   f'MHA Layer {layer_idx}\\nGPU {gpu_id(0, pp_rank, 0, 0)}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Connect input to first MHA
            if layer_idx == 0:
                dot.edge('input', mha_node)
            else:
                prev_agg = f'agg_layer_{layer_idx-1}'
                dot.edge(prev_agg, mha_node)
            
            # Gate routing for MoE
            gate_node = f'gate_layer_{layer_idx}'
            c.node(gate_node,
                   f'Gate Layer {layer_idx}\\nGPU {gpu_id(0, pp_rank, 0, 0)}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]',
                   shape='parallelogram', fillcolor='lightyellow')
            dot.edge(mha_node, gate_node)
            
            # Show a few representative experts
            for ep_rank in [0, 1, 8, 15]:  # Show first, middle, and last experts
                # Expert computation nodes (with TP)
                for tp_rank in range(2):  # 2-way TP
                    expert_node = f'expert_layer_{layer_idx}_expert_{ep_rank}_tp_{tp_rank}'
                    gpu_num = gpu_id(0, pp_rank, ep_rank, tp_rank)
                    
                    c.node(expert_node,
                           f'Expert {ep_rank} TP{tp_rank}\\nGPU {gpu_num}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Routing communication (dashed line for gate selection)
                    route_comm = f'route_comm_layer_{layer_idx}_expert_{ep_rank}_tp_{tp_rank}'
                    c.node(route_comm,
                           f'Route Comm\\nGPU Gate -> GPU {gpu_num}',
                           shape='ellipse', fillcolor='lightblue')
                    dot.edge(gate_node, route_comm, style='dashed')
                    dot.edge(route_comm, expert_node)
                
                # TP AllReduce communication
                tp_comm = f'tp_comm_layer_{layer_idx}_expert_{ep_rank}'
                c.node(tp_comm,
                       f'TP AllReduce\\nGPU {gpu_id(0, pp_rank, ep_rank, 0)} <-> GPU {gpu_id(0, pp_rank, ep_rank, 1)}',
                       shape='ellipse', fillcolor='lightblue')
                dot.edge(f'expert_layer_{layer_idx}_expert_{ep_rank}_tp_0', tp_comm)
                dot.edge(f'expert_layer_{layer_idx}_expert_{ep_rank}_tp_1', tp_comm)
            
            # Expert aggregation (simplified - showing aggregation of shown experts)
            agg_node = f'agg_layer_{layer_idx}'
            c.node(agg_node,
                   f'Expert Aggregation Layer {layer_idx}\\nGPU {gpu_id(0, pp_rank, 0, 0)}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Connect experts to aggregation
            for ep_rank in [0, 1, 8, 15]:
                tp_comm = f'tp_comm_layer_{layer_idx}_expert_{ep_rank}'
                dot.edge(tp_comm, agg_node)
    
    # Add pipeline communication to next stage
    pp_comm = 'pp_comm_stage0_to_stage1'
    dot.node(pp_comm,
             f'PP Comm Stage0->Stage1\\nGPU {gpu_id(0, 0, 0, 0)} -> GPU {gpu_id(0, 1, 0, 0)}',
             shape='ellipse', fillcolor='lightblue')
    dot.edge('agg_layer_3', pp_comm)
    
    # Add output node
    output_node = 'output'
    dot.node(output_node,
            f'Output\\nGPU Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size=?]',
            shape='ellipse', fillcolor='lightblue')
    
    # Connect last aggregation to output (simplified)
    dot.edge('agg_layer_3', output_node, style='dashed', label='after all stages')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='rounded,filled', fillcolor='white', fontname='Arial Bold')
        c.node('legend_comp', 'Computation\\n(Rectangle)', shape='rectangle', fillcolor='lightgreen')
        c.node('legend_comm', 'Communication\\n(Ellipse)', shape='ellipse', fillcolor='lightblue')
        c.node('legend_route', 'Routing/Aggregation\\n(Parallelogram)', shape='parallelogram', fillcolor='lightyellow')
        c.node('legend_dash', 'Gate Selection\\n(Dashed Line)', shape='none', label='--- Dashed ---')
    
    return dot

if __name__ == '__main__':
    # Generate the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file = './outputs/2025-12-27-10-47-14/moe_parallel_strategy.dot'
    dag.save(dot_file)
    
    # Render as SVG
    svg_file = './outputs/2025-12-27-10-47-14/moe_parallel_strategy.svg'
    dag.render(svg_file, format='svg', cleanup=True)
    
    print(f"DAG generated and saved to:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")