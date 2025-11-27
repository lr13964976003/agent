#!/usr/bin/env python3

"""
Generate comprehensive DAG for Large-Scale Cross-Node Expert Parallelism
Shows integration of DP, TP, PP with the proposed EP strategy
"""

import graphviz

def create_large_ep_dag():
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Define colors for different operations
    compute_color = 'lightblue'
    comm_color = 'lightyellow'
    route_color = 'lightgreen'
    split_color = 'lightcoral'
    aggregate_color = 'lightpink'
    
    # Model parameters from the paper
    batch_size = 1024
    seq_len = 2048
    hidden_dim = 7168
    ffn_hidden = 18432
    num_heads = 128
    head_dim = 56
    vocab_size = 128000
    
    # Parallel configuration
    dp_degree = 2  # Data parallelism degree
    tp_degree = 2  # Tensor parallelism degree  
    pp_degree = 4  # Pipeline parallelism degree
    ep_degree = 16  # Expert parallelism degree (minimum 16 for large EP)
    
    total_gpus = dp_degree * tp_degree * pp_degree * ep_degree
    
    # =============================================================================
    # INPUT NODE
    # =============================================================================
    dot.node('input', 
             f'INPUT\\nGPU: ALL_DP_RANKS\\nInput: [batch={batch_size}, seq={seq_len}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]',
             shape='ellipse', fillcolor=comm_color)
    
    # =============================================================================
    # DATA PARALLEL SPLIT
    # =============================================================================
    dot.node('dp_split',
             f'DP_SPLIT\\nGPU: DataParallelController\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]',
             shape='parallelogram', fillcolor=split_color)
    
    dot.edge('input', 'dp_split', style='dashed')
    
    # =============================================================================
    # LAYER 1: Dense Layer (Representative of layers 1-3)
    # =============================================================================
    
    # Pipeline Stage 0 - Layer 1 Computation
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            gpu_id = dp_rank * (tp_degree * pp_degree * ep_degree) + tp_rank * (pp_degree * ep_degree)
            
            # MLA Computation (split across TP)
            dot.node(f'layer1_mla_dp{dp_rank}_tp{tp_rank}',
                     f'LAYER1_MLA\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, heads={num_heads//tp_degree}, d_k={head_dim}]',
                     fillcolor=compute_color)
            
            # FFN Computation (tensor parallel)
            dot.node(f'layer1_ffn_dp{dp_rank}_tp{tp_rank}',
                     f'LAYER1_FFN_TP\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]',
                     fillcolor=compute_color)
            
            # TP All-reduce for FFN
            dot.node(f'layer1_ffn_allreduce_dp{dp_rank}_tp{tp_rank}',
                     f'LAYER1_FFN_ALLREDUCE\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]',
                     shape='ellipse', fillcolor=comm_color)
            
            # Connections
            if dp_rank == 0 and tp_rank == 0:
                dot.edge('dp_split', f'layer1_mla_dp{dp_rank}_tp{tp_rank}')
            
            dot.edge(f'layer1_mla_dp{dp_rank}_tp{tp_rank}', f'layer1_ffn_dp{dp_rank}_tp{tp_rank}')
            dot.edge(f'layer1_ffn_dp{dp_rank}_tp{tp_rank}', f'layer1_ffn_allreduce_dp{dp_rank}_tp{tp_rank}')
    
    # =============================================================================
    # LAYER 4: First MoE Layer (Representative of MoE layers)
    # =============================================================================
    
    # MLA Computation for Layer 4
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            gpu_base = dp_rank * (tp_degree * pp_degree * ep_degree) + tp_rank * (pp_degree * ep_degree) + pp_degree
            
            dot.node(f'layer4_mla_dp{dp_rank}_tp{tp_rank}',
                     f'LAYER4_MLA\\nGPU: {gpu_base}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, heads={num_heads//tp_degree}, d_k={head_dim}]',
                     fillcolor=compute_color)
            
            # Connection from Layer 1
            for prev_tp_rank in range(tp_degree):
                dot.edge(f'layer1_ffn_allreduce_dp{dp_rank}_tp{prev_tp_rank}', f'layer4_mla_dp{dp_rank}_tp{tp_rank}')
    
    # Expert Routing and Computation for Layer 4
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            for expert_id in range(ep_degree):
                gpu_id = dp_rank * (tp_degree * pp_degree * ep_degree) + tp_rank * (pp_degree * ep_degree) + pp_degree + expert_id
                
                # Gate computation (routing)
                dot.node(f'layer4_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_GATE\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, expert_scores={ep_degree}]',
                         shape='parallelogram', fillcolor=route_color)
                
                # Expert selection and token dispatch
                dot.node(f'layer4_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_DISPATCH\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [tokens_selected={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='parallelogram', fillcolor=route_color)
                
                # Expert MLP computation (with tensor parallelism)
                dot.node(f'layer4_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_EXPERT_MLP\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={ffn_hidden//tp_degree}]',
                         fillcolor=compute_color)
                
                # Expert output projection
                dot.node(f'layer4_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_EXPERT_PROJ\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={ffn_hidden//tp_degree}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]',
                         fillcolor=compute_color)
                
                # Expert all-reduce
                dot.node(f'layer4_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_EXPERT_ALLREDUCE\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='ellipse', fillcolor=comm_color)
                
                # Token aggregation
                dot.node(f'layer4_aggregate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER4_AGGREGATE\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='parallelogram', fillcolor=aggregate_color)
                
                # Connections with dashed lineses for routing
                dot.edge(f'layer4_mla_dp{dp_rank}_tp{tp_rank}', f'layer4_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', style='dashed')
                dot.edge(f'layer4_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer4_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', style='dashed')
                dot.edge(f'layer4_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer4_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer4_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer4_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer4_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer4_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer4_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer4_aggregate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
    
    # =============================================================================
    # LAYER 61: Final MoE Layer (Representative of final layers)
    # =============================================================================
    
    # MLA Computation for Layer 61
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            gpu_base = dp_rank * (tp_degree * pp_degree * ep_degree) + tp_rank * (pp_degree * ep_degree) + (pp_degree-1) * ep_degree
            
            dot.node(f'layer61_mla_dp{dp_rank}_tp{tp_rank}',
                     f'LAYER61_MLA\\nGPU: {gpu_base}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, heads={num_heads//tp_degree}, d_k={head_dim}]',
                     fillcolor=compute_color)
            
            # Connection from Layer 4
            for prev_tp_rank in range(tp_degree):
                for expert_id in range(ep_degree):
                    dot.edge(f'layer4_aggregate_dp{dp_rank}_tp{prev_tp_rank}_ep{expert_id}', f'layer61_mla_dp{dp_rank}_tp{tp_rank}')
    
    # Expert Routing and Computation for Layer 61
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            for expert_id in range(ep_degree):
                gpu_id = dp_rank * (tp_degree * pp_degree * ep_degree) + tp_rank * (pp_degree * ep_degree) + (pp_degree-1) * ep_degree + expert_id
                
                # Gate computation (routing)
                dot.node(f'layer61_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_GATE\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, expert_scores={ep_degree}]',
                         shape='parallelogram', fillcolor=route_color)
                
                # Expert selection and token dispatch
                dot.node(f'layer61_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_DISPATCH\\nGPU: {gpu_id}\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [tokens_selected={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='parallelogram', fillcolor=route_color)
                
                # Expert MLP computation (with tensor parallelism)
                dot.node(f'layer61_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_EXPERT_MLP\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={ffn_hidden//tp_degree}]',
                         fillcolor=compute_color)
                
                # Expert output projection
                dot.node(f'layer61_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_EXPERT_PROJ\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={ffn_hidden//tp_degree}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]',
                         fillcolor=compute_color)
                
                # Expert all-reduce
                dot.node(f'layer61_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_EXPERT_ALLREDUCE\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim//tp_degree}]\\nOutput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='ellipse', fillcolor=comm_color)
                
                # Token aggregation
                dot.node(f'layer61_aggregate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}',
                         f'LAYER61_AGGREGATE\\nGPU: {gpu_id}\\nInput: [tokens={batch_size//dp_degree//ep_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]',
                         shape='parallelogram', fillcolor=aggregate_color)
                
                # Connections with dashed lines for routing
                dot.edge(f'layer61_mla_dp{dp_rank}_tp{tp_rank}', f'layer61_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', style='dashed')
                dot.edge(f'layer61_gate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer61_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', style='dashed')
                dot.edge(f'layer61_dispatch_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer61_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer61_expert_mlp_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer61_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer61_expert_proj_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer61_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
                dot.edge(f'layer61_expert_allreduce_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', f'layer61_aggregate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}')
    
    # =============================================================================
    # OUTPUT NODE
    # =============================================================================
    
    # Final output aggregation across data parallel ranks
    dot.node('output_agg',
             f'OUTPUT_AGGREGATION\\nGPU: OutputController\\nInput: [batch={batch_size//dp_degree}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]',
             shape='parallelogram', fillcolor=aggregate_color)
    
    dot.node('output',
             f'OUTPUT\\nGPU: ALL_DP_RANKS\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, vocab={vocab_size}]',
             shape='ellipse', fillcolor=comm_color)
    
    # Connect final layer to output
    for dp_rank in range(dp_degree):
        for tp_rank in range(tp_degree):
            for expert_id in range(ep_degree):
                dot.edge(f'layer61_aggregate_dp{dp_rank}_tp{tp_rank}_ep{expert_id}', 'output_agg')
    
    dot.edge('output_agg', 'output')
    
    return dot

if __name__ == "__main__":
    # Generate the DAG
    dag = create_large_ep_dag()
    
    # Save as DOT file
    dot_file_path = "../outputs/2025-11-27-10-10-33/large_ep_comprehensive_dag.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Render as SVG
    svg_file_path = "../outputs/2025-11-27-10-10-33/large_ep_comprehensive_dag.svg"
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")