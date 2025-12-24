#!/usr/bin/env python3
"""
DAG Generator for LLM Parallel Strategy: TP×EP×PP
Total GPUs: 32 (4 PP stages × 4 EP ranks × 2 TP ranks)
"""

import os
from graphviz import Digraph

def create_dag():
    # Create DAG
    dag = Digraph('LLM_TP_EP_PP_Parallelism', 
                  filename='llm_parallel_strategy_dag',
                  format='svg',
                  graph_attr={
                      'rankdir': 'TB',
                      'bgcolor': 'white',
                      'fontname': 'Arial',
                      'fontsize': '12'
                  })
    
    # Set node styles
    dag.attr('node', shape='rectangle', style='filled', fillcolor='lightblue', fontname='Arial')
    
    # Define dimensions
    batch_size = 128
    seq_len = 10240  # max sequence length
    hidden_dim = 512
    heads = 16
    d_k = 32
    moe_hidden = 1024
    
    # PP stages: 4 stages, each with 4 layers
    pp_stages = 4
    layers_per_stage = 4
    total_layers = 16
    
    # EP configuration: 4 experts per stage
    ep_ranks = 4
    experts_per_rank = 4
    total_experts = 16
    
    # TP configuration: 2-way tensor parallelism
    tp_ranks = 2
    
    # Input node
    dag.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create nodes for each PP stage
    for pp_stage in range(pp_stages):
        stage_name = f'PP_Stage_{pp_stage}'
        
        # Create subgraph for this stage
        with dag.subgraph(name=f'cluster_{stage_name}') as stage:
            stage.attr(label=f'Pipeline Stage {pp_stage} (Layers {pp_stage*layers_per_stage}-{(pp_stage+1)*layers_per_stage-1})',
                      style='rounded,filled', fillcolor='lightgray', fontname='Arial Bold')
            
            # For each layer in this stage
            for layer in range(layers_per_stage):
                global_layer = pp_stage * layers_per_stage + layer
                layer_name = f'Layer_{global_layer}'
                
                # Attention components
                # QKV Linear (TP=2)
                for tp_rank in range(tp_ranks):
                    qkv_name = f'{layer_name}_QKV_TP{tp_rank}_PP{pp_stage}'
                    stage.node(qkv_name,
                              f'QKV Linear\\nTP Rank {tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads//tp_ranks}, d_k={d_k}]',
                              fillcolor='lightblue')
                
                # All-Reduce for QKV (communication)
                ar_qkv_name = f'{layer_name}_QKV_AllReduce_PP{pp_stage}'
                stage.node(ar_qkv_name,
                          f'All-Reduce QKV\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]',
                          shape='ellipse', fillcolor='yellow')
                
                # Attention Score Computation (per TP rank)
                for tp_rank in range(tp_ranks):
                    attn_name = f'{layer_name}_Attention_TP{tp_rank}_PP{pp_stage}'
                    stage.node(attn_name,
                              f'Softmax Attention\\nTP Rank {tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads//tp_ranks}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads//tp_ranks}, d_k={d_k}]',
                              fillcolor='lightblue')
                
                # Attention Output Projection (TP=2)
                for tp_rank in range(tp_ranks):
                    attn_out_name = f'{layer_name}_AttnOut_TP{tp_rank}_PP{pp_stage}'
                    stage.node(attn_out_name,
                              f'Attention Output Proj\\nTP Rank {tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads//tp_ranks}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//tp_ranks}]',
                              fillcolor='lightblue')
                
                # All-Reduce for Attention Output
                ar_attn_name = f'{layer_name}_AttnOut_AllReduce_PP{pp_stage}'
                stage.node(ar_attn_name,
                          f'All-Reduce Attn Output\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                          shape='ellipse', fillcolor='yellow')
                
                # MoE components with EP=4
                # Routing (gate) - replicated across EP ranks
                for ep_rank in range(ep_ranks):
                    gate_name = f'{layer_name}_Gate_EP{ep_rank}_PP{pp_stage}'
                    stage.node(gate_name,
                              f'MoE Gate/Router\\nEP Rank {ep_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, expert_ids=2]',
                              shape='parallelogram', fillcolor='orange', style='dashed')
                
                # Expert computation - 4 experts per EP rank, each with TP=2
                for ep_rank in range(ep_ranks):
                    for expert in range(experts_per_rank):
                        for tp_rank in range(tp_ranks):
                            expert_name = f'{layer_name}_Expert_{ep_rank*experts_per_rank+expert}_EP{ep_rank}_TP{tp_rank}_PP{pp_stage}'
                            stage.node(expert_name,
                                      f'Expert {ep_rank*experts_per_rank+expert}\\nEP {ep_rank}, TP {tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//tp_ranks}]',
                                      fillcolor='lightcoral')
                
                # All-to-All communication for expert dispatch
                a2a_dispatch_name = f'{layer_name}_A2A_Dispatch_PP{pp_stage}'
                stage.node(a2a_dispatch_name,
                          f'All-to-All Dispatch\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                          shape='ellipse', fillcolor='yellow')
                
                # All-to-All communication for expert combine
                a2a_combine_name = f'{layer_name}_A2A_Combine_PP{pp_stage}'
                stage.node(a2a_combine_name,
                          f'All-to-All Combine\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                          shape='ellipse', fillcolor='yellow')
                
                # MLP output projection with TP=2
                for tp_rank in range(tp_ranks):
                    mlp_out_name = f'{layer_name}_MLP_Out_TP{tp_rank}_PP{pp_stage}'
                    stage.node(mlp_out_name,
                              f'MLP Output Proj\\nTP Rank {tp_rank}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden//tp_ranks}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//tp_ranks}]',
                              fillcolor='lightblue')
                
                # All-Reduce for MLP output
                ar_mlp_name = f'{layer_name}_MLP_AllReduce_PP{pp_stage}'
                stage.node(ar_mlp_name,
                          f'All-Reduce MLP Output\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                          shape='ellipse', fillcolor='yellow')
                
                # LayerNorm (replicated)
                layernorm_name = f'{layer_name}_LayerNorm_PP{pp_stage}'
                stage.node(layernorm_name,
                          f'LayerNorm\\nPP Stage {pp_stage}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                          fillcolor='lightgreen')
    
    # Create edges (dependencies)
    # Connect input to first stage
    dag.edge('input', 'Layer_0_QKV_TP0_PP0')
    
    # Connect nodes within each layer
    for pp_stage in range(pp_stages):
        for layer in range(layers_per_stage):
            global_layer = pp_stage * layers_per_stage + layer
            layer_name = f'Layer_{global_layer}'
            
            # QKV computation flow
            for tp_rank in range(tp_ranks):
                qkv_name = f'{layer_name}_QKV_TP{tp_rank}_PP{pp_stage}'
                ar_qkv_name = f'{layer_name}_QKV_AllReduce_PP{pp_stage}'
                dag.edge(qkv_name, ar_qkv_name)
                
                # Attention flow
                attn_name = f'{layer_name}_Attention_TP{tp_rank}_PP{pp_stage}'
                dag.edge(ar_qkv_name, attn_name)
                
                # Attention output flow
                attn_out_name = f'{layer_name}_AttnOut_TP{tp_rank}_PP{pp_stage}'
                dag.edge(attn_name, attn_out_name)
                
                # All-reduce for attention output
                ar_attn_name = f'{layer_name}_AttnOut_AllReduce_PP{pp_stage}'
                dag.edge(attn_out_name, ar_attn_name)
            
            # MoE flow
            a2a_dispatch_name = f'{layer_name}_A2A_Dispatch_PP{pp_stage}'
            dag.edge(ar_attn_name, a2a_dispatch_name)
            
            # Expert computation
            for ep_rank in range(ep_ranks):
                for expert in range(experts_per_rank):
                    for tp_rank in range(tp_ranks):
                        expert_name = f'{layer_name}_Expert_{ep_rank*experts_per_rank+expert}_EP{ep_rank}_TP{tp_rank}_PP{pp_stage}'
                        dag.edge(a2a_dispatch_name, expert_name)
                        
                        # Connect to A2A combine
                        a2a_combine_name = f'{layer_name}_A2A_Combine_PP{pp_stage}'
                        dag.edge(expert_name, a2a_combine_name)
            
            # MLP output flow
            for tp_rank in range(tp_ranks):
                mlp_out_name = f'{layer_name}_MLP_Out_TP{tp_rank}_PP{pp_stage}'
                dag.edge(a2a_combine_name, mlp_out_name)
                
                # All-reduce for MLP
                ar_mlp_name = f'{layer_name}_MLP_AllReduce_PP{pp_stage}'
                dag.edge(mlp_out_name, ar_mlp_name)
            
            # LayerNorm
            layernorm_name = f'{layer_name}_LayerNorm_PP{pp_stage}'
            dag.edge(ar_mlp_name, layernorm_name)
            
            # Connect to next layer or next stage
            if layer < layers_per_stage - 1:
                # Next layer in same stage
                next_layer = global_layer + 1
                next_qkv_name = f'Layer_{next_layer}_QKV_TP0_PP{pp_stage}'
                dag.edge(layernorm_name, next_qkv_name)
            elif pp_stage < pp_stages - 1:
                # Next stage
                next_qkv_name = f'Layer_{global_layer+1}_QKV_TP0_PP{pp_stage+1}'
                dag.edge(layernorm_name, next_qkv_name)
    
    # Output node
    dag.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect last layer to output
    dag.edge('Layer_15_LayerNorm_PP3', 'output')
    
    return dag

def main():
    # Create output directory if it doesn't exist
    output_dir = '../outputs/2025-12-24-15-52-53'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate DAG
    dag = create_dag()
    
    # Save DOT file
    dot_file = os.path.join(output_dir, 'llm_parallel_strategy_dag.dot')
    dag.save(dot_file)
    print(f"DOT file saved: {dot_file}")
    
    # Save SVG file
    svg_file = os.path.join(output_dir, 'llm_parallel_strategy_dag.svg')
    dag.render(os.path.join(output_dir, 'llm_parallel_strategy_dag'), format='svg', cleanup=True)
    print(f"SVG file saved: {svg_file}")
    
    # Also save as PNG for easier viewing
    png_file = os.path.join(output_dir, 'llm_parallel_strategy_dag.png')
    dag.render(os.path.join(output_dir, 'llm_parallel_strategy_dag'), format='png', cleanup=True)
    print(f"PNG file saved: {png_file}")
    
    print("DAG generation completed successfully!")

if __name__ == '__main__':
    main()