#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_parallel_strategy_dag():
    """
    Generate a complete DAG for the LLM parallel strategy deployment.
    
    Configuration:
    - 128 GPUs total (4 DP replicas × 2 PP stages × 16 GPUs per stage)
    - EP=16: 16 experts per layer, 1 per GPU
    - PP=2: 2 pipeline stages with 8 layers each  
    - TP=4: 4-way tensor parallelism for attention
    - DP=4: 4 independent replicas
    """
    
    dot = Digraph(comment='LLM Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]', 
             shape='ellipse', fillcolor='lightblue')
    
    # DP Level - 4 replicas
    for dp_id in range(4):
        dp_cluster = f'dp_{dp_id}'
        
        with dot.subgraph(name=f'cluster_dp_{dp_id}') as dp_subgraph:
            dp_subgraph.attr(label=f'DP Replica {dp_id} (GPUs {dp_id*32}-{(dp_id+1)*32-1})', 
                           style='rounded', bgcolor='lightgray')
            
            # PP Stage 1 (Layers 1-8)
            with dp_subgraph.subgraph(name=f'cluster_pp1_dp{dp_id}') as pp1_subgraph:
                pp1_subgraph.attr(label=f'PP Stage 1: Layers 1-8 (GPUs {dp_id*32}-{dp_id*32+15})', 
                                style='rounded', bgcolor='lightcyan')
                
                # Input to Stage 1
                if dp_id == 0:
                    dot.edge('input', f'input_dp{dp_id}_pp1', style='dashed', constraint='false')
                
                # Layer 1-8 processing
                for layer_id in range(1, 9):
                    layer_base = f'dp{dp_id}_pp1_layer{layer_id}'
                    
                    # Attention with TP=4
                    with pp1_subgraph.subgraph(name=f'cluster_{layer_base}_attn') as attn_subgraph:
                        attn_subgraph.attr(label=f'Layer {layer_id} Attention (TP=4)', 
                                         style='rounded', bgcolor='lightpink')
                        
                        # TP shards for attention
                        for tp_id in range(4):
                            gpu_id = dp_id * 32 + tp_id
                            attn_node = f'{layer_base}_attn_tp{tp_id}'
                            attn_subgraph.node(attn_node, 
                                             f'Attention TP{tp_id}\\nGPU {gpu_id}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=4, d_k=32]',
                                             shape='rectangle', fillcolor='lightgreen')
                        
                        # AllReduce for attention
                        allreduce_attn = f'{layer_base}_attn_allreduce'
                        attn_subgraph.node(allreduce_attn, 
                                         f'AllReduce Attention\\nGPUs {dp_id*32}-{dp_id*32+3}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                         shape='ellipse', fillcolor='lightblue')
                        
                        # Connect TP shards to AllReduce
                        for tp_id in range(4):
                            dot.edge(f'{layer_base}_attn_tp{tp_id}', allreduce_attn)
                    
                    # MoE with EP=16
                    with pp1_subgraph.subgraph(name=f'cluster_{layer_base}_moe') as moe_subgraph:
                        moe_subgraph.attr(label=f'Layer {layer_id} MoE (EP=16)', 
                                        style='rounded', bgcolor='lightyellow')
                        
                        # Router
                        router_node = f'{layer_base}_router'
                        moe_subgraph.node(router_node, 
                                        f'Router\\nGPU {dp_id*32}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                        shape='parallelogram', fillcolor='lightyellow')
                        
                        # Expert selection (dashed lines)
                        for expert_id in range(16):
                            gpu_id = dp_id * 32 + expert_id
                            expert_node = f'{layer_base}_expert{expert_id}'
                            moe_subgraph.node(expert_node, 
                                            f'Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=?, hidden=512]\\nOutput: [batch_size=?, seq_len=?, hidden=512]',
                                            shape='rectangle', fillcolor='lightgreen')
                            # Dashed line for expert selection
                            dot.edge(router_node, expert_node, style='dashed')
                        
                        # Expert aggregation
                        expert_agg = f'{layer_base}_expert_agg'
                        moe_subgraph.node(expert_agg, 
                                        f'Expert Aggregation\\nGPUs {dp_id*32}-{dp_id*32+15}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                        shape='parallelogram', fillcolor='lightyellow')
                        
                        # Connect experts to aggregation
                        for expert_id in range(16):
                            dot.edge(f'{layer_base}_expert{expert_id}', expert_agg)
                    
                    # Connect attention to router
                    dot.edge(f'{layer_base}_attn_allreduce', router_node, constraint='false')
                    
                    # Connect expert aggregation to next layer or stage transition
                    if layer_id < 8:
                        next_layer_base = f'dp{dp_id}_pp1_layer{layer_id+1}'
                        dot.edge(expert_agg, f'{next_layer_base}_attn_tp0', constraint='false')
            
            # PP Stage 2 (Layers 9-16)
            with dp_subgraph.subgraph(name=f'cluster_pp2_dp{dp_id}') as pp2_subgraph:
                pp2_subgraph.attr(label=f'PP Stage 2: Layers 9-16 (GPUs {dp_id*32+16}-{dp_id*32+31})', 
                                style='rounded', bgcolor='lightcyan')
                
                # Layer 9-16 processing
                for layer_id in range(9, 17):
                    layer_base = f'dp{dp_id}_pp2_layer{layer_id}'
                    
                    # Attention with TP=4
                    with pp2_subgraph.subgraph(name=f'cluster_{layer_base}_attn') as attn_subgraph:
                        attn_subgraph.attr(label=f'Layer {layer_id} Attention (TP=4)', 
                                         style='rounded', bgcolor='lightpink')
                        
                        # TP shards for attention
                        for tp_id in range(4):
                            gpu_id = dp_id * 32 + 16 + tp_id
                            attn_node = f'{layer_base}_attn_tp{tp_id}'
                            attn_subgraph.node(attn_node, 
                                             f'Attention TP{tp_id}\\nGPU {gpu_id}\\nInput: [batch_size=1, seq_len=1024, heads=4, d_k=32]\\nOutput: [batch_size=1, seq_len=1024, heads=4, d_k=32]',
                                             shape='rectangle', fillcolor='lightgreen')
                        
                        # AllReduce for attention
                        allreduce_attn = f'{layer_base}_attn_allreduce'
                        attn_subgraph.node(allreduce_attn, 
                                         f'AllReduce Attention\\nGPUs {dp_id*32+16}-{dp_id*32+19}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                         shape='ellipse', fillcolor='lightblue')
                        
                        # Connect TP shards to AllReduce
                        for tp_id in range(4):
                            dot.edge(f'{layer_base}_attn_tp{tp_id}', allreduce_attn)
                    
                    # MoE with EP=16
                    with pp2_subgraph.subgraph(name=f'cluster_{layer_base}_moe') as moe_subgraph:
                        moe_subgraph.attr(label=f'Layer {layer_id} MoE (EP=16)', 
                                        style='rounded', bgcolor='lightyellow')
                        
                        # Router
                        router_node = f'{layer_base}_router'
                        moe_subgraph.node(router_node, 
                                        f'Router\\nGPU {dp_id*32+16}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                        shape='parallelogram', fillcolor='lightyellow')
                        
                        # Expert selection (dashed lines)
                        for expert_id in range(16):
                            gpu_id = dp_id * 32 + 16 + expert_id
                            expert_node = f'{layer_base}_expert{expert_id}'
                            moe_subgraph.node(expert_node, 
                                            f'Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=?, hidden=512]\\nOutput: [batch_size=?, seq_len=?, hidden=512]',
                                            shape='rectangle', fillcolor='lightgreen')
                            # Dashed line for expert selection
                            dot.edge(router_node, expert_node, style='dashed')
                        
                        # Expert aggregation
                        expert_agg = f'{layer_base}_expert_agg'
                        moe_subgraph.node(expert_agg, 
                                        f'Expert Aggregation\\nGPUs {dp_id*32+16}-{dp_id*32+31}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                                        shape='parallelogram', fillcolor='lightyellow')
                        
                        # Connect experts to aggregation
                        for expert_id in range(16):
                            dot.edge(f'{layer_base}_expert{expert_id}', expert_agg)
                    
                    # Connect attention to router
                    dot.edge(f'{layer_base}_attn_allreduce', router_node, constraint='false')
                    
                    # Connect expert aggregation to next layer or output
                    if layer_id < 16:
                        next_layer_base = f'dp{dp_id}_pp2_layer{layer_id+1}'
                        dot.edge(expert_agg, f'{next_layer_base}_attn_tp0', constraint='false')
    
    # PP Stage transitions
    for dp_id in range(4):
        last_layer_pp1 = f'dp{dp_id}_pp1_layer8_expert_agg'
        first_layer_pp2 = f'dp{dp_id}_pp2_layer9_attn_tp0'
        
        # Communication between stages
        stage_comm = f'dp{dp_id}_pp1_to_pp2'
        dot.node(stage_comm, 
                f'PP Stage Transition\\nGPU {dp_id*32+15} → GPU {dp_id*32+16}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                shape='ellipse', fillcolor='lightblue')
        
        dot.edge(last_layer_pp1, stage_comm)
        dot.edge(stage_comm, first_layer_pp2)
    
    # Output node
    output_nodes = []
    for dp_id in range(4):
        last_layer = f'dp{dp_id}_pp2_layer16_expert_agg'
        output_node = f'output_dp{dp_id}'
        dot.node(output_node, 
                f'Output DP{dp_id}\\nGPU {dp_id*32+31}\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
                shape='ellipse', fillcolor='lightblue')
        dot.edge(last_layer, output_node)
        output_nodes.append(output_node)
    
    # Final output aggregation
    dot.node('final_output', 'Final Output\\nInput: [batch_size=4, seq_len=1024, hidden=512]\\nOutput: [batch_size=4, seq_len=1024, hidden=512]',
             shape='ellipse', fillcolor='lightblue')
    
    for output_node in output_nodes:
        dot.edge(output_node, 'final_output', style='dashed')
    
    return dot

if __name__ == '__main__':
    dag = create_parallel_strategy_dag()
    
    # Save DOT file
    dag.save('./outputs/2025-12-30-10-33-13/parallel_strategy_dag.dot')
    
    # Save SVG image
    dag.render('./outputs/2025-12-30-10-33-13/parallel_strategy_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ./outputs/2025-12-30-10-33-13/parallel_strategy_dag.dot")
    print(f"SVG image: ./outputs/2025-12-30-10-33-13/parallel_strategy_dag.svg")