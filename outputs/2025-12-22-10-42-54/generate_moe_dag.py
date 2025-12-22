#!/usr/bin/env python3
"""
DAG Generator for 30B MoE Model Deployment
==========================================

Generates a comprehensive DAG showing EP64-TP8-PP2-DP2 parallel strategy
with detailed operator-level breakdown and communication patterns.
"""

import graphviz
from typing import Dict, List, Tuple
import os

def create_moe_deployment_dag():
    """Create comprehensive DAG for 30B MoE model deployment"""
    
    # Create main DAG
    dot = graphviz.Digraph('MoE_30B_Deployment_EP64_TP8_PP2_DP2')
    dot.attr(rankdir='TB', size='50,30', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Define styles
    compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    comm_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen'}
    aggregation_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightcoral'}
    
    # Configuration
    config = {
        'ep_degree': 64, 'tp_degree': 8, 'pp_degree': 2, 'dp_degree': 2,
        'num_layers': 16, 'num_experts': 64, 'hidden_size': 1024,
        'num_heads': 16, 'head_dim': 64, 'batch_size': 128, 'seq_length': 1024
    }
    
    # Input node
    dot.node('input', 
             f'Input\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}]',
             {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white'})
    
    # Create pipeline stages
    layers_per_stage = config['num_layers'] // config['pp_degree']
    
    for pp_rank in range(config['pp_degree']):
        with dot.subgraph(name=f'cluster_pp_{pp_rank}') as c:
            c.attr(label=f'Pipeline Stage {pp_rank} (Layers {pp_rank*layers_per_stage}-{(pp_rank+1)*layers_per_stage-1})')
            c.attr(style='rounded', bgcolor='lightgray')
            
            # Create layers within this pipeline stage
            for layer_idx in range(layers_per_stage):
                global_layer_id = pp_rank * layers_per_stage + layer_idx
                create_layer_subgraph(c, global_layer_id, config, pp_rank, compute_style, comm_style, routing_style, aggregation_style)
    
    # Pipeline communication between stages
    if config['pp_degree'] > 1:
        for pp_rank in range(config['pp_degree'] - 1):
            # PP send/recv communication
            dot.node(f'pp_comm_{pp_rank}_{pp_rank+1}', 
                    f'PP Send/Recv\\nStage {pp_rank} -> {pp_rank+1}',
                    comm_style)
            # Connect last layer of stage to communication
            last_layer_stage = (pp_rank + 1) * layers_per_stage - 1
            dot.edge(f'layer_{last_layer_stage}_output', f'pp_comm_{pp_rank}_{pp_rank+1}')
            # Connect communication to first layer of next stage
            first_layer_next = (pp_rank + 1) * layers_per_stage
            dot.edge(f'pp_comm_{pp_rank}_{pp_rank+1}', f'layer_{first_layer_next}_input')
    
    # Output node
    dot.node('output', 
             f'Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, vocab_size=32000]',
             {'shape': 'doublecircle', 'style': 'filled', 'fillcolor': 'white'})
    
    # Connect final pipeline stage to output
    final_layer = config['num_layers'] - 1
    dot.edge(f'layer_{final_layer}_output', 'output')
    
    return dot

def create_layer_subgraph(dot, layer_id: int, config: dict, pp_rank: int, 
                         compute_style: dict, comm_style: dict, 
                         routing_style: dict, aggregation_style: dict):
    """Create detailed subgraph for a single transformer layer"""
    
    with dot.subgraph(name=f'cluster_layer_{layer_id}') as c:
        c.attr(label=f'Layer {layer_id}')
        c.attr(style='dashed', bgcolor='white')
        
        # Layer input
        c.node(f'layer_{layer_id}_input', 
               f'Layer {layer_id} Input\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]',
               {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.5', 'height': '0.5'})
        
        # 1. Attention Block
        create_attention_block(c, layer_id, config, compute_style, comm_style)
        
        # 2. MoE Block
        create_moe_block(c, layer_id, config, pp_rank, compute_style, comm_style, routing_style, aggregation_style)
        
        # Layer output
        c.node(f'layer_{layer_id}_output', 
               f'Layer {layer_id} Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]',
               {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.5', 'height': '0.5'})
        
        # Connect attention to MoE and MoE to output
        c.edge(f'attn_{layer_id}_output', f'moe_{layer_id}_input')
        c.edge(f'moe_{layer_id}_output', f'layer_{layer_id}_output')

def create_attention_block(dot, layer_id: int, config: dict, 
                          compute_style: dict, comm_style: dict):
    """Create detailed attention block with TP decomposition"""
    
    with dot.subgraph(name=f'cluster_attn_{layer_id}') as c:
        c.attr(label=f'Attention Block (Layer {layer_id})')
        c.attr(style='rounded', bgcolor='lightblue', penwidth='2')
        
        # Attention input
        if layer_id == 0:
            c.node(f'attn_{layer_id}_input', 'Attention Input', {'style': 'invis'})
        else:
            c.node(f'attn_{layer_id}_input', f'Attention Input\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]')
        
        # 1. QKV Projection with TP8
        c.node(f'qkv_proj_{layer_id}', 
               f'QKV Projection (TP8)\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}->3×{config["hidden_size"]}]\\nGPU: TP0-TP7',
               compute_style)
        
        # TP All-Gather for QKV
        c.node(f'qkv_ag_{layer_id}', 
               f'TP All-Gather QKV\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7',
               comm_style)
        
        # 2. Attention Scores
        c.node(f'attn_scores_{layer_id}', 
               f'Attention Scores\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7',
               compute_style)
        
        # 3. Attention Weights (Softmax)
        c.node(f'attn_softmax_{layer_id}', 
               f'Attention Softmax\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, heads={config["num_heads"]}]\\nGPU: TP0-TP7',
               compute_style)
        
        # 4. Attention Output
        c.node(f'attn_output_{layer_id}', 
               f'Attention Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7',
               compute_style)
        
        # TP All-Reduce for attention output
        c.node(f'attn_ar_{layer_id}', 
               f'TP All-Reduce Attn Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: TP0-TP7',
               comm_style)
        
        # Output projection with TP8
        c.node(f'attn_proj_{layer_id}', 
               f'Attention Output Projection (TP8)\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}->{config["hidden_size"]}]\\nGPU: TP0-TP7',
               compute_style)
        
        # Final TP All-Reduce
        c.node(f'attn_final_ar_{layer_id}', 
               f'TP All-Reduce Final Attn\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: TP0-TP7',
               comm_style)
        
        c.node(f'attn_{layer_id}_output', f'Attention Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]')
        
        # Connect attention flow
        c.edge(f'attn_{layer_id}_input', f'qkv_proj_{layer_id}')
        c.edge(f'qkv_proj_{layer_id}', f'qkv_ag_{layer_id}')
        c.edge(f'qkv_ag_{layer_id}', f'attn_scores_{layer_id}')
        c.edge(f'attn_scores_{layer_id}', f'attn_softmax_{layer_id}')
        c.edge(f'attn_softmax_{layer_id}', f'attn_output_{layer_id}')
        c.edge(f'attn_output_{layer_id}', f'attn_ar_{layer_id}')
        c.edge(f'attn_ar_{layer_id}', f'attn_proj_{layer_id}')
        c.edge(f'attn_proj_{layer_id}', f'attn_final_ar_{layer_id}')
        c.edge(f'attn_final_ar_{layer_id}', f'attn_{layer_id}_output')

def create_moe_block(dot, layer_id: int, config: dict, pp_rank: int,
                    compute_style: dict, comm_style: dict, 
                    routing_style: dict, aggregation_style: dict):
    """Create detailed MoE block with EP64 decomposition"""
    
    with dot.subgraph(name=f'cluster_moe_{layer_id}') as c:
        c.attr(label=f'MoE Block (Layer {layer_id})')
        c.attr(style='rounded', bgcolor='lightgreen', penwidth='2')
        
        # MoE input
        c.node(f'moe_{layer_id}_input', f'MoE Input\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]')
        
        # 1. Gate/Router - selects top-2 experts
        c.node(f'gate_{layer_id}', 
               f'Gate Selection (Top-2)\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, experts={config["num_experts"]}]\\nGPU: EP0-EP63',
               routing_style)
        
        # 2. Expert dispatch (All-to-All communication)
        c.node(f'expert_dispatch_{layer_id}', 
               f'Expert Dispatch (All-to-All)\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}]\\nGPU: EP0-EP63',
               comm_style)
        
        # 3. Expert computation - show parallel experts
        experts_per_ep = config['num_experts'] // config['ep_degree']
        for ep_rank in range(min(4, config['ep_degree'])):  # Show first 4 EP groups for clarity
            start_expert = ep_rank * experts_per_ep
            end_expert = start_expert + experts_per_ep - 1
            
            with c.subgraph(name=f'cluster_experts_ep{ep_rank}') as exp:
                exp.attr(label=f'EP Rank {ep_rank} (Experts {start_expert}-{end_expert})')
                exp.attr(style='rounded', bgcolor='lightyellow')
                
                for expert_id in range(start_expert, min(start_expert+2, end_expert+1)):
                    # Expert MLP with TP8
                    exp.node(f'expert_{layer_id}_{expert_id}', 
                            f'Expert {expert_id} MLP (TP8)\\n[batch_size=?, seq_len=?, hidden={config["hidden_size"]}->{config["moe_hidden_size"]}->{config["hidden_size"]}]\\nGPU: EP{ep_rank}, TP0-TP7',
                            compute_style)
                    
                    # TP All-Reduce within expert
                    exp.node(f'expert_ar_{layer_id}_{expert_id}', 
                            f'Expert {expert_id} TP All-Reduce\\n[batch_size=?, seq_len=?, hidden={config["hidden_size"]}]\\nGPU: EP{ep_rank}, TP0-TP7',
                            comm_style)
                    
                    exp.edge(f'expert_{layer_id}_{expert_id}', f'expert_ar_{layer_id}_{expert_id}')
        
        # 4. Expert combine (All-to-All communication)
        c.node(f'expert_combine_{layer_id}', 
               f'Expert Combine (All-to-All)\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}]\\nGPU: EP0-EP63',
               comm_style)
        
        # 5. Weighted aggregation of expert outputs
        c.node(f'expert_aggregate_{layer_id}', 
               f'Expert Output Aggregation\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: EP0-EP63',
               aggregation_style)
        
        c.node(f'moe_{layer_id}_output', f'MoE Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}, hidden={config["hidden_size"]}]')
        
        # Connect MoE flow
        c.edge(f'moe_{layer_id}_input', f'gate_{layer_id}')
        c.edge(f'gate_{layer_id}', f'expert_dispatch_{layer_id}')
        c.edge(f'expert_dispatch_{layer_id}', f'expert_{layer_id}_0')
        c.edge(f'expert_dispatch_{layer_id}', f'expert_{layer_id}_1')
        c.edge(f'expert_ar_{layer_id}_0', f'expert_combine_{layer_id}')
        c.edge(f'expert_ar_{layer_id}_1', f'expert_combine_{layer_id}')
        c.edge(f'expert_combine_{layer_id}', f'expert_aggregate_{layer_id}')
        c.edge(f'expert_aggregate_{layer_id}', f'moe_{layer_id}_output')
        
        # Add dashed lines for gate selection (top-2 experts)
        c.edge(f'gate_{layer_id}', f'expert_{layer_id}_0', style='dashed', label='Top-2 Selection')
        c.edge(f'gate_{layer_id}', f'expert_{layer_id}_1', style='dashed', label='Top-2 Selection')

def main():
    """Generate and save the complete DAG"""
    
    print("Generating 30B MoE deployment DAG...")
    
    # Create the DAG
    dag = create_moe_deployment_dag()
    
    # Save as DOT file
    dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_deployment_ep64_tp8_pp2_dp2.dot"
    dag.save(dot_path.replace('.dot', ''))
    print(f"DOT file saved: {dot_path}")
    
    # Render as SVG
    svg_path = "../outputs/2025-12-22-10-42-54/moe_30b_deployment_ep64_tp8_pp2_dp2.svg"
    dag.render(format='svg', cleanup=True)
    print(f"SVG file saved: {svg_path}")
    
    # Also create a simplified version focusing on one layer for clarity
    simple_dag = create_simplified_moe_dag()
    simple_dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_simplified.dot"
    simple_svg_path = "../outputs/2025-12-22-10-42-54/moe_30b_simplified.svg"
    simple_dag.save(simple_dot_path.replace('.dot', ''))
    simple_dag.render(format='svg', cleanup=True)
    print(f"Simplified DOT file saved: {simple_dot_path}")
    print(f"Simplified SVG file saved: {simple_svg_path}")
    
    return {
        'complete_dot': dot_path,
        'complete_svg': svg_path,
        'simplified_dot': simple_dot_path,
        'simplified_svg': simple_svg_path
    }

def create_simplified_moe_dag():
    """Create a simplified DAG showing just one layer for clarity"""
    
    dot = graphviz.Digraph('MoE_30B_Simplified')
    dot.attr(rankdir='TB', size='20,30', dpi='300')
    dot.attr('node', fontsize='10', margin='0.1,0.05')
    
    # Define styles
    compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    comm_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen'}
    aggregation_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightcoral'}
    
    config = {
        'ep_degree': 64, 'tp_degree': 8, 'pp_degree': 2, 'dp_degree': 2,
        'num_layers': 16, 'num_experts': 64, 'hidden_size': 1024,
        'num_heads': 16, 'head_dim': 64, 'batch_size': 128, 'seq_length': 1024
    }
    
    # Input
    dot.node('input', f'Input\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}]')
    
    # Attention block (simplified)
    dot.node('attn_input', 'Attention Input')
    dot.node('qkv_proj', f'QKV Projection (TP8)\\nGPU: TP0-TP7', compute_style)
    dot.node('attn_compute', f'Attention Compute (TP8)\\nGPU: TP0-TP7', compute_style)
    dot.node('attn_ar', f'TP All-Reduce\\nGPU: TP0-TP7', comm_style)
    dot.node('attn_output', 'Attention Output')
    
    # MoE block (simplified but showing EP64)
    dot.node('moe_input', 'MoE Input')
    dot.node('gate', f'Gate Selection (Top-2)\\nGPU: EP0-EP63', routing_style)
    dot.node('dispatch', f'Expert Dispatch (All-to-All)\\nGPU: EP0-EP63', comm_style)
    
    # Show a few representative experts
    for i in range(4):
        dot.node(f'expert_{i}', f'Expert {i} (TP8)\\nGPU: EP{i}, TP0-TP7', compute_style)
        dot.node(f'expert_ar_{i}', f'Expert {i} TP All-Reduce\\nGPU: EP{i}, TP0-TP7', comm_style)
    
    dot.node('combine', f'Expert Combine (All-to-All)\\nGPU: EP0-EP63', comm_style)
    dot.node('aggregate', f'Expert Aggregation\\nGPU: EP0-EP63', aggregation_style)
    dot.node('moe_output', 'MoE Output')
    dot.node('layer_output', 'Layer Output')
    
    # Output
    dot.node('output', f'Final Output\\n[batch_size={config["batch_size"]}, seq_len={config["seq_length"]}]')
    
    # Connections
    dot.edge('input', 'attn_input')
    dot.edge('attn_input', 'qkv_proj')
    dot.edge('qkv_proj', 'attn_compute')
    dot.edge('attn_compute', 'attn_ar')
    dot.edge('attn_ar', 'attn_output')
    dot.edge('attn_output', 'moe_input')
    dot.edge('moe_input', 'gate')
    dot.edge('gate', 'dispatch')
    dot.edge('dispatch', 'expert_0')
    dot.edge('dispatch', 'expert_1')
    dot.edge('dispatch', 'expert_2')
    dot.edge('dispatch', 'expert_3')
    dot.edge('expert_0', 'expert_ar_0')
    dot.edge('expert_1', 'expert_ar_1')
    dot.edge('expert_2', 'expert_ar_2')
    dot.edge('expert_3', 'expert_ar_3')
    dot.edge('expert_ar_0', 'combine')
    dot.edge('expert_ar_1', 'combine')
    dot.edge('expert_ar_2', 'combine')
    dot.edge('expert_ar_3', 'combine')
    dot.edge('combine', 'aggregate')
    dot.edge('aggregate', 'moe_output')
    dot.edge('moe_output', 'layer_output')
    dot.edge('layer_output', 'output')
    
    # Gate selection with dashed lines
    for i in range(4):
        dot.edge('gate', f'expert_{i}', style='dashed', label='Top-2')
    
    return dot

if __name__ == "__main__":
    paths = main()
    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")