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
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontsize='9', margin='0.05,0.02')
    
    # Define styles
    compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    comm_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen'}
    aggregation_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightcoral'}
    
    # Configuration
    config = {
        'ep_degree': 64, 'tp_degree': 8, 'pp_degree': 2, 'dp_degree': 2,
        'num_layers': 16, 'num_experts': 64, 'hidden_size': 1024,
        'moe_hidden_size': 2048, 'num_heads': 16, 'head_dim': 64, 
        'batch_size': 128, 'seq_length': 1024
    }
    
    # Input node
    dot.node('input', 
             f'Model Input\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
             {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '2'})
    
    # Create pipeline stages
    layers_per_stage = config['num_layers'] // config['pp_degree']
    
    for pp_rank in range(config['pp_degree']):
        with dot.subgraph(name=f'cluster_pp_{pp_rank}') as c:
            c.attr(label=f'Pipeline Stage {pp_rank}\\nLayers {pp_rank*layers_per_stage}-{(pp_rank+1)*layers_per_stage-1}\\nPP Rank: {pp_rank}')
            c.attr(style='rounded, filled', bgcolor='lightgray', penwidth='2')
            
            # Show representative layers (first and last of this stage)
            if pp_rank == 0:
                # First stage - show first 2 layers
                for layer_idx in [0, 1]:
                    global_layer_id = pp_rank * layers_per_stage + layer_idx
                    create_simplified_layer(c, global_layer_id, config, pp_rank, 
                                          compute_style, comm_style, routing_style, aggregation_style)
                # Add ellipsis for remaining layers
                c.node(f'ellipsis_pp{pp_rank}', '...', {'shape': 'none'})
            else:
                # Second stage - show last 2 layers
                for layer_idx in [layers_per_stage-2, layers_per_stage-1]:
                    global_layer_id = pp_rank * layers_per_stage + layer_idx
                    create_simplified_layer(c, global_layer_id, config, pp_rank,
                                          compute_style, comm_style, routing_style, aggregation_style)
    
    # Pipeline communication between stages
    dot.node('pp_comm_0_1', 
            f'PP Communication\\nSend/Recv All-Reduce\\nStage 0 → 1',
            comm_style)
    
    # Output node
    dot.node('output', 
             f'Model Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, vocab=32000]',
             {'shape': 'doublecircle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '2'})
    
    # Connect pipeline stages
    dot.edge('layer_1_output', 'pp_comm_0_1')
    dot.edge('pp_comm_0_1', 'layer_14_input')
    dot.edge('layer_15_output', 'output')
    
    return dot

def create_simplified_layer(dot, layer_id: int, config: dict, pp_rank: int, 
                          compute_style: dict, comm_style: dict, 
                          routing_style: dict, aggregation_style: dict):
    """Create simplified but complete layer representation"""
    
    with dot.subgraph(name=f'cluster_layer_{layer_id}') as c:
        c.attr(label=f'Layer {layer_id}\\nPP Rank: {pp_rank}')
        c.attr(style='dashed, rounded', bgcolor='white', penwidth='1')
        
        # Layer input
        c.node(f'layer_{layer_id}_input', 
               f'Layer {layer_id} Input\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
               {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.4', 'height': '0.4'})
        
        # Attention Block
        with c.subgraph(name=f'cluster_attn_{layer_id}') as attn:
            attn.attr(label=f'Attention Block (TP8)')
            attn.attr(style='rounded, filled', bgcolor='lightblue', penwidth='1')
            
            # QKV Projection with TP8
            attn.node(f'qkv_{layer_id}', 
                     f'QKV Projection\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [3×heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: All TP Ranks',
                     compute_style)
            
            # Attention computation
            attn.node(f'attn_comp_{layer_id}', 
                     f'Attention Scores+Softmax\\n[batch={config["batch_size"]}, heads={config["num_heads"]}, seq={config["seq_length"]}]\\nGPU: All TP Ranks',
                     compute_style)
            
            # TP All-Reduce
            attn.node(f'attn_ar_{layer_id}', 
                     f'TP All-Reduce\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: All TP Ranks',
                     comm_style)
            
            # Output projection
            attn.node(f'attn_proj_{layer_id}', 
                     f'Output Projection\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: All TP Ranks',
                     compute_style)
            
            attn.node(f'attn_out_{layer_id}', 
                     f'Attention Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
                     {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.3', 'height': '0.3'})
            
            # Connect attention flow
            attn.edge(f'qkv_{layer_id}', f'attn_comp_{layer_id}')
            attn.edge(f'attn_comp_{layer_id}', f'attn_ar_{layer_id}')
            attn.edge(f'attn_ar_{layer_id}', f'attn_proj_{layer_id}')
            attn.edge(f'attn_proj_{layer_id}', f'attn_out_{layer_id}')
        
        # MoE Block
        with c.subgraph(name=f'cluster_moe_{layer_id}') as moe:
            moe.attr(label=f'MoE Block (EP64)')
            moe.attr(style='rounded, filled', bgcolor='lightgreen', penwidth='1')
            
            # Gate selection
            moe.node(f'gate_{layer_id}', 
                    f'Gate Selection\\nTop-2 Experts\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [num_experts={config["num_experts"]}]\\nGPU: All EP Ranks',
                    routing_style)
            
            # Expert dispatch (All-to-All)
            moe.node(f'dispatch_{layer_id}', 
                    f'Expert Dispatch\\nAll-to-All\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\nGPU: All EP Ranks',
                    comm_style)
            
            # Expert computation (show representative experts)
            with moe.subgraph(name=f'cluster_experts_{layer_id}') as experts:
                experts.attr(label=f'Expert Computation (64 Experts Total, TP8 Each)')
                experts.attr(style='rounded, filled', bgcolor='lightyellow', penwidth='1')
                
                # Show first 2 experts as representatives
                for expert_id in [0, 1]:
                    experts.node(f'expert_{layer_id}_{expert_id}', 
                                f'Expert {expert_id} MLP\\n[batch=?, seq=?, hidden={config["hidden_size"]}]\\n→ [{config["moe_hidden_size"]}] → [{config["hidden_size"]}]\\nGPU: EP{expert_id}, TP0-TP7',
                                compute_style)
                    
                    experts.node(f'expert_ar_{layer_id}_{expert_id}', 
                                f'Expert {expert_id} TP All-Reduce\\n[batch=?, seq=?, hidden={config["hidden_size"]}]\\nGPU: EP{expert_id}, TP0-TP7',
                                comm_style)
                    
                    experts.edge(f'expert_{layer_id}_{expert_id}', f'expert_ar_{layer_id}_{expert_id}')
                
                experts.node(f'expert_ellipsis_{layer_id}', '...', {'shape': 'none'})
            
            # Expert combine (All-to-All)
            moe.node(f'combine_{layer_id}', 
                    f'Expert Combine\\nAll-to-All\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\nGPU: All EP Ranks',
                    comm_style)
            
            # Weighted aggregation
            moe.node(f'aggregate_{layer_id}', 
                    f'Expert Output Aggregation\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: All EP Ranks',
                    aggregation_style)
            
            moe.node(f'moe_out_{layer_id}', 
                    f'MoE Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
                    {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.3', 'height': '0.3'})
            
            # Connect MoE flow
            moe.edge(f'gate_{layer_id}', f'dispatch_{layer_id}')
            moe.edge(f'dispatch_{layer_id}', f'expert_{layer_id}_0')
            moe.edge(f'dispatch_{layer_id}', f'expert_{layer_id}_1')
            moe.edge(f'expert_ar_{layer_id}_0', f'combine_{layer_id}')
            moe.edge(f'expert_ar_{layer_id}_1', f'combine_{layer_id}')
            moe.edge(f'combine_{layer_id}', f'aggregate_{layer_id}')
            moe.edge(f'aggregate_{layer_id}', f'moe_out_{layer_id}')
            
            # Gate selection with dashed lines (top-2 experts)
            moe.edge(f'gate_{layer_id}', f'expert_{layer_id}_0', style='dashed', label='Top-2')
            moe.edge(f'gate_{layer_id}', f'expert_{layer_id}_1', style='dashed', label='Top-2')
        
        # Layer output
        c.node(f'layer_{layer_id}_output', 
               f'Layer {layer_id} Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
               {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.4', 'height': '0.4'})
        
        # Connect layer flow
        c.edge(f'layer_{layer_id}_input', f'qkv_{layer_id}')
        c.edge(f'attn_out_{layer_id}', f'gate_{layer_id}')
        c.edge(f'moe_out_{layer_id}', f'layer_{layer_id}_output')

def create_detailed_single_layer_dag():
    """Create a highly detailed single layer DAG for close inspection"""
    
    dot = graphviz.Digraph('MoE_Single_Layer_Detailed')
    dot.attr(rankdir='TB', size='25,35', dpi='300')
    dot.attr('node', fontsize='8', margin='0.03,0.01')
    
    # Define styles
    compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    comm_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen'}
    aggregation_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightcoral'}
    
    config = {
        'ep_degree': 64, 'tp_degree': 8, 'pp_degree': 2, 'dp_degree': 2,
        'num_layers': 16, 'num_experts': 64, 'hidden_size': 1024,
        'moe_hidden_size': 2048, 'num_heads': 16, 'head_dim': 64, 
        'batch_size': 128, 'seq_length': 1024
    }
    
    # Input
    dot.node('input', 
             f'Layer Input\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
             {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '2'})
    
    # Attention Block - Detailed
    with dot.subgraph(name='cluster_attn_detailed') as attn:
        attn.attr(label='Attention Block - Detailed Operator Breakdown')
        attn.attr(style='rounded, filled', bgcolor='lightblue', penwidth='2')
        
        # Q projection with TP8
        attn.node('q_proj', 
                 f'Q Projection (TP8)\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7 (1/8 each)',
                 compute_style)
        
        # K projection with TP8
        attn.node('k_proj', 
                 f'K Projection (TP8)\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7 (1/8 each)',
                 compute_style)
        
        # V projection with TP8
        attn.node('v_proj', 
                 f'V Projection (TP8)\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [heads={config["num_heads"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7 (1/8 each)',
                 compute_style)
        
        # TP All-Gather for Q, K, V
        attn.node('q_ag', 
                 f'TP All-Gather Q\\nComplete Q tensor\\nGPU: TP0-TP7',
                 comm_style)
        attn.node('k_ag', 
                 f'TP All-Gather K\\nComplete K tensor\\nGPU: TP0-TP7',
                 comm_style)
        attn.node('v_ag', 
                 f'TP All-Gather V\\nComplete V tensor\\nGPU: TP0-TP7',
                 comm_style)
        
        # Attention scores computation
        attn.node('attn_scores', 
                 f'Attention Scores\\nQ × K^T / sqrt(d_k)\\n[batch={config["batch_size"]}, heads={config["num_heads"]}, seq_q={config["seq_length"]}, seq_k={config["seq_length"]}]\\nGPU: TP0-TP7',
                 compute_style)
        
        # Attention mask (causal)
        attn.node('attn_mask', 
                 f'Causal Mask\\nTriangular mask\\n[seq={config["seq_length"]}, seq={config["seq_length"]}]',
                 compute_style)
        
        # Softmax
        attn.node('attn_softmax', 
                 f'Softmax\\n[batch={config["batch_size"]}, heads={config["num_heads"]}, seq={config["seq_length"]}, seq={config["seq_length"]}]\\nGPU: TP0-TP7',
                 compute_style)
        
        # Attention output (weighted sum)
        attn.node('attn_output', 
                 f'Attention Output\\nWeights × V\\n[batch={config["batch_size"]}, heads={config["num_heads"]}, seq={config["seq_length"]}, d_k={config["head_dim"]}]\\nGPU: TP0-TP7',
                 compute_style)
        
        # TP All-Reduce for attention output
        attn.node('attn_ar', 
                 f'TP All-Reduce\\nSum partial results\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: TP0-TP7',
                 comm_style)
        
        # Output projection
        attn.node('out_proj', 
                 f'Output Projection (TP8)\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\nGPU: TP0-TP7 (1/8 each)',
                 compute_style)
        
        # Final TP All-Reduce
        attn.node('final_ar', 
                 f'Final TP All-Reduce\\nComplete attention output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: TP0-TP7',
                 comm_style)
        
        attn.node('attn_final', 
                 f'Attention Block Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
                 {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '1'})
        
        # Connect attention detailed flow
        attn.edge('input', 'q_proj')
        attn.edge('input', 'k_proj')
        attn.edge('input', 'v_proj')
        attn.edge('q_proj', 'q_ag')
        attn.edge('k_proj', 'k_ag')
        attn.edge('v_proj', 'v_ag')
        attn.edge('q_ag', 'attn_scores')
        attn.edge('k_ag', 'attn_scores')
        attn.edge('attn_scores', 'attn_mask')
        attn.edge('attn_mask', 'attn_softmax')
        attn.edge('attn_softmax', 'attn_output')
        attn.edge('v_ag', 'attn_output')
        attn.edge('attn_output', 'attn_ar')
        attn.edge('attn_ar', 'out_proj')
        attn.edge('out_proj', 'final_ar')
        attn.edge('final_ar', 'attn_final')
    
    # MoE Block - Detailed
    with dot.subgraph(name='cluster_moe_detailed') as moe:
        moe.attr(label='MoE Block - Detailed Expert Parallelism')
        moe.attr(style='rounded, filled', bgcolor='lightgreen', penwidth='2')
        
        # Input to MoE
        moe.node('moe_input', 
                f'MoE Input\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
                {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '1'})
        
        # Gate computation
        moe.node('gate_compute', 
                f'Gate Computation\\nLinear + Softmax\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\n→ [num_experts={config["num_experts"]}]\\nGPU: All EP Ranks',
                routing_style)
        
        # Top-2 expert selection
        moe.node('top2_select', 
                f'Top-2 Expert Selection\\nSelect highest scoring experts\\nGPU: All EP Ranks',
                routing_style)
        
        # Expert dispatch (All-to-All)
        moe.node('dispatch', 
                f'Expert Dispatch\\nAll-to-All Communication\\nSend tokens to expert GPUs\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\nGPU: EP0-EP63',
                comm_style)
        
        # Show representative experts (first 4)
        for expert_id in range(4):
            ep_rank = expert_id
            with moe.subgraph(name=f'cluster_expert_{expert_id}') as exp:
                exp.attr(label=f'Expert {expert_id} (EP Rank {ep_rank})')
                exp.attr(style='rounded, filled', bgcolor='lightyellow', penwidth='1')
                
                # Expert receives tokens
                exp.node(f'expert_recv_{expert_id}', 
                        f'Expert {expert_id} Receive\\nSubset of tokens\\nGPU: EP{ep_rank}',
                        {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.3', 'height': '0.3'})
                
                # Expert MLP first layer
                exp.node(f'expert_mlp1_{expert_id}', 
                        f'Expert {expert_id} MLP Layer 1 (TP8)\\n[hidden={config["hidden_size"]}]\\n→ [{config["moe_hidden_size"]}]\\nGPU: EP{ep_rank}, TP0-TP7 (1/8 each)',
                        compute_style)
                
                # Activation function
                exp.node(f'expert_act_{expert_id}', 
                        f'Activation (SiLU/GeLU)\\nGPU: EP{ep_rank}, TP0-TP7',
                        compute_style)
                
                # Expert MLP second layer
                exp.node(f'expert_mlp2_{expert_id}', 
                        f'Expert {expert_id} MLP Layer 2 (TP8)\\n[{config["moe_hidden_size"]}]\\n→ [hidden={config["hidden_size"]}]\\nGPU: EP{expert_id}, TP0-TP7 (1/8 each)',
                        compute_style)
                
                # Expert TP All-Reduce
                exp.node(f'expert_ar_{expert_id}', 
                        f'Expert {expert_id} TP All-Reduce\\n[hidden={config["hidden_size"]}]\\nGPU: EP{ep_rank}, TP0-TP7',
                        comm_style)
                
                exp.node(f'expert_done_{expert_id}', 
                        f'Expert {expert_id} Output\\n[hidden={config["hidden_size"]}]',
                        {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'width': '0.3', 'height': '0.3'})
                
                exp.edge(f'expert_recv_{expert_id}', f'expert_mlp1_{expert_id}')
                exp.edge(f'expert_mlp1_{expert_id}', f'expert_act_{expert_id}')
                exp.edge(f'expert_act_{expert_id}', f'expert_mlp2_{expert_id}')
                exp.edge(f'expert_mlp2_{expert_id}', f'expert_ar_{expert_id}')
                exp.edge(f'expert_ar_{expert_id}', f'expert_done_{expert_id}')
        
        # Ellipsis for remaining experts
        moe.node('experts_ellipsis', '...\\n60 more experts\\nEP4-EP63', {'shape': 'none'})
        
        # Expert combine (All-to-All)
        moe.node('combine', 
                f'Expert Combine\\nAll-to-All Communication\\nGather expert outputs\\n[batch={config["batch_size"]}, seq={config["seq_length"]}]\\nGPU: EP0-EP63',
                comm_style)
        
        # Weighted aggregation (multiply by gate scores)
        moe.node('weighted_sum', 
                f'Weighted Aggregation\\nGate scores × Expert outputs\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]\\nGPU: All EP Ranks',
                aggregation_style)
        
        moe.node('moe_final', 
                f'MoE Block Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
                {'shape': 'circle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '1'})
        
        # Connect MoE detailed flow
        moe.edge('attn_final', 'moe_input')
        moe.edge('moe_input', 'gate_compute')
        moe.edge('gate_compute', 'top2_select')
        moe.edge('top2_select', 'dispatch')
        moe.edge('dispatch', 'expert_recv_0')
        moe.edge('dispatch', 'expert_recv_1')
        moe.edge('dispatch', 'expert_recv_2')
        moe.edge('dispatch', 'expert_recv_3')
        moe.edge('expert_done_0', 'combine')
        moe.edge('expert_done_1', 'combine')
        moe.edge('expert_done_2', 'combine')
        moe.edge('expert_done_3', 'combine')
        moe.edge('combine', 'weighted_sum')
        moe.edge('weighted_sum', 'moe_final')
        
        # Gate selection with dashed lines (top-2 experts)
        moe.edge('top2_select', 'expert_recv_0', style='dashed', label='Top-2 Selection')
        moe.edge('top2_select', 'expert_recv_1', style='dashed', label='Top-2 Selection')
        moe.edge('top2_select', 'expert_recv_2', style='dashed', label='Top-2 Selection')
        moe.edge('top2_select', 'expert_recv_3', style='dashed', label='Top-2 Selection')
    
    # Output
    dot.node('output', 
             f'Layer Output\\n[batch={config["batch_size"]}, seq={config["seq_length"]}, hidden={config["hidden_size"]}]',
             {'shape': 'doublecircle', 'style': 'filled', 'fillcolor': 'white', 'penwidth': '2'})
    
    dot.edge('moe_final', 'output')
    
    return dot

def main():
    """Generate and save the complete DAG"""
    
    print("Generating 30B MoE deployment DAG...")
    
    # Create the main deployment DAG
    dag = create_moe_deployment_dag()
    
    # Save as DOT file
    dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_deployment_ep64_tp8_pp2_dp2.dot"
    dag.save(dot_path.replace('.dot', ''))
    print(f"Main DOT file saved: {dot_path}")
    
    # Render as SVG
    svg_path = "../outputs/2025-12-22-10-42-54/moe_30b_deployment_ep64_tp8_pp2_dp2.svg"
    dag.render(format='svg', cleanup=True)
    print(f"Main SVG file saved: {svg_path}")
    
    # Create detailed single layer DAG
    detailed_dag = create_detailed_single_layer_dag()
    detailed_dot_path = "../outputs/2025-12-22-10-42-54/moe_30b_single_layer_detailed.dot"
    detailed_svg_path = "../outputs/2025-12-22-10-42-54/moe_30b_single_layer_detailed.svg"
    detailed_dag.save(detailed_dot_path.replace('.dot', ''))
    detailed_dag.render(format='svg', cleanup=True)
    print(f"Detailed DOT file saved: {detailed_dot_path}")
    print(f"Detailed SVG file saved: {detailed_svg_path}")
    
    return {
        'main_dot': dot_path,
        'main_svg': svg_path,
        'detailed_dot': detailed_dot_path,
        'detailed_svg': detailed_svg_path
    }

if __name__ == "__main__":
    paths = main()
    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")