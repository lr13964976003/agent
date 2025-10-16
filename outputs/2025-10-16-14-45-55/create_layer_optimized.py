#!/usr/bin/env python3

import graphviz
import os

def create_layer_optimized_dag(layer_num, output_dir):
    """Create optimized DAG for individual layer"""
    dot = graphviz.Digraph(f'layer_{layer_num}_optimized', 
                          comment=f'Layer {layer_num} Optimized Deployment',
                          format='svg')
    
    dot.attr(rankdir='TB', ranksep='0.8', nodesep='0.3')
    
    # Main cluster for layer
    with dot.subgraph(name=f'cluster_layer_{layer_num}') as layer:
        layer.attr(label=f'Layer {layer_num} (Optimized Pipeline)',
                  style='dashed', color='purple')
        
        # Attention block (GPUs 0-7)
        with layer.subgraph(name=f'cluster_l{layer_num}_attention') as attn:
            attn.attr(label='Attention Block (GPUs 0-7)', style='rounded', color='red')
            
            # Input processing
            attn.node(f'input_l{layer_num}', f'Layer {layer_num} Input\n[1024,2048,4096]', 
                     shape='ellipse', fillcolor='lightgreen')
            
            # Token split across GPUs
            attn.node(f'token_split_l{layer_num}', 
                     f'Token Split\n[1024,2048,4096] → 8×[128,2048,4096]\nGPU: 0-7',
                     shape='parallelogram', fillcolor='lightcyan')
            
            # QKV projections with tensor parallelism
            for gpu in range(8):
                attn.node(f'qkv_l{layer_num}_g{gpu}', 
                         f'QKV Projection\nGPU {gpu}\n[128,2048,4096]×[4096,512]\n→ [128,2048,512]',
                         fillcolor='yellow')
            
            # Multi-head attention
            for gpu in range(8):
                attn.node(f'mha_l{layer_num}_g{gpu}', 
                         f'Multi-Head Attention\nGPU {gpu}\n[128,2048,512]\n16 heads per GPU',
                         fillcolor='yellow')
            
            # Output projections
            for gpu in range(8):
                attn.node(f'out_l{layer_num}_g{gpu}', 
                         f'Output Projection\nGPU {gpu}\n[128,2048,512]×[512,4096]\n→ [128,2048,4096]',
                         fillcolor='yellow')
            
            # Token gather
            attn.node(f'gather_l{layer_num}', 
                     f'Token Gather\n8×[128,2048,4096] → [1024,2048,4096]\nGPU: 0-7',
                     shape='parallelogram', fillcolor='lightcyan')
            
            # Residual connection
            attn.node(f'residual_l{layer_num}', 
                     f'Residual Add\n[1024,2048,4096]\nGPU: 0-7',
                     fillcolor='orange')
        
        # MoE block (GPUs 8-15)
        with layer.subgraph(name=f'cluster_l{layer_num}_moe') as moe:
            moe.attr(label='MoE Block (GPUs 8-15)', style='rounded', color='blue')
            
            # Gate network
            moe.node(f'gate_l{layer_num}', 
                    f'Gate Network\n[1024,2048,4096]×[4096,16]\n→ [1024,2048,16]\nGPU: 8-15',
                    fillcolor='lightcoral')
            
            # Expert routing
            moe.node(f'routing_l{layer_num}', 
                    f'Expert Routing\n[1024,2048,4096] → 8×[128,2048,4096]\nGPU: 8-15',
                    shape='parallelogram', fillcolor='lightcyan')
            
            # Expert computations (one per GPU)
            for gpu in range(8):
                moe.node(f'expert_l{layer_num}_g{gpu+8}', 
                        f'Expert {gpu}\n[128,2048,4096]×[4096,4096]\n→ [128,2048,4096]\nGPU: {gpu+8}',
                        fillcolor='lightpink')
            
            # Expert aggregation
            moe.node(f'agg_l{layer_num}', 
                    f'Expert Aggregation\n8×[128,2048,4096] → [1024,2048,4096]\nGPU: 8-15',
                    shape='parallelogram', fillcolor='lightcyan')
            
            # MoE residual
            moe.node(f'moe_residual_l{layer_num}', 
                    f'MoE Residual Add\n[1024,2048,4096]\nGPU: 8-15',
                    fillcolor='orange')
    
    # Connections
    dot.edge(f'input_l{layer_num}', f'token_split_l{layer_num}')
    
    # Connect all GPUs in attention block
    for gpu in range(8):
        dot.edge(f'token_split_l{layer_num}', f'qkv_l{layer_num}_g{gpu}')
        dot.edge(f'qkv_l{layer_num}_g{gpu}', f'mha_l{layer_num}_g{gpu}')
        dot.edge(f'mha_l{layer_num}_g{gpu}', f'out_l{layer_num}_g{gpu}')
        dot.edge(f'out_l{layer_num}_g{gpu}', f'gather_l{layer_num}')
    
    dot.edge(f'gather_l{layer_num}', f'residual_l{layer_num}')
    dot.edge(f'residual_l{layer_num}', f'gate_l{layer_num}')
    dot.edge(f'residual_l{layer_num}', f'routing_l{layer_num}')
    dot.edge(f'residual_l{layer_num}', f'moe_residual_l{layer_num}')
    
    # Expert routing connections
    for gpu in range(8):
        dot.edge(f'routing_l{layer_num}', f'expert_l{layer_num}_g{gpu+8}')
        dot.edge(f'expert_l{layer_num}_g{gpu+8}', f'agg_l{layer_num}')
    
    dot.edge(f'agg_l{layer_num}', f'moe_residual_l{layer_num}')
    
    # Save files
    dot.save(os.path.join(output_dir, f'layer_{layer_num}_optimized.dot'))
    dot.render(os.path.join(output_dir, f'layer_{layer_num}_optimized'), 
               format='svg', cleanup=True)

# Generate all layers
if __name__ == "__main__":
    output_dir = "../outputs/2025-10-16-14-45-55"
    
    for layer in range(4):
        create_layer_optimized_dag(layer, output_dir)
        print(f"Generated Layer {layer} optimized DAG")