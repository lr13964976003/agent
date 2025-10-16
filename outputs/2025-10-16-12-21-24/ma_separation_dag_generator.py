#!/usr/bin/env python3
"""
MA Separation Model DAG Generator
Generates complete deployment DAG for 4-layer MoE model with MA Separation
"""

import os
from graphviz import Digraph

def create_ma_separation_dag():
    dot = Digraph(comment='MA Separation Model Deployment DAG')
    
    # Set graph attributes
    dot.attr(rankdir='TB', compound='true', fontsize='12', fontname='Arial')
    dot.attr('node', fontname='Arial')
    
    # Define node shapes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')   # Communication
    
    # Color coding for GPUs
    gpu_colors = {
        0: '#FFE4E1',   # MistyRose
        1: '#E6E6FA',   # Lavender  
        2: '#F0E68C',   # Khaki
        3: '#98FB98',   # PaleGreen
        4: '#87CEEB',   # SkyBlue
        5: '#DDA0DD',   # Plum
        6: '#F4A460',   # SandyBrown
        7: '#D3D3D3',   # LightGray
        8: '#FFB6C1',   # LightPink
        9: '#20B2AA',   # LightSeaGreen
        10: '#B0C4DE',  # LightSteelBlue
        11: '#FFDEAD',  # NavajoWhite
        12: '#9ACD32',  # YellowGreen
        13: '#D8BFD',   # BurlyWood
        14: '#E0E0E0',  # Gray90
        15: '#FFF8DC'   # Cornsilk
    }
    
    # Global input
    dot.node('input', 'Model Input\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: Host', 
             shape='ellipse', fillcolor='lightblue', style='filled,bold')
    
    # Process through each layer
    for layer in range(4):
        layer_subgraph = Digraph(name=f'cluster_layer_{layer}')
        layer_subgraph.attr(label=f'Layer {layer+1}', style='rounded', bgcolor='lightgray')
        
        # Input to layer
        if layer == 0:
            prev_output = 'input'
        else:
            prev_output = f'layer_{layer}_output'
            
        # Layer input distribution
        distribute_node = f'layer_{layer}_distribute'
        layer_subgraph.node(distribute_node, 
                           f'Layer {layer+1} Input Distribution\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096] replicated 8x\\nGPU: All GPUs', 
                           shape='diamond', fillcolor='lightcoral')
        
        dot.edge(prev_output, distribute_node)
        
        # === ATTENTION COMPONENT (GPUs 0-7) ===
        attention_cluster = Digraph(name=f'cluster_attention_{layer}')
        attention_cluster.attr(label=f'Attention - Layer {layer+1}', style='dashed', bgcolor='#E6F3FF')
        
        # QKV Projection across 8 GPUs
        qkv_nodes = []
        for gpu_id in range(8):
            qkv_node = f'layer_{layer}_qkv_gpu_{gpu_id}'
            attention_cluster.node(qkv_node,
                                 f'QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', 
                                 shape='box', fillcolor=gpu_colors[gpu_id])
            layer_subgraph.edge(distribute_node, qkv_node)
            qkv_nodes.append(qkv_node)
        
        # Cross-GPU all-reduce for K,V
        kv_reduce_node = f'layer_{layer}_kv_reduce'
        attention_cluster.node(kv_reduce_node,
                             f'KV All-Reduce\\nInput: [batch_size=?, seq_len=2048, heads=4, d_k=128] from 8 GPUs\\nOutput: [batch_size=?, seq_len=2048, heads=32, d_k=128]\\nGPU: All 0-7', 
                             shape='diamond', fillcolor='lightcoral')
        
        for qkv_node in qkv_nodes:
            attention_cluster.edge(qkv_node, kv_reduce_node)
        
        # Attention computation per GPU
        attn_nodes = []
        for gpu_id in range(8):
            attn_node = f'layer_{layer}_attn_gpu_{gpu_id}'
            attention_cluster.node(attn_node,
                                 f'Multi-Head Attention\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=2048, heads=4, d_k=128]\\nOutput: [batch_size=?, seq_len=2048, heads=4, d_k=128]\\nGPU: {gpu_id}', 
                                 shape='box', fillcolor=gpu_colors[gpu_id])
            attention_cluster.edge(kv_reduce_node, attn_node)
            attn_nodes.append(attn_node)
        
        # Attention output aggregation
        attn_agg_node = f'layer_{layer}_attn_agg'
        attention_cluster.node(attn_agg_node,
                             f'Attention Output All-Reduce\\nInput: [batch_size=?, seq_len=2048, heads=4, d_k=128] from 8 GPUs\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: All 0-7', 
                             shape='diamond', fillcolor='lightcoral')
        
        for attn_node in attn_nodes:
            attention_cluster.edge(attn_node, attn_agg_node)
        
        # Residual connection and LayerNorm
        attn_ln_node = f'layer_{layer}_attn_ln'
        attention_cluster.node(attn_ln_node,
                             f'LayerNorm + Residual\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: 0-7', 
                             shape='box', fillcolor='#FFE4B5')
        
        attention_cluster.edge(attn_agg_node, attn_ln_node)
        attention_cluster.edge(distribute_node, attn_ln_node, style='dashed')  # Residual
        
        # === MOE COMPONENT (GPUs 8-15) ===
        moe_cluster = Digraph(name=f'cluster_moe_{layer}')
        moe_cluster.attr(label=f'MoE - Layer {layer+1}', style='dashed', bgcolor='#FFE6E6')
        
        # Gate computation (replicated across MoE GPUs)
        gate_nodes = []
        for gpu_id in range(8, 16):
            gate_node = f'layer_{layer}_gate_gpu_{gpu_id}'
            moe_cluster.node(gate_node,
                           f'Gating Network\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, experts=16]\\nGPU: {gpu_id}', 
                           shape='parallelogram', fillcolor=gpu_colors[gpu_id])
            layer_subgraph.edge(attn_ln_node, gate_node)
            gate_nodes.append(gate_node)
        
        # Expert computation per GPU
        expert_nodes = []
        for gpu_id in range(8, 16):
            for expert_id in range(2):  # 2 experts per GPU
                expert_node = f'layer_{layer}_expert_{(gpu_id-8)*2+expert_id}_gpu_{gpu_id}'
                moe_cluster.node(expert_node,
                               f'Expert {(gpu_id-8)*2+expert_id}\\nGPU {gpu_id}\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, expert_hidden=16384]\\nGPU: {gpu_id}', 
                               shape='box', fillcolor=gpu_colors[gpu_id])
                
                # Routing connection (dashed)
                gate_node = f'layer_{layer}_gate_gpu_{gpu_id}'
                moe_cluster.edge(gate_node, expert_node, style='dashed')
                expert_nodes.append(expert_node)
        
        # Expert aggregation
        expert_agg_node = f'layer_{layer}_expert_agg'
        moe_cluster.node(expert_agg_node,
                       f'Expert Aggregation (Top-2)\\nInput: [batch_size=?, seq_len=2048, expert_hidden=16384] from 16 experts\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15', 
                       shape='parallelogram', fillcolor='yellow')
        
        for expert_node in expert_nodes:
            moe_cluster.edge(expert_node, expert_agg_node)
        
        # Final output for layer
        layer_output_node = f'layer_{layer}_output'
        layer_subgraph.node(layer_output_node,
                           f'Layer {layer+1} Output\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: 8-15', 
                           shape='box', fillcolor='#FFE4B5')
        
        moe_cluster.edge(expert_agg_node, layer_output_node)
        layer_subgraph.edge(attn_ln_node, layer_output_node, style='dashed')  # Residual
        
        # Add clusters to main graph
        dot.subgraph(layer_subgraph)
        dot.subgraph(attention_cluster)
        dot.subgraph(moe_cluster)
    
    # Global output
    dot.node('output', 'Model Output\\nInput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=2048, hidden_dim=4096]\\nGPU: 15', 
             shape='ellipse', fillcolor='lightblue', style='filled,bold')
    
    dot.edge('layer_3_output', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('./outputs/2025-10-16-12-21-24', exist_ok=True)
    
    # Generate MA Separation DAG
    ma_dag = create_ma_separation_dag()
    
    # Save DOT file
    dot_path = './outputs/2025-10-16-12-21-24/ma_separation_model.dot'
    ma_dag.save(dot_path)
    
    # Save SVG image  
    svg_path = './outputs/2025-10-16-12-21-24/ma_separation_model.svg'
    ma_dag.render('./outputs/2025-10-16-12-21-24/ma_separation_model', format='svg', cleanup=True)
    
    print(f"MA Separation DAG generated:")
    print(f"DOT file: {dot_path}")
    print(f"SVG file: {svg_path}")
    
    # Verify DAG structure
    from ExtractInfoFromDAG import extract_dag_info
    info = extract_dag_info(dot_path)
    print(f"DAG verification: {info}")