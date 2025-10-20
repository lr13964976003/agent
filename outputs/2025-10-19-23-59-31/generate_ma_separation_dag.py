#!/usr/bin/env python3
"""
Generate complete DAG for MA Separation parallel strategy
16 GPUs total: 12 for attention, 4 for MoE
4-layer MoE transformer with 16 experts per layer
"""

import graphviz
from typing import Dict, List

def create_ma_separation_dag():
    """Create complete DAG for MA Separation strategy"""
    
    # Create directed graph
    dot = graphviz.Digraph('ma_separation_dag', comment='MA Separation Complete Model DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse')  # Input/Output
    dot.attr('node', style='filled')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 2048
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    vocab_size = 50265
    expert_hidden = 16384
    
    # GPU assignments
    attention_gpus = list(range(12))  # GPUs 0-11
    moe_gpus = list(range(12, 16))   # GPUs 12-15
    
    # Colors for GPU identification
    gpu_colors = {}
    for gpu in range(16):
        if gpu < 12:
            # Attention GPUs - shades of blue
            gpu_colors[gpu] = f'#{(gpu*15):02x}{128+(gpu*10):02x}ff'
        else:
            # MoE GPUs - shades of green
            gpu_idx = gpu - 12
            gpu_colors[gpu] = f'#ff{(gpu_idx*50):02x}{gpu_idx*50:02x}'
    
    # Model Input
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
             shape='ellipse', fillcolor='#e6f3ff')
    
    # Embedding layer (on GPU 0)
    dot.node('embedding', 
             f'Embedding\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
             shape='rectangle', fillcolor=gpu_colors[0])
    
    # Create 4 layers
    for layer_idx in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer_idx}') as layer:
            layer.attr(label=f'Layer {layer_idx}', style='dashed', color='gray')
            
            # === ATTENTION BLOCK (12 GPUs) ===
            
            # Layer Normalization (on GPU 0)
            ln1_name = f'ln1_layer_{layer_idx}'
            layer.node(ln1_name,
                      f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
                      shape='rectangle', fillcolor=gpu_colors[0])
            
            # Query-Key-Value Projections across 12 GPUs
            qkv_nodes = []
            for gpu_idx in range(12):
                heads_per_gpu = num_heads // 12  # 2.67 heads, but we'll use 3 for some, 2 for others
                actual_heads = 3 if gpu_idx < 8 else 2  # 8*3 + 4*2 = 32
                qkv_name = f'qkv_layer_{layer_idx}_gpu_{gpu_idx}'
                qkv_nodes.append(qkv_name)
                layer.node(qkv_name,
                          f'QKV Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={actual_heads}, d_k={head_dim}]\\nK:[batch_size={batch_size}, seq_len={seq_len}, heads={actual_heads}, d_k={head_dim}]\\nV:[batch_size={batch_size}, seq_len={seq_len}, heads={actual_heads}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                          shape='rectangle', fillcolor=gpu_colors[gpu_idx])
            
            # Attention computation nodes for each GPU
            attention_nodes = []
            for gpu_idx in range(12):
                actual_heads = 3 if gpu_idx < 8 else 2
                attn_name = f'attention_layer_{layer_idx}_gpu_{gpu_idx}'
                attention_nodes.append(attn_name)
                layer.node(attn_name,
                          f'Multi-Head Attention\\nInput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={actual_heads}, d_k={head_dim}]\\nK_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nV_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={actual_heads}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                          shape='rectangle', fillcolor=gpu_colors[gpu_idx])
            
            # All-reduce for attention outputs
            attn_agg_name = f'attention_agg_layer_{layer_idx}'
            layer.node(attn_agg_name,
                      f'Attention All-Reduce\\nInput: [12× batch_size={batch_size}, seq_len={seq_len}, heads=2-3, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all attention GPUs',
                      shape='parallelogram', fillcolor='#ffcccc')
            
            # Output projection
            out_proj_name = f'out_proj_layer_{layer_idx}'
            layer.node(out_proj_name,
                      f'Output Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
                      shape='rectangle', fillcolor=gpu_colors[0])
            
            # Residual connection 1
            residual1_name = f'residual1_layer_{layer_idx}'
            layer.node(residual1_name,
                      f'Residual Add 1\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
                      shape='parallelogram', fillcolor='#ffffcc')
            
            # === MOE BLOCK (4 GPUs) ===
            
            # Layer Normalization 2 (on GPU 0)
            ln2_name = f'ln2_layer_{layer_idx}'
            layer.node(ln2_name,
                      f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
                      shape='rectangle', fillcolor=gpu_colors[0])
            
            # Gate network (replicated on all MoE GPUs)
            gate_name = f'gate_layer_{layer_idx}'
            layer.node(gate_name,
                      f'Gating Network\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={16}]\\nGPU: all MoE GPUs (12-15)',
                      shape='parallelogram', fillcolor='#ccffcc', style='dashed')
            
            # Expert computation across 4 GPUs (4 experts per GPU)
            expert_nodes = []
            for expert_gpu_idx in range(4):
                gpu_num = 12 + expert_gpu_idx
                for expert_local_idx in range(4):
                    expert_id = expert_gpu_idx * 4 + expert_local_idx
                    expert_name = f'expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_num}'
                    expert_nodes.append(expert_name)
                    layer.node(expert_name,
                              f'Expert{expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {gpu_num}',
                              shape='rectangle', fillcolor=gpu_colors[gpu_num])
            
            # Expert aggregation
            expert_agg_name = f'expert_agg_layer_{layer_idx}'
            layer.node(expert_agg_name,
                      f'Expert Aggregation\\nInput: [16× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all MoE GPUs',
                      shape='parallelogram', fillcolor='#ffcccc')
            
            # Residual connection 2
            residual2_name = f'residual2_layer_{layer_idx}'
            layer.node(residual2_name,
                      f'Residual Add 2\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
                      shape='parallelogram', fillcolor='#ffffcc')
    
    # Final LayerNorm and Output
    dot.node('final_layernorm',
             f'Final LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
             shape='rectangle', fillcolor=gpu_colors[0])
    
    dot.node('output',
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: 0',
             shape='ellipse', fillcolor='#e6f3ff')
    
    # === CONNECTIONS ===
    
    # Input to embedding
    dot.edge('input', 'embedding')
    
    prev_node = 'embedding'
    
    # Connect all layers
    for layer_idx in range(4):
        ln1 = f'ln1_layer_{layer_idx}'
        
        # LayerNorm -> QKV projections
        dot.edge(prev_node, ln1)
        
        for gpu_idx in range(12):
            qkv_name = f'qkv_layer_{layer_idx}_gpu_{gpu_idx}'
            attn_name = f'attention_layer_{layer_idx}_gpu_{gpu_idx}'
            
            # Create edges for cross-GPU communication
            dot.edge(ln1, qkv_name)
            dot.edge(qkv_name, attn_name)
            
            # Attention computation needs all K,V from all GPUs
            for other_gpu in range(12):
                if other_gpu != gpu_idx:
                    other_qkv = f'qkv_layer_{layer_idx}_gpu_{other_gpu}'
                    # K,V communication (dashed)
                    dot.edge(other_qkv, attn_name, style='dashed', color='blue')
        
        # Attention all-reduce
        attn_agg = f'attention_agg_layer_{layer_idx}'
        for gpu_idx in range(12):
            attn_name = f'attention_layer_{layer_idx}_gpu_{gpu_idx}'
            dot.edge(attn_name, attn_agg)
        
        # Output projection and residual
        out_proj = f'out_proj_layer_{layer_idx}'
        dot.edge(attn_agg, out_proj)
        
        residual1 = f'residual1_layer_{layer_idx}'
        dot.edge(out_proj, residual1)
        dot.edge(prev_node, residual1)  # Skip connection
        
        # MoE block
        ln2 = f'ln2_layer_{layer_idx}'
        dot.edge(residual1, ln2)
        
        gate = f'gate_layer_{layer_idx}'
        dot.edge(ln2, gate, style='dashed')
        
        # Expert routing and computation
        for expert_gpu_idx in range(4):
            gpu_num = 12 + expert_gpu_idx
            for expert_local_idx in range(4):
                expert_id = expert_gpu_idx * 4 + expert_local_idx
                expert_name = f'expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_num}'
                
                # Data routing (dashed gate to expert)
                dot.edge(gate, expert_name, style='dashed', color='red')
                # Data flow
                dot.edge(ln2, expert_name)
        
        # Expert aggregation
        expert_agg = f'expert_agg_layer_{layer_idx}'
        for expert_gpu_idx in range(4):
            gpu_num = 12 + expert_gpu_idx
            for expert_local_idx in range(4):
                expert_id = expert_gpu_idx * 4 + expert_local_idx
                expert_name = f'expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_num}'
                dot.edge(expert_name, expert_agg)
        
        # Residual connection 2
        residual2 = f'residual2_layer_{layer_idx}'
        dot.edge(expert_agg, residual2)
        dot.edge(residual1, residual2)  # Skip connection
        
        prev_node = residual2
    
    # Final connections
    dot.edge(prev_node, 'final_layernorm')
    dot.edge('final_layernorm', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_ma_separation_dag()
    
    # Save DOT file
    dag.save('../outputs/2025-10-19-23-59-31/ma_separation_dag.dot')
    
    # Generate SVG
    dag.render('../outputs/2025-10-19-23-59-31/ma_separation_dag', format='svg', cleanup=True)
    
    print("MA Separation DAG generated successfully!")