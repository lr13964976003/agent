#!/usr/bin/env python3
"""
DAG Generator for Final Optimized LLM Deployment Strategy
Strategy: EP1-TP1-PP1-DP8 (8 GPUs total)
"""

import graphviz
from typing import Dict, List, Tuple

def create_llm_deployment_dag():
    """Create comprehensive DAG for EP1-TP1-PP1-DP8 strategy"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP1-TP1-PP1-DP8')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Model parameters
    batch_size = 2048
    seq_len = 512  # average sequence length
    num_layers = 16
    hidden_size = 2048
    num_heads = 16
    head_dim = 64
    experts_per_layer = 64
    vocab_size = 32000
    
    # GPU assignments for DP8 strategy
    gpus = [f'GPU{i}' for i in range(8)]
    
    # Color scheme
    compute_color = 'lightblue'
    comm_color = 'lightgreen'
    routing_color = 'lightyellow'
    input_color = 'lightpink'
    output_color = 'lightgray'
    
    # Input node
    dot.node('input', 
             f'Input\\nBatch: {batch_size} sequences\\nSeq_len: {seq_len}\\nGPU: All',
             shape='ellipse', style='filled', fillcolor=input_color)
    
    # Data parallel split
    dot.node('dp_split', 
             f'Data Parallel Split\\n{int(batch_size/8)} seq/GPU\\nGPU: All',
             shape='parallelogram', style='filled', fillcolor=routing_color)
    
    dot.edge('input', 'dp_split', label=f'Split batch\\n{batch_size}→{int(batch_size/8)} seq/GPU')
    
    # Process each GPU's data
    for gpu_idx, gpu in enumerate(gpus):
        gpu_prefix = f'gpu{gpu_idx}'
        
        # Embedding layer
        dot.node(f'{gpu_prefix}_embed',
                 f'Embedding Layer\\nInput: [{int(batch_size/8)}, {seq_len}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                 shape='rectangle', style='filled', fillcolor=compute_color)
        
        dot.edge('dp_split', f'{gpu_prefix}_embed', 
                label=f'GPU {gpu_idx}\\n{int(batch_size/8)} sequences')
        
        prev_node = f'{gpu_prefix}_embed'
        
        # Process each transformer layer
        for layer in range(num_layers):
            layer_prefix = f'{gpu_prefix}_layer{layer}'
            
            # Layer normalization 1
            dot.node(f'{layer_prefix}_ln1',
                     f'LayerNorm1\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Self-attention
            # QKV projection
            dot.node(f'{layer_prefix}_qkv',
                     f'QKV Projection\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {num_heads*head_dim*3}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Reshape for attention
            dot.node(f'{layer_prefix}_reshape',
                     f'Reshape QKV\\nInput: [{int(batch_size/8)}, {seq_len}, {num_heads*head_dim*3}]\\nOutput: [{int(batch_size/8)}, {num_heads}, {seq_len}, {head_dim}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Attention computation
            dot.node(f'{layer_prefix}_attn',
                     f'Self-Attention\\nInput: [{int(batch_size/8)}, {num_heads}, {seq_len}, {head_dim}]\\nOutput: [{int(batch_size/8)}, {num_heads}, {seq_len}, {head_dim}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Attention output projection
            dot.node(f'{layer_prefix}_attn_out',
                     f'Attention Output Proj\\nInput: [{int(batch_size/8)}, {num_heads}, {seq_len}, {head_dim}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Reshape back
            dot.node(f'{layer_prefix}_attn_reshape',
                     f'Reshape Back\\nInput: [{int(batch_size/8)}, {num_heads}, {seq_len}, {head_dim}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # KV cache update
            dot.node(f'{layer_prefix}_kv_update',
                     f'KV Cache Update\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='ellipse', style='filled', fillcolor=comm_color)
            
            # First residual connection
            dot.node(f'{layer_prefix}_res1',
                     f'Residual Add1\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Layer normalization 2
            dot.node(f'{layer_prefix}_ln2',
                     f'LayerNorm2\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # MoE routing (since EP=1, all experts on same GPU)
            dot.node(f'{layer_prefix}_routing',
                     f'MoE Router\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nExperts: {experts_per_layer}\\nGPU: {gpu}',
                     shape='parallelogram', style='filled', fillcolor=routing_color)
            
            # Expert computation (all experts on same GPU due to EP=1)
            for expert in range(min(4, experts_per_layer)):  # Show first 4 experts
                dot.node(f'{layer_prefix}_expert{expert}',
                         f'Expert {expert}\\nInput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nOutput: [{int(batch_size/8)}, {seq_len}, {hidden_size}]\\nGPU: {gpu}',