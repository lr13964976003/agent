#!/usr/bin/env python3
"""
Generate LLaMA-7B DAG with HPipe token-dimension pipeline parallelism
"""

import graphviz
from typing import Dict, List, Tuple
import os

def create_llama_hpipe_dag():
    """Create LLaMA-7B HPipe DAG"""
    dot = graphviz.Digraph(comment='LLaMA-7B HPipe Pipeline Parallelism', format='svg')
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Model specifications
    batch_size = 1
    seq_len = 2048
    hidden_size = 4096
    ffn_hidden_size = 11008
    num_heads = 32
    head_dim = hidden_size // num_heads  # 128
    vocab_size = 32000
    
    # Device assignments from deployment_config
    device_mapping = {
        'P@1': {'layers': [1, 2, 3], 'type': 'P100', 'host': 'host1'},
        'P@2': {'layers': [4, 5, 6, 7], 'type': 'P100', 'host': 'host1'},
        'P@3': {'layers': [8, 9, 10, 11, 12], 'type': 'P100', 'host': 'host1'},
        'P@4': {'layers': [13, 14, 15, 16, 17, 18], 'type': 'P100', 'host': 'host1'},
        'R@1': {'layers': [19, 20, 21, 22, 23, 24, 25], 'type': 'RTX3090', 'host': 'host2'},
        'R@2': {'layers': [26, 27, 28, 29, 30, 31, 32], 'type': 'RTX3090', 'host': 'host2'}
    }
    
    # Sequence slicing scheme
    slicing_scheme = [384, 360, 336, 312, 288, 264, 240, 216, 192, 160]
    
    # Define node shapes based on type
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Communication
    
    # Global input
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}]\\nGPU: host1', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Embedding layer (on P@1)
    dot.node('embedding', f'Embedding\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: P@1', 
             shape='rectangle', fillcolor='lightblue')
    dot.edge('input', 'embedding')
    
    # Create pipeline stages
    prev_node = 'embedding'
    
    for stage_idx, (device, config) in enumerate(device_mapping.items()):
        with dot.subgraph(name=f'cluster_{device}') as stage:
            stage.attr(label=f'Stage {stage_idx+1}\\n{device} ({config["type"]})', style='dashed')
            
            for layer_idx in config['layers']:
                layer_name = f'layer_{layer_idx}'
                
                # Create nodes for multi-head attention
                attn_qkv = f'{layer_name}_attn_qkv'
                attn_score = f'{layer_name}_attn_score'
                attn_softmax = f'{layer_name}_attn_softmax'
                attn_dropout = f'{layer_name}_attn_dropout'
                attn_out = f'{layer_name}_attn_out'
                attn_residual = f'{layer_name}_attn_residual'
                attn_norm = f'{layer_name}_attn_norm'
                
                # Create nodes for FFN
                ffn_up = f'{layer_name}_ffn_up'
                ffn_act = f'{layer_name}_ffn_act'
                ffn_down = f'{layer_name}_ffn_down'
                ffn_residual = f'{layer_name}_ffn_residual'
                ffn_norm = f'{layer_name}_ffn_norm'
                
                # Attention computation
                stage.node(attn_qkv, f'Layer {layer_idx}\\nAttention QKV\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: {device}')
                stage.node(attn_score, f'Layer {layer_idx}\\nAttention Score\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: {device}')
                stage.node(attn_softmax, f'Layer {layer_idx}\\nAttention Softmax\\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: {device}')
                stage.node(attn_dropout, f'Layer {layer_idx}\\nAttention Dropout\\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: {device}')
                stage.node(attn_out, f'Layer {layer_idx}\\nAttention Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                stage.node(attn_residual, f'Layer {layer_idx}\\nAttention Residual\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                stage.node(attn_norm, f'Layer {layer_idx}\\nAttention Norm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                
                # FFN computation
                stage.node(ffn_up, f'Layer {layer_idx}\\nFFN Up\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nGPU: {device}')
                stage.node(ffn_act, f'Layer {layer_idx}\\nFFN Activation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nGPU: {device}')
                stage.node(ffn_down, f'Layer {layer_idx}\\nFFN Down\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                stage.node(ffn_residual, f'Layer {layer_idx}\\nFFN Residual\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                stage.node(ffn_norm, f'Layer {layer_idx}\\nFFN Norm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                
                if layer_idx == 1:
                    dot.edge(prev_node, attn_qkv)
                else:
                    # Add communication node between stages
                    comm_node = f'comm_stage_{stage_idx-1}_{layer_idx}'
                    if stage_idx > 0:
                        prev_device = list(device_mapping.keys())[stage_idx-1]
                        dot.node(comm_node, f'Token Transfer\\nInput: [batch_size={batch_size}, seq_len=slice, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len=slice, hidden_size={hidden_size}]\\nFrom: {prev_device} To: {device}\\nType: inter-stage communication', 
                                 shape='parallelogram', fillcolor='lightyellow')
                        dot.edge(f'layer_{layer_idx-1}_ffn_norm', comm_node)
                        dot.edge(comm_node, attn_qkv)
                    else:
                        dot.edge(f'layer_{layer_idx-1}_ffn_norm', attn_qkv)
                
                # Connect attention sub-operations
                stage.edge(attn_qkv, attn_score)
                stage.edge(attn_score, attn_softmax)
                stage.edge(attn_softmax, attn_dropout)
                stage.edge(attn_dropout, attn_out)
                stage.edge(attn_out, attn_residual)
                stage.edge(attn_residual, attn_norm)
                
                # Connect FFN sub-operations
                stage.edge(attn_norm, ffn_up)
                stage.edge(ffn_up, ffn_act)
                stage.edge(ffn_act, ffn_down)
                stage.edge(ffn_down, ffn_residual)
                stage.edge(ffn_residual, ffn_norm)
                
                prev_node = ffn_norm
    
    # Final output layer
    dot.node('lm_head', f'LM Head\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: R@2', 
             shape='rectangle', fillcolor='lightblue')
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: host2', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect final layer
    final_comm = 'comm_final'
    dot.node(final_comm, f'Token Transfer\\nInput: [batch_size={batch_size}, seq_len=slice, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len=slice, hidden_size={hidden_size}]\\nFrom: R@2 To: host2\\nType: final aggregation', 
             shape='parallelogram', fillcolor='lightyellow')
    
    dot.edge('layer_32_ffn_norm', final_comm)
    dot.edge(final_comm, 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_llama_hpipe_dag()
    dag.save('../outputs/2025-10-31-11-22-09/llama_7b_hpipe')
    print("LLaMA-7B HPipe DAG saved")