#!/usr/bin/env python3
"""
Proposed DAG Generator for Layer-wise Cache-Optimized Deployment
Model: 16-layer dense network with 1 layer per GPU
"""

import os
from graphviz import Digraph

def create_proposed_dag():
    """Create comprehensive DAG for layer-wise cache-optimized deployment"""
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    num_heads = 16
    head_dim = 512
    mlp_hidden_size = 32768
    
    # Create new directed graph
    dag = Digraph('proposed_layer_wise_cache_optimized')
    dag.attr(rankdir='TB', size='200,400')
    dag.attr('node', fontname='Arial', fontsize='10')
    
    # Global styles
    dag.attr('node', shape='rectangle', style='filled')
    dag.attr('edge', fontname='Arial', fontsize='8')
    
    # Create subgraph for each GPU/device
    for gpu_id in range(16):
        layer_num = gpu_id + 1
        with dag.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu_cluster:
            gpu_cluster.attr(label=f'GPU {gpu_id} - Layer {layer_num}\n100% SRAM/L2 Cache', 
                           style='filled', color='lightgreen', fillcolor='lightgreen1')
            
            # Layer input reception
            gpu_cluster.node(f'layer{layer_num}_input',
                           f'Layer {layer_num} Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
            
            # Multi-Head Attention - Query Linear
            gpu_cluster.node(f'layer{layer_num}_mha_query',
                           f'Layer {layer_num} MHA Query Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_q={head_dim}]\nGPU: {gpu_id}',
                           fillcolor='pink')
            
            # Multi-Head Attention - Key Linear
            gpu_cluster.node(f'layer{layer_num}_mha_key',
                           f'Layer {layer_num} MHA Key Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nGPU: {gpu_id}',
                           fillcolor='pink')
            
            # Multi-Head Attention - Value Linear
            gpu_cluster.node(f'layer{layer_num}_mha_value',
                           f'Layer {layer_num} MHA Value Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_v={head_dim}]\nGPU: {gpu_id}',
                           fillcolor='pink')
            
            # Multi-Head Attention - QKT calculation
            gpu_cluster.node(f'layer{layer_num}_mha_qkt',
                           f'Layer {layer_num} MHA QKT Calculation\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
            
            # Multi-Head Attention - Attention Weights (Softmax)
            gpu_cluster.node(f'layer{layer_num}_mha_softmax',
                           f'Layer {layer_num} MHA Softmax\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
            
            # Multi-Head Attention - Apply Attention to V
            gpu_cluster.node(f'layer{layer_num}_mha_apply',
                           f'Layer {layer_num} MHA Apply Attention\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}], [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, d_v={head_dim}]\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, d_v={head_dim}]\nGPU: {gpu_id}',
                           fillcolor='lightcoral')
            
            # Multi-Head Attention - Output Linear
            gpu_cluster.node(f'layer{layer_num}_mha_out',
                           f'Layer {layer_num} MHA Output Linear\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, d_v={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='pink')
            
            # Residual Add 1
            gpu_cluster.node(f'layer{layer_num}_residual1',
                           f'Layer {layer_num} Residual Add 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 1
            gpu_cluster.node(f'layer{layer_num}_layernorm1',
                           f'Layer {layer_num} Layer Norm 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='lightgreen')
            
            # MLP - Gate Linear
            gpu_cluster.node(f'layer{layer_num}_mlp_gate',
                           f'Layer {layer_num} MLP Gate Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='lightyellow')
            
            # MLP - Up Linear
            gpu_cluster.node(f'layer{layer_num}_mlp_up',
                           f'Layer {layer_num} MLP Up Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='lightyellow')
            
            # MLP - GELU Activation
            gpu_cluster.node(f'layer{layer_num}_mlp_gelu',
                           f'Layer {layer_num} MLP GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='orange')
            
            # MLP - Element-wise multiplication (Gate * GELU(Up))
            gpu_cluster.node(f'layer{layer_num}_mlp_multiply',
                           f'Layer {layer_num} MLP Element-wise Multiply\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='orange')
            
            # MLP - Down Linear
            gpu_cluster.node(f'layer{layer_num}_mlp_down',
                           f'Layer {layer_num} MLP Down Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='lightyellow')
            
            # Residual Add 2
            gpu_cluster.node(f'layer{layer_num}_residual2',
                           f'Layer {layer_num} Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 2
            gpu_cluster.node(f'layer{layer_num}_layernorm2',
                           f'Layer {layer_num} Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: {gpu_id}',
                           fillcolor='lightgreen')
            
            # Layer output transfer
            if layer_num < 16:
                dag.node(f'layer{layer_num}_transfer',
                        f'Layer {layer_num} → {layer_num+1} Transfer\nTransfer: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nFrom: GPU {gpu_id}\nTo: GPU {gpu_id+1}',
                        shape='ellipse', fillcolor='gray')

    # Global input
    dag.node('input',
            f'Model Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Host → GPU 0',
            shape='ellipse', fillcolor='lightgreen')
    
    # Global output
    dag.node('output',
            f'Model Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: GPU 15 → Host',
            shape='ellipse', fillcolor='red')

    # Create edges connecting all layers
    dag.edge('input', 'layer1_input')
    
    for layer_num in range(1, 17):
        # MHA connections
        dag.edge(f'layer{layer_num}_input', f'layer{layer_num}_mha_query')
        dag.edge(f'layer{layer_num}_input', f'layer{layer_num}_mha_key')
        dag.edge(f'layer{layer_num}_input', f'layer{layer_num}_mha_value')
        dag.edge(f'layer{layer_num}_mha_query', f'layer{layer_num}_mha_qkt')
        dag.edge(f'layer{layer_num}_mha_key', f'layer{layer_num}_mha_qkt')
        dag.edge(f'layer{layer_num}_mha_qkt', f'layer{layer_num}_mha_softmax')
        dag.edge(f'layer{layer_num}_mha_softmax', f'layer{layer_num}_mha_apply')
        dag.edge(f'layer{layer_num}_mha_value', f'layer{layer_num}_mha_apply')
        dag.edge(f'layer{layer_num}_mha_apply', f'layer{layer_num}_mha_out')
        dag.edge(f'layer{layer_num}_mha_out', f'layer{layer_num}_residual1')
        dag.edge(f'layer{layer_num}_input', f'layer{layer_num}_residual1')
        dag.edge(f'layer{layer_num}_residual1', f'layer{layer_num}_layernorm1')
        
        # MLP connections
        dag.edge(f'layer{layer_num}_layernorm1', f'layer{layer_num}_mlp_gate')
        dag.edge(f'layer{layer_num}_layernorm1', f'layer{layer_num}_mlp_up')
        dag.edge(f'layer{layer_num}_mlp_gate', f'layer{layer_num}_mlp_gelu')
        dag.edge(f'layer{layer_num}_mlp_up', f'layer{layer_num}_mlp_multiply')
        dag.edge(f'layer{layer_num}_mlp_gelu', f'layer{layer_num}_mlp_multiply')
        dag.edge(f'layer{layer_num}_mlp_multiply', f'layer{layer_num}_mlp_down')
        dag.edge(f'layer{layer_num}_mlp_down', f'layer{layer_num}_residual2')
        dag.edge(f'layer{layer_num}_layernorm1', f'layer{layer_num}_residual2')
        dag.edge(f'layer{layer_num}_residual2', f'layer{layer_num}_layernorm2')
        
        # Transfer to next layer
        if layer_num < 16:
            dag.edge(f'layer{layer_num}_layernorm2', f'layer{layer_num}_transfer')
            dag.edge(f'layer{layer_num}_transfer', f'layer{layer_num+1}_input')
        else:
            # Final output
            dag.edge('layer16_layernorm2', 'output')

    return dag

if __name__ == '__main__':
    dag = create_proposed_dag()
    
    # Save files
    output_dir = '../outputs/2025-10-19-20-32-46'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DOT file
    dot_path = os.path.join(output_dir, 'proposed_layer_wise_cache_optimized.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG
    svg_path = os.path.join(output_dir, 'proposed_layer_wise_cache_optimized.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Proposed DAG generated:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")