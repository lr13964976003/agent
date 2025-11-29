#!/usr/bin/env python3
"""
Generate DAG for proposed layer-wise partitioning configuration
"""

import graphviz

def create_proposed_layer_wise_dag():
    """
    Create DAG for proposed layer-wise configuration with 2 layers per GPU
    8 GPUs used, 16 layers total, sequential execution with cache optimization
    """
    dot = graphviz.Digraph(comment='Proposed Layer-wise Partitioning DAG')
    dot.attr(rankdir='TB', size='25,35', dpi='300')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model specifications from deployment config
    num_layers = 16
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    mlp_hidden = 16384
    batch_size = 128
    seq_length = 10000
    layers_per_gpu = 2
    
    # Input node
    dot.node('input', f'Input\\nBatch: {batch_size}\\nSeq: {seq_length}\\nDim: {hidden_dim}', 
             shape='diamond', fillcolor='lightcoral')
    
    # Create nodes for each GPU with 2 layers each
    for gpu_id in range(8):  # 8 GPUs used in proposed config
        start_layer = gpu_id * layers_per_gpu
        end_layer = start_layer + layers_per_gpu
        
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu_sg:
            gpu_sg.attr(label=f'GPU {gpu_id} (Layers {start_layer}-{end_layer-1})\\nCache: 29.46 GB', 
                       style='rounded,filled', fillcolor='lightsteelblue')
            
            for layer_idx in range(start_layer, end_layer):
                # Attention computation nodes (no tensor parallelism, full layers)
                with gpu_sg.subgraph(name=f'cluster_layer_{layer_idx}_attn_gpu{gpu_id}') as attn_sg:
                    attn_sg.attr(label=f'Layer {layer_idx} Attention (Full)', 
                                style='rounded,filled', fillcolor='lightgray')
                    
                    # QKV projection (full computation on this GPU)
                    attn_sg.node(f'layer_{layer_idx}_qkv_gpu{gpu_id}', 
                               f'QKV Proj GPU{gpu_id}\\nFull Layer\\nDim: {hidden_dim}x{3*hidden_dim}',
                               shape='rectangle', fillcolor='lightblue')
                    
                    # Attention computation (full heads)
                    attn_sg.node(f'layer_{layer_idx}_attn_comp_gpu{gpu_id}', 
                               f'Attention Compute GPU{gpu_id}\\n{num_heads} heads\\nSeq: {seq_length}x{head_dim}',
                               shape='rectangle', fillcolor='lightblue')
                    
                    # Attention output projection
                    attn_sg.node(f'layer_{layer_idx}_attn_out_gpu{gpu_id}', 
                               f'Attn Out Proj GPU{gpu_id}\\nFull Layer\\nDim: {hidden_dim}x{hidden_dim}',
                               shape='rectangle', fillcolor='lightblue')
                
                # Residual add after attention
                gpu_sg.node(f'layer_{layer_idx}_residual1_gpu{gpu_id}', f'Residual Add {layer_idx} GPU{gpu_id}\\n{hidden_dim} dim',
                           shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu_sg.node(f'layer_{layer_idx}_ln1_gpu{gpu_id}', f'LayerNorm {layer_idx} GPU{gpu_id}\\n{hidden_dim} dim',
                           shape='rectangle', fillcolor='lightblue')
                
                # MLP computation nodes (full computation on this GPU)
                with gpu_sg.subgraph(name=f'cluster_layer_{layer_idx}_mlp_gpu{gpu_id}') as mlp_sg:
                    mlp_sg.attr(label=f'Layer {layer_idx} MLP (Full)', 
                               style='rounded,filled', fillcolor='lightgray')
                    
                    # First MLP layer
                    mlp_sg.node(f'layer_{layer_idx}_mlp1_gpu{gpu_id}', 
                               f'MLP1 GPU{gpu_id}\\nFull Layer\\nDim: {hidden_dim}x{mlp_hidden}',
                               shape='rectangle', fillcolor='lightblue')
                    
                    # GELU activation
                    mlp_sg.node(f'layer_{layer_idx}_gelu_gpu{gpu_id}', 
                               f'GELU GPU{gpu_id}\\n{mlp_hidden} dim',
                               shape='rectangle', fillcolor='lightblue')
                    
                    # Second MLP layer
                    mlp_sg.node(f'layer_{layer_idx}_mlp2_gpu{gpu_id}', 
                               f'MLP2 GPU{gpu_id}\\nFull Layer\\nDim: {mlp_hidden}x{hidden_dim}',
                               shape='rectangle', fillcolor='lightblue')
                
                # Residual add after MLP
                gpu_sg.node(f'layer_{layer_idx}_residual2_gpu{gpu_id}', f'Residual Add {layer_idx} GPU{gpu_id}\\n{hidden_dim} dim',
                           shape='parallelogram', fillcolor='lightyellow')
                
                # Layer normalization
                gpu_sg.node(f'layer_{layer_idx}_ln2_gpu{gpu_id}', f'LayerNorm {layer_idx} GPU{gpu_id}\\n{hidden_dim} dim',
                           shape='rectangle', fillcolor='lightblue')
    
    # Output node
    dot.node('output', f'Output\\nBatch: {batch_size}\\nSeq: {seq_length}\\nDim: {hidden_dim}', 
             shape='diamond', fillcolor='lightcoral')
    
    # Connect nodes - Input to first GPU
    dot.edge('input', 'layer_0_qkv_gpu0')
    
    # Connect within GPUs and between GPUs
    for gpu_id in range(8):
        start_layer = gpu_id * layers_per_gpu
        end_layer = start_layer + layers_per_gpu
        
        for layer_idx in range(start_layer, end_layer):
            # Attention connections
            dot.edge(f'layer_{layer_idx}_qkv_gpu{gpu_id}', f'layer_{layer_idx}_attn_comp_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_attn_comp_gpu{gpu_id}', f'layer_{layer_idx}_attn_out_gpu{gpu_id}')
            
            # Connect attention to residual and layer norm
            dot.edge(f'layer_{layer_idx}_attn_out_gpu{gpu_id}', f'layer_{layer_idx}_residual1_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_residual1_gpu{gpu_id}', f'layer_{layer_idx}_ln1_gpu{gpu_id}')
            
            # MLP connections
            dot.edge(f'layer_{layer_idx}_ln1_gpu{gpu_id}', f'layer_{layer_idx}_mlp1_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_mlp1_gpu{gpu_id}', f'layer_{layer_idx}_gelu_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_gelu_gpu{gpu_id}', f'layer_{layer_idx}_mlp2_gpu{gpu_id}')
            
            # Connect MLP to residual and layer norm
            dot.edge(f'layer_{layer_idx}_mlp2_gpu{gpu_id}', f'layer_{layer_idx}_residual2_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_residual2_gpu{gpu_id}', f'layer_{layer_idx}_ln2_gpu{gpu_id}')
            
            # Connect to next layer within same GPU
            if layer_idx < end_layer - 1:
                next_layer = layer_idx + 1
                dot.edge(f'layer_{layer_idx}_ln2_gpu{gpu_id}', f'layer_{next_layer}_qkv_gpu{gpu_id}')
    
    # Connect between GPUs with point-to-point communication
    for gpu_id in range(7):  # 7 connections between 8 GPUs
        last_layer_gpu = (gpu_id + 1) * layers_per_gpu - 1
        first_layer_next_gpu = (gpu_id + 1) * layers_per_gpu
        
        # Add inter-GPU communication node
        dot.node(f'gpu_comm_{gpu_id}_to_{gpu_id+1}', 
                f'GPU-to-GPU Send/Recv\\nGPU {gpu_id} → {gpu_id+1}\\nLayer {last_layer_gpu} → {first_layer_next_gpu}',
                shape='ellipse', fillcolor='orange')
        
        dot.edge(f'layer_{last_layer_gpu}_ln2_gpu{gpu_id}', f'gpu_comm_{gpu_id}_to_{gpu_id+1}')
        dot.edge(f'gpu_comm_{gpu_id}_to_{gpu_id+1}', f'layer_{first_layer_next_gpu}_qkv_gpu{gpu_id+1}')
    
    # Connect last GPU to output
    dot.edge('layer_15_ln2_gpu7', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_layer_wise_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-11-29-14-59-32/proposed_layer_wise_dag.dot'
    dag.save(dot_file)
    
    # Save as SVG image
    svg_file = '../outputs/2025-11-29-14-59-32/proposed_layer_wise_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Proposed DAG generated:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")