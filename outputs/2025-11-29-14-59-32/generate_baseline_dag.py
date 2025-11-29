#!/usr/bin/env python3
"""
Generate DAG for baseline tensor + pipeline parallelism configuration
"""

import graphviz

def create_baseline_tensor_pipeline_dag():
    """
    Create DAG for baseline configuration with TP=8, PP=2
    16 GPUs total, 16 layers, each layer has attention + MLP
    """
    dot = graphviz.Digraph(comment='Baseline Tensor+Pipeline Parallelism DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    
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
    tensor_parallel = 8
    
    # Input node
    dot.node('input', f'Input\\nBatch: {batch_size}\\nSeq: {seq_length}\\nDim: {hidden_dim}', 
             shape='diamond', fillcolor='lightcoral')
    
    # Create nodes for each layer across pipeline stages
    for stage in range(2):  # 2 pipeline stages
        stage_layers = range(8) if stage == 0 else range(8, 16)
        
        for layer_idx in stage_layers:
            # Attention computation nodes (tensor parallel across 8 GPUs)
            with dot.subgraph(name=f'cluster_layer_{layer_idx}_attn') as attn_sg:
                attn_sg.attr(label=f'Layer {layer_idx} Attention (TP=8)', style='rounded,filled', fillcolor='lightgray')
                
                # QKV projection (column parallel)
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    attn_sg.node(f'layer_{layer_idx}_qkv_gpu{gpu_id}', 
                               f'QKV Proj GPU{gpu_id}\\nColParallel\\nDim: {hidden_dim//tensor_parallel}x{3*hidden_dim//tensor_parallel}',
                               shape='rectangle', fillcolor='lightblue')
                
                # QKV communication/all-gather
                attn_sg.node(f'layer_{layer_idx}_qkv_comm', f'QKV AllGather\\n8 GPUs', 
                           shape='ellipse', fillcolor='lightgreen')
                
                # Attention computation (parallel across heads)
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    attn_sg.node(f'layer_{layer_idx}_attn_comp_gpu{gpu_id}', 
                               f'Attention Compute GPU{gpu_id}\\n{num_heads//tensor_parallel} heads\\nSeq: {seq_length}x{head_dim}',
                               shape='rectangle', fillcolor='lightblue')
                
                # Attention output projection (row parallel)
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    attn_sg.node(f'layer_{layer_idx}_attn_out_gpu{gpu_id}', 
                               f'Attn Out Proj GPU{gpu_id}\\nRowParallel\\nDim: {hidden_dim//tensor_parallel}x{hidden_dim//tensor_parallel}',
                               shape='rectangle', fillcolor='lightblue')
                
                # Attention all-reduce
                attn_sg.node(f'layer_{layer_idx}_attn_allreduce', f'Attention AllReduce\\n8 GPUs', 
                           shape='ellipse', fillcolor='lightgreen')
            
            # Residual add after attention
            dot.node(f'layer_{layer_idx}_residual1', f'Residual Add {layer_idx}\\n{hidden_dim} dim',
                     shape='parallelogram', fillcolor='lightyellow')
            
            # Layer normalization
            dot.node(f'layer_{layer_idx}_ln1', f'LayerNorm {layer_idx}\\n{hidden_dim} dim',
                     shape='rectangle', fillcolor='lightblue')
            
            # MLP computation nodes (tensor parallel across 8 GPUs)
            with dot.subgraph(name=f'cluster_layer_{layer_idx}_mlp') as mlp_sg:
                mlp_sg.attr(label=f'Layer {layer_idx} MLP (TP=8)', style='rounded,filled', fillcolor='lightgray')
                
                # First MLP layer (column parallel)
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    mlp_sg.node(f'layer_{layer_idx}_mlp1_gpu{gpu_id}', 
                               f'MLP1 GPU{gpu_id}\\nColParallel\\nDim: {hidden_dim//tensor_parallel}x{mlp_hidden//tensor_parallel}',
                               shape='rectangle', fillcolor='lightblue')
                
                # GELU activation
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    mlp_sg.node(f'layer_{layer_idx}_gelu_gpu{gpu_id}', 
                               f'GELU GPU{gpu_id}\\n{mlp_hidden//tensor_parallel} dim',
                               shape='rectangle', fillcolor='lightblue')
                
                # Second MLP layer (row parallel)
                for tp_rank in range(tensor_parallel):
                    gpu_id = stage * 8 + tp_rank
                    mlp_sg.node(f'layer_{layer_idx}_mlp2_gpu{gpu_id}', 
                               f'MLP2 GPU{gpu_id}\\nRowParallel\\nDim: {mlp_hidden//tensor_parallel}x{hidden_dim//tensor_parallel}',
                               shape='rectangle', fillcolor='lightblue')
                
                # MLP all-reduce
                mlp_sg.node(f'layer_{layer_idx}_mlp_allreduce', f'MLP AllReduce\\n8 GPUs', 
                           shape='ellipse', fillcolor='lightgreen')
            
            # Residual add after MLP
            dot.node(f'layer_{layer_idx}_residual2', f'Residual Add {layer_idx}\\n{hidden_dim} dim',
                     shape='parallelogram', fillcolor='lightyellow')
            
            # Layer normalization
            dot.node(f'layer_{layer_idx}_ln2', f'LayerNorm {layer_idx}\\n{hidden_dim} dim',
                     shape='rectangle', fillcolor='lightblue')
    
    # Output node
    dot.node('output', f'Output\\nBatch: {batch_size}\\nSeq: {seq_length}\\nDim: {hidden_dim}', 
             shape='diamond', fillcolor='lightcoral')
    
    # Connect nodes - Input to first layer
    dot.edge('input', 'layer_0_qkv_gpu0')    
    
    # Connect within layers and between layers
    for layer_idx in range(num_layers):
        stage = 0 if layer_idx < 8 else 1
        
        # Attention connections
        for tp_rank in range(tensor_parallel):
            gpu_id = stage * 8 + tp_rank
            dot.edge(f'layer_{layer_idx}_qkv_gpu{gpu_id}', f'layer_{layer_idx}_qkv_comm')
        
        # Connect QKV comm to attention compute
        for tp_rank in range(tensor_parallel):
            gpu_id = stage * 8 + tp_rank
            dot.edge(f'layer_{layer_idx}_qkv_comm', f'layer_{layer_idx}_attn_comp_gpu{gpu_id}')
        
        # Connect attention compute to output projection
        for tp_rank in range(tensor_parallel):
            gpu_id = stage * 8 + tp_rank
            dot.edge(f'layer_{layer_idx}_attn_comp_gpu{gpu_id}', f'layer_{layer_idx}_attn_out_gpu{gpu_id}')
        
        # Connect attention output to all-reduce
        for tp_rank in range(tensor_parallel):
            gpu_id = stage * 8 + tp_rank
            dot.edge(f'layer_{layer_idx}_attn_out_gpu{gpu_id}', f'layer_{layer_idx}_attn_allreduce')
        
        # Connect attention to residual and layer norm
        dot.edge(f'layer_{layer_idx}_attn_allreduce', f'layer_{layer_idx}_residual1')
        dot.edge(f'layer_{layer_idx}_residual1', f'layer_{layer_idx}_ln1')
        
        # MLP connections
        for tp_rank in range(tensor_parallel):
            gpu_id = stage * 8 + tp_rank
            dot.edge(f'layer_{layer_idx}_ln1', f'layer_{layer_idx}_mlp1_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_mlp1_gpu{gpu_id}', f'layer_{layer_idx}_gelu_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_gelu_gpu{gpu_id}', f'layer_{layer_idx}_mlp2_gpu{gpu_id}')
            dot.edge(f'layer_{layer_idx}_mlp2_gpu{gpu_id}', f'layer_{layer_idx}_mlp_allreduce')
        
        # Connect MLP to residual and layer norm
        dot.edge(f'layer_{layer_idx}_mlp_allreduce', f'layer_{layer_idx}_residual2')
        dot.edge(f'layer_{layer_idx}_residual2', f'layer_{layer_idx}_ln2')
        
        # Connect to next layer or output
        if layer_idx < num_layers - 1:
            next_layer = layer_idx + 1
            next_stage = 0 if next_layer < 8 else 1
            
            # Pipeline communication between stages
            if stage != next_stage:
                # Add pipeline communication node
                dot.node(f'pipeline_comm_{layer_idx}_to_{next_layer}', 
                        f'Pipeline Send/Recv\\nLayer {layer_idx} → {next_layer}',
                        shape='ellipse', fillcolor='orange')
                dot.edge(f'layer_{layer_idx}_ln2', f'pipeline_comm_{layer_idx}_to_{next_layer}')
                dot.edge(f'pipeline_comm_{layer_idx}_to_{next_layer}', f'layer_{next_layer}_qkv_gpu0')
            else:
                dot.edge(f'layer_{layer_idx}_ln2', f'layer_{next_layer}_qkv_gpu0')
        else:
            # Last layer connects to output
            dot.edge(f'layer_{layer_idx}_ln2', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_tensor_pipeline_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-11-29-14-59-32/baseline_tensor_pipeline_dag.dot'
    dag.save(dot_file)
    
    # Save as SVG image
    svg_file = '../outputs/2025-11-29-14-59-32/baseline_tensor_pipeline_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Baseline DAG generated:")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")