#!/usr/bin/env python3

import graphviz
import os

def create_moe_dag():
    """Create a comprehensive MoE model deployment DAG"""
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='100,100')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow') # Routing/Aggregation
    
    # Model configuration
    layers = 16
    total_gpus = 128
    experts_per_layer = 64
    experts_per_gpu = 8
    token_dim = 1024
    moe_hidden = 2048
    batch_size = 128
    seq_length = 10240
    
    # Global input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Global Input', style='rounded,filled', fillcolor='lightgray')
        c.node('input', 
               f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
               shape='rectangle', fillcolor='lightgray')
    
    # Process each layer
    for layer_idx in range(layers):
        layer_name = f'layer_{layer_idx}'
        
        with dot.subgraph(name=f'cluster_{layer_name}') as c:
            c.attr(label=f'Layer {layer_idx}', style='rounded,filled', fillcolor='lightcyan')
            
            # LayerNorm (before attention)
            c.node(f'{layer_name}_ln1', 
                   f'LayerNorm\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightblue')
            
            # Multi-Head Attention
            c.node(f'{layer_name}_attn_qkv', 
                   f'Attention QKV Linear\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, heads=32, d_k=64]',
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'{layer_name}_attn_score', 
                   f'Attention Scores\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, heads=32, d_k=64]\\nOutput: [batch_size={batch_size}, heads=32, seq_len={seq_length}, seq_len={seq_length}]',
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'{layer_name}_attn_softmax', 
                   f'Attention Softmax\\nGPU: 0-127\\nInput: [batch_size={batch_size}, heads=32, seq_len={seq_length}, seq_len={seq_length}]\\nOutput: [batch_size={batch_size}, heads=32, seq_len={seq_length}, seq_len={seq_length}]',
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'{layer_name}_attn_weight', 
                   f'Attention Weights\\nGPU: 0-127\\nInput: [batch_size={batch_size}, heads=32, seq_len={seq_length}, seq_len={seq_length}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, heads=32, d_k=64]',
                   shape='rectangle', fillcolor='lightblue')
            
            c.node(f'{layer_name}_attn_out', 
                   f'Attention Output Linear\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, heads=32, d_k=64]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightblue')
            
            # Attention residual add
            c.node(f'{layer_name}_attn_residual', 
                   f'Attention Residual Add\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='yellow')
            
            # LayerNorm (before MoE)
            c.node(f'{layer_name}_ln2', 
                   f'LayerNorm\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightblue')
            
            # MoE Gate
            c.node(f'{layer_name}_gate', 
                   f'MoE Gate\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, experts={experts_per_layer}]',
                   shape='parallelogram', fillcolor='yellow')
            
            # Expert routing (dashed line for gate selection)
            c.node(f'{layer_name}_routing', 
                   f'Expert Routing\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='ellipse', fillcolor='lightgreen', style='dashed,filled')
            
            # Process experts per GPU
            for gpu_idx in range(total_gpus):
                gpu_experts = experts_per_gpu
                expert_start = gpu_idx * gpu_experts
                expert_end = (gpu_idx + 1) * gpu_experts
                
                with dot.subgraph(name=f'cluster_{layer_name}_gpu_{gpu_idx}') as gpu_c:
                    gpu_c.attr(label=f'GPU {gpu_idx} (Experts {expert_start}-{expert_end-1})', 
                              style='rounded,filled', fillcolor='lightyellow')
                    
                    # Expert processing
                    for expert_idx in range(gpu_experts):
                        global_expert_id = expert_start + expert_idx
                        
                        # Expert linear 1
                        gpu_c.node(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear1',
                                  f'Expert {global_expert_id} Linear 1\\nGPU: {gpu_idx}\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, hidden={moe_hidden}]',
                                  shape='rectangle', fillcolor='lightblue')
                        
                        # Expert activation
                        gpu_c.node(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_activation',
                                  f'Expert {global_expert_id} GELU\\nGPU: {gpu_idx}\\nInput: [batch_size={batch_size}, seq_len={seq_length}, hidden={moe_hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, hidden={moe_hidden}]',
                                  shape='rectangle', fillcolor='lightblue')
                        
                        # Expert linear 2
                        gpu_c.node(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear2',
                                  f'Expert {global_expert_id} Linear 2\\nGPU: {gpu_idx}\\nInput: [batch_size={batch_size}, seq_len={seq_length}, hidden={moe_hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                                  shape='rectangle', fillcolor='lightblue')
            
            # Expert aggregation
            c.node(f'{layer_name}_expert_agg', 
                   f'Expert Aggregation\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='yellow')
            
            # MoE residual add
            c.node(f'{layer_name}_moe_residual', 
                   f'MoE Residual Add\\nGPU: 0-127\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='yellow')
    
    # Global output node
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Global Output', style='rounded,filled', fillcolor='lightgray')
        c.node('output', 
               f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_length}, token_dim={token_dim}]',
               shape='rectangle', fillcolor='lightgray')
    
    # Connect all nodes
    # Input to first layer
    dot.edge('input', 'layer_0_ln1')
    
    for layer_idx in range(layers):
        layer_name = f'layer_{layer_idx}'
        
        # Attention path
        dot.edge(f'{layer_name}_ln1', f'{layer_name}_attn_qkv')
        dot.edge(f'{layer_name}_attn_qkv', f'{layer_name}_attn_score')
        dot.edge(f'{layer_name}_attn_score', f'{layer_name}_attn_softmax')
        dot.edge(f'{layer_name}_attn_softmax', f'{layer_name}_attn_weight')
        dot.edge(f'{layer_name}_attn_weight', f'{layer_name}_attn_out')
        
        # Attention residual
        if layer_idx == 0:
            dot.edge('input', f'{layer_name}_attn_residual')
        else:
            dot.edge(f'layer_{layer_idx-1}_moe_residual', f'{layer_name}_attn_residual')
        dot.edge(f'{layer_name}_attn_out', f'{layer_name}_attn_residual')
        
        # MoE path
        dot.edge(f'{layer_name}_attn_residual', f'{layer_name}_ln2')
        dot.edge(f'{layer_name}_ln2', f'{layer_name}_gate')
        dot.edge(f'{layer_name}_gate', f'{layer_name}_routing')
        
        # Connect routing to all experts
        for gpu_idx in range(total_gpus):
            for expert_idx in range(experts_per_gpu):
                dot.edge(f'{layer_name}_routing', 
                        f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear1')
                dot.edge(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear1',
                        f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_activation')
                dot.edge(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_activation',
                        f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear2')
                dot.edge(f'{layer_name}_gpu_{gpu_idx}_expert_{expert_idx}_linear2',
                        f'{layer_name}_expert_agg')
        
        # MoE residual
        dot.edge(f'{layer_name}_attn_residual', f'{layer_name}_moe_residual')
        dot.edge(f'{layer_name}_expert_agg', f'{layer_name}_moe_residual')
        
        # Connect to next layer or output
        if layer_idx < layers - 1:
            dot.edge(f'{layer_name}_moe_residual', f'layer_{layer_idx+1}_ln1')
        else:
            dot.edge(f'{layer_name}_moe_residual', 'output')
    
    return dot

if __name__ == "__main__":
    # Create the DAG
    dag = create_moe_dag()
    
    # Save DOT file
    dot_file = '../outputs/2025-12-01-14-50-03/moe_deployment_dag.dot'
    dag.save(dot_file)
    
    # Render to SVG
    svg_file = '../outputs/2025-12-01-14-50-03/moe_deployment_dag.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    # Verify the DAG
    try:
        import subprocess
        result = subprocess.run(['dot', '-Tsvg', dot_file, '-o', svg_file], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("SVG rendering successful!")
        else:
            print(f"SVG rendering failed: {result.stderr}")
    except Exception as e:
        print(f"SVG rendering error: {e}")