#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """Create DAG for baseline model (TP=8, PP=2) with no cycles and proper connectivity"""
    
    dot = graphviz.Digraph(comment='MoE Baseline Model DAG (TP=8, PP=2)')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input specifications
    batch_size = 128
    seq_len = 10000
    d_model = 4096
    num_heads = 32
    head_dim = 128
    experts_per_layer = 16
    
    # Create input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0: Layers 0-7 on GPUs 0-7
    for layer in range(8):
        for gpu in range(8):
            gpu_id = gpu
            
            # Layer Norm 1 (before attention)
            ln1_node = f'ln1_l{layer}_gpu{gpu_id}'
            dot.node(ln1_node, 
                     f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightyellow')
            
            # Multi-Head Attention
            attn_node = f'attn_l{layer}_gpu{gpu_id}'
            dot.node(attn_node,
                     f'MHA L{layer} Head{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nGPU: {gpu_id}',
                     fillcolor='lightcoral')
            
            # Residual Add 1 (after attention)
            residual1_node = f'residual1_l{layer}_gpu{gpu_id}'
            dot.node(residual1_node,
                     f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightpink')
            
            # Layer Norm 2 (before MoE)
            ln2_node = f'ln2_l{layer}_gpu{gpu_id}'
            dot.node(ln2_node,
                     f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightyellow')
            
            # MoE Experts (all 16 experts shared on each GPU)
            for expert_id in range(experts_per_layer):
                expert_node = f'expert_l{layer}_e{expert_id}_gpu{gpu_id}'
                dot.node(expert_node,
                         f'Expert L{layer}E{expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                         fillcolor='lightsalmon')
            
            # Expert Aggregation
            agg_node = f'agg_l{layer}_gpu{gpu_id}'
            dot.node(agg_node,
                     f'ExpertAgg L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     shape='parallelogram', fillcolor='lightgray')
            
            # Residual Add 2 (after MoE)
            residual2_node = f'residual2_l{layer}_gpu{gpu_id}'
            dot.node(residual2_node,
                     f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightpink')
    
    # Stage 1: Layers 8-15 on GPUs 8-15
    for layer in range(8, 16):
        for gpu in range(8):
            gpu_id = gpu + 8
            
            # Layer Norm 1 (before attention)
            ln1_node = f'ln1_l{layer}_gpu{gpu_id}'
            dot.node(ln1_node, 
                     f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightyellow')
            
            # Multi-Head Attention
            attn_node = f'attn_l{layer}_gpu{gpu_id}'
            dot.node(attn_node,
                     f'MHA L{layer} Head{gpu_id-8}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nGPU: {gpu_id}',
                     fillcolor='lightcoral')
            
            # Residual Add 1 (after attention)
            residual1_node = f'residual1_l{layer}_gpu{gpu_id}'
            dot.node(residual1_node,
                     f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightpink')
            
            # Layer Norm 2 (before MoE)
            ln2_node = f'ln2_l{layer}_gpu{gpu_id}'
            dot.node(ln2_node,
                     f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightyellow')
            
            # MoE Experts (all 16 experts shared on each GPU)
            for expert_id in range(experts_per_layer):
                expert_node = f'expert_l{layer}_e{expert_id}_gpu{gpu_id}'
                dot.node(expert_node,
                         f'Expert L{layer}E{expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                         fillcolor='lightsalmon')
            
            # Expert Aggregation
            agg_node = f'agg_l{layer}_gpu{gpu_id}'
            dot.node(agg_node,
                     f'ExpertAgg L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     shape='parallelogram', fillcolor='lightgray')
            
            # Residual Add 2 (after MoE)
            residual2_node = f'residual2_l{layer}_gpu{gpu_id}'
            dot.node(residual2_node,
                     f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model//8}]\\nGPU: {gpu_id}',
                     fillcolor='lightpink')
    
    # Create output node
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect nodes properly - NO CYCLES
    # Input to first layer
    for gpu in range(8):
        dot.edge('input', f'ln1_l0_gpu{gpu}')
    
    # Connect within each layer
    for layer in range(16):
        for gpu in range(8):
            if layer < 8:
                gpu_id = gpu
            else:
                gpu_id = gpu + 8
            
            # Forward flow within layer
            dot.edge(f'ln1_l{layer}_gpu{gpu_id}', f'attn_l{layer}_gpu{gpu_id}')
            dot.edge(f'attn_l{layer}_gpu{gpu_id}', f'residual1_l{layer}_gpu{gpu_id}')
            dot.edge(f'residual1_l{layer}_gpu{gpu_id}', f'ln2_l{layer}_gpu{gpu_id}')
            
            # Connect experts
            for expert_id in range(experts_per_layer):
                dot.edge(f'ln2_l{layer}_gpu{gpu_id}', f'expert_l{layer}_e{expert_id}_gpu{gpu_id}')
                dot.edge(f'expert_l{layer}_e{expert_id}_gpu{gpu_id}', f'agg_l{layer}_gpu{gpu_id}')
            
            dot.edge(f'agg_l{layer}_gpu{gpu_id}', f'residual2_l{layer}_gpu{gpu_id}')
            
            # Connect to next layer (if not last layer)
            if layer < 15:
                for next_gpu in range(8):
                    if layer < 7:
                        next_gpu_id = next_gpu
                    else:
                        next_gpu_id = next_gpu + 8
                    dot.edge(f'residual2_l{layer}_gpu{gpu_id}', f'ln1_l{layer+1}_gpu{next_gpu_id}')
    
    # Connect last layer to output
    for gpu in range(8):
        dot.edge(f'residual2_l15_gpu{gpu+8}', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    
    # Save DOT file
    dot_file_path = "../outputs/2025-11-25-14-29-13/baseline_model_dag_fixed.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG file
    svg_file_path = "../outputs/2025-11-25-14-29-13/baseline_model_dag_fixed.svg"
    dag.render(dot_file_path.replace('.dot', ''), format='svg', cleanup=True)
    
    print(f"Baseline DAG generated:")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")