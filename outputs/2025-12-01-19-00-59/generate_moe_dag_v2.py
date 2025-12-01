#!/usr/bin/env python3
"""
Generate a complete MoE (Mixture of Experts) DAG for single GPU deployment
EP1_TP1 configuration with detailed operator-level breakdown
"""

import graphviz
from graphviz import Digraph

def create_moe_dag():
    """Create a complete DAG for MoE model with single GPU deployment"""
    
    # Create directed graph
    dot = Digraph(comment='MoE Single GPU Deployment DAG')
    dot.attr(rankdir='TB', splines='ortho', concentrate='true')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow') # Routing/Aggregation
    
    # Model parameters from deployment plan
    batch_size = 8
    seq_len = 256
    token_dim = 1024
    attention_heads = 8
    moe_hidden_dim = 4096
    num_layers = 8
    num_experts = 4
    
    # Create input node
    dot.node('input', f'Input\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='rectangle', fillcolor='lightblue')
    
    prev_node = 'input'
    
    # Create layers
    for layer_idx in range(num_layers):
        layer_prefix = f'layer{layer_idx}'
        
        # Layer Normalization (Attention)
        ln_attn = f'{layer_prefix}_ln_attn'
        dot.node(ln_attn, f'LayerNorm (Attention)\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(prev_node, ln_attn)
        
        # Multi-Head Attention - QKV projections
        q_proj = f'{layer_prefix}_q_proj'
        k_proj = f'{layer_prefix}_k_proj'
        v_proj = f'{layer_prefix}_v_proj'
        
        d_k = token_dim // attention_heads
        
        dot.node(q_proj, f'Q Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]',
                shape='rectangle', fillcolor='lightblue')
        dot.node(k_proj, f'K Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]',
                shape='rectangle', fillcolor='lightblue')
        dot.node(v_proj, f'V Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]',
                shape='rectangle', fillcolor='lightblue')
        
        dot.edge(ln_attn, q_proj)
        dot.edge(ln_attn, k_proj)
        dot.edge(ln_attn, v_proj)
        
        # Attention Score Computation
        attn_scores = f'{layer_prefix}_attn_scores'
        dot.node(attn_scores, f'Attention Scores\\nGPU: 0\\nInput: Q=[batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}], K=[batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, heads={attention_heads}, seq_len={seq_len}, seq_len={seq_len}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(q_proj, attn_scores)
        dot.edge(k_proj, attn_scores)
        
        # Attention Weights (Softmax)
        attn_weights = f'{layer_prefix}_attn_weights'
        dot.node(attn_weights, f'Softmax (Attention)\\nGPU: 0\\nInput: [batch_size={batch_size}, heads={attention_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={attention_heads}, seq_len={seq_len}, seq_len={seq_len}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(attn_scores, attn_weights)
        
        # Attention Output
        attn_out = f'{layer_prefix}_attn_out'
        dot.node(attn_out, f'Attention Output\\nGPU: 0\\nInput: Weights=[batch_size={batch_size}, heads={attention_heads}, seq_len={seq_len}, seq_len={seq_len}], V=[batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(attn_weights, attn_out)
        dot.edge(v_proj, attn_out)
        
        # Attention Output Projection
        attn_proj = f'{layer_prefix}_attn_proj'
        dot.node(attn_proj, f'Attention Output Projection\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={attention_heads}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(attn_out, attn_proj)
        
        # Residual Addition (Attention)
        residual_attn = f'{layer_prefix}_residual_attn'
        dot.node(residual_attn, f'Residual Add (Attention)\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}], [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='parallelogram', fillcolor='yellow')
        dot.edge(ln_attn, residual_attn)
        dot.edge(attn_proj, residual_attn)
        
        # Layer Normalization (MoE)
        ln_moe = f'{layer_prefix}_ln_moe'
        dot.node(ln_moe, f'LayerNorm (MoE)\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='rectangle', fillcolor='lightblue')
        dot.edge(residual_attn, ln_moe)
        
        # Gate Network
        gate = f'{layer_prefix}_gate'
        dot.node(gate, f'Gate Network\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts={num_experts}]',
                shape='parallelogram', fillcolor='yellow')
        dot.edge(ln_moe, gate)
        
        # Expert routing (dashed line for gate selection)
        expert_nodes = []
        for expert_idx in range(num_experts):
            expert = f'{layer_prefix}_expert{expert_idx}'
            dot.node(expert, f'Expert {expert_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                    shape='rectangle', fillcolor='lightblue')
            
            # Dashed line for gate selection
            dot.edge(gate, expert, style='dashed', label=f'select expert {expert_idx}')
            dot.edge(ln_moe, expert)
            expert_nodes.append(expert)
        
        # Expert aggregation
        expert_agg = f'{layer_prefix}_expert_agg'
        dot.node(expert_agg, f'Expert Aggregation\\nGPU: 0\\nInput: Expert0=[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}], Expert1=[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}], Expert2=[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}], Expert3=[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='parallelogram', fillcolor='yellow')
        
        for expert in expert_nodes:
            dot.edge(expert, expert_agg)
        
        # Residual Addition (MoE)
        residual_moe = f'{layer_prefix}_residual_moe'
        dot.node(residual_moe, f'Residual Add (MoE)\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}], [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                shape='parallelogram', fillcolor='yellow')
        dot.edge(residual_attn, residual_moe)
        dot.edge(expert_agg, residual_moe)
        
        prev_node = residual_moe
    
    # Output node
    dot.node('output', f'Output\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightblue')
    dot.edge(prev_node, 'output')
    
    return dot

def main():
    """Generate and save the MoE DAG"""
    # Create the DAG
    dag = create_moe_dag()
    
    # Save DOT file explicitly
    dot_content = dag.source
    dot_path = '../outputs/2025-12-01-19-00-59/moe_single_gpu_deployment.dot'
    with open(dot_path, 'w') as f:
        f.write(dot_content)
    
    # Render to SVG
    svg_path = '../outputs/2025-12-01-19-00-59/moe_single_gpu_deployment.svg'
    dag.render('../outputs/2025-12-01-19-00-59/moe_single_gpu_deployment', format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_path}")
    print(f"SVG file: {svg_path}")
    
    return {
        'dot_path': dot_path,
        'svg_path': svg_path
    }

if __name__ == '__main__':
    main()