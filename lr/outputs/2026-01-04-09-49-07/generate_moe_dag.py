#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def create_moe_dag():
    # Create a new directed graph
    dot = Digraph(comment='Qwen3-235B MoE Parallel Strategy DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='rectangle', fillcolor='lightcoral')
    
    # Token embedding (all GPUs)
    for gpu_id in range(8):
        dot.node(f'embed_gpu{gpu_id}', f'GPU{gpu_id}: Token Embedding\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                 shape='rectangle', fillcolor='lightgreen')
        dot.edge('input', f'embed_gpu{gpu_id}')
    
    # Layer processing - showing a representative layer (layer 1)
    layer_idx = 1
    
    # Attention computation across all GPUs (TP=8)
    for gpu_id in range(8):
        # Attention - QKV projection
        dot.node(f'attn_qkv_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Layer{layer_idx} Attention QKV Proj\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]', 
                 shape='rectangle', fillcolor='lightgreen')
        
        # All-reduce for QKV (communication)
        dot.node(f'comm_qkv_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: All-Reduce QKV\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Attention computation
        dot.node(f'attn_comp_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Layer{layer_idx} Attention Compute\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=8, d_k=64]', 
                 shape='rectangle', fillcolor='lightgreen')
        
        # All-reduce for attention output
        dot.node(f'comm_attn_out_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: All-Reduce Attention Output\\nInput: [batch_size=128, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Connections for attention
        dot.edge(f'embed_gpu{gpu_id}', f'attn_qkv_gpu{gpu_id}_layer{layer_idx}')
        dot.edge(f'attn_qkv_gpu{gpu_id}_layer{layer_idx}', f'comm_qkv_gpu{gpu_id}_layer{layer_idx}')
        dot.edge(f'comm_qkv_gpu{gpu_id}_layer{layer_idx}', f'attn_comp_gpu{gpu_id}_layer{layer_idx}')
        dot.edge(f'attn_comp_gpu{gpu_id}_layer{layer_idx}', f'comm_attn_out_gpu{gpu_id}_layer{layer_idx}')
    
    # MoE layer - Gate computation (routing)
    for gpu_id in range(8):
        dot.node(f'gate_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Layer{layer_idx} Gate Routing\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=8]', 
                 shape='parallelogram', fillcolor='lightyellow')
        
        # Connect gate to attention output
        dot.edge(f'comm_attn_out_gpu{gpu_id}_layer{layer_idx}', f'gate_gpu{gpu_id}_layer{layer_idx}')
    
    # Expert processing - each GPU has 16 experts, showing first few
    for gpu_id in range(8):
        for expert_id in range(4):  # Show first 4 experts per GPU for clarity
            dot.node(f'expert_gpu{gpu_id}_exp{expert_id}_layer{layer_idx}', 
                     f'GPU{gpu_id}: Expert {expert_id}\\nInput: [batch_size=?, seq_len=?, hidden=4096]\\nOutput: [batch_size=?, seq_len=?, hidden=1536]', 
                     shape='rectangle', fillcolor='lightgreen')
    
    # Expert selection with dashed lines (gate decisions)
    for gpu_id in range(8):
        for target_gpu in range(8):
            # Dashed lines for expert selection (gate decisions)
            dot.edge(f'gate_gpu{gpu_id}_layer{layer_idx}', 
                     f'expert_gpu{target_gpu}_0_layer{layer_idx}', 
                     style='dashed', color='red', 
                     label=f'GPU{gpu_id} selects experts on GPU{target_gpu}')
    
    # Communication for expert routing
    for gpu_id in range(8):
        dot.node(f'comm_expert_send_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Send Tokens to Experts\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=?, seq_len=?, hidden=4096]', 
                 shape='ellipse', fillcolor='lightblue')
        
        dot.node(f'comm_expert_recv_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Receive from Experts\\nInput: [batch_size=?, seq_len=?, hidden=1536]\\nOutput: [batch_size=128, seq_len=2048, hidden=1536]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Connect gate to communication
        dot.edge(f'gate_gpu{gpu_id}_layer{layer_idx}', f'comm_expert_send_gpu{gpu_id}_layer{layer_idx}')
        
        # Connect communication to experts and back
        for expert_id in range(4):
            dot.edge(f'comm_expert_send_gpu{gpu_id}_layer{layer_idx}', 
                     f'expert_gpu{gpu_id}_exp{expert_id}_layer{layer_idx}')
            dot.edge(f'expert_gpu{gpu_id}_exp{expert_id}_layer{layer_idx}', 
                     f'comm_expert_recv_gpu{gpu_id}_layer{layer_idx}')
    
    # Expert aggregation
    for gpu_id in range(8):
        dot.node(f'expert_agg_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: Expert Aggregation\\nInput: [batch_size=128, seq_len=2048, experts=8, hidden=1536]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                 shape='parallelogram', fillcolor='lightyellow')
        
        dot.node(f'mlp_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: MLP Compute\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'mlp_agg_gpu{gpu_id}_layer{layer_idx}', 
                 f'GPU{gpu_id}: MLP All-Reduce\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Connect expert communication to aggregation
        dot.edge(f'comm_expert_recv_gpu{gpu_id}_layer{layer_idx}', f'expert_agg_gpu{gpu_id}_layer{layer_idx}')
        dot.edge(f'expert_agg_gpu{gpu_id}_layer{layer_idx}', f'mlp_gpu{gpu_id}_layer{layer_idx}')
        dot.edge(f'mlp_gpu{gpu_id}_layer{layer_idx}', f'mlp_agg_gpu{gpu_id}_layer{layer_idx}')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='rectangle', fillcolor='lightcoral')
    
    # Connect final layer to output
    for gpu_id in range(8):
        dot.edge(f'mlp_agg_gpu{gpu_id}_layer{layer_idx}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_moe_dag()
    
    # Get the absolute path
    output_dir = os.path.abspath('./outputs/2026-01-04-09-49-07')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as DOT file
    dot_path = os.path.join(output_dir, 'moe_parallel_dag.dot')
    svg_path = os.path.join(output_dir, 'moe_parallel_dag.svg')
    
    print(f"Saving DOT file to: {dot_path}")
    print(f"Saving SVG file to: {svg_path}")
    
    dag.save(dot_path)
    
    # Render as SVG
    dag.render(svg_path, format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files created:")
    print(f"- {dot_path} (Graphviz source)")
    print(f"- {svg_path} (Visual diagram)")