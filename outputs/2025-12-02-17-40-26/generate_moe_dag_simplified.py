#!/usr/bin/env python3
"""
Generate simplified but complete DAG for EP64_TP2 hybrid parallel MoE model
Shows one complete layer with all 128 GPUs, then indicates repetition pattern
"""

import graphviz
from graphviz import Digraph

def create_simplified_moe_dag():
    """Create simplified but complete DAG showing one full layer pattern"""
    
    # Create directed graph
    dot = Digraph(comment='EP64_TP2 Hybrid Parallel MoE Model DAG - Simplified Complete View')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.5')
    
    # Model parameters
    batch_size = 128
    seq_len = 1024
    token_dim = 1024
    moe_hidden_dim = 2048
    num_heads = 16
    d_k = token_dim // num_heads  # 64
    
    # GPU organization: 64 expert groups, 2 GPUs per group
    num_expert_groups = 64
    gpus_per_group = 2
    total_gpus = 128
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Create input node
    dot.node('input', 
             f'INPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='octagon', style='filled', fillcolor='white')
    
    # Process Layer 0 in detail (showing all 128 GPUs)
    layer_name = 'layer_0'
    
    # Layer Norm 1 (on all GPUs)
    for gpu_id in range(total_gpus):
        dot.node(f'{layer_name}_ln1_gpu{gpu_id}',
                 f'LayerNorm1_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Multi-Head Attention with Tensor Parallelism (degree 2)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        
        # QKV projection with column parallelism
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.node(f'{layer_name}_qkv_gpu{gpu_id}',
                     f'QKV_Proj_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//gpus_per_group}, d_k={d_k}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention computation
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.node(f'{layer_name}_attn_gpu{gpu_id}',
                     f'Attention_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//gpus_per_group}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//gpus_per_group}, d_k={d_k}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention output projection with row parallelism
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.node(f'{layer_name}_attn_out_gpu{gpu_id}',
                     f'Attn_Out_Proj_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//gpus_per_group}, d_k={d_k}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//gpus_per_group}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # All-reduce for attention output
        dot.node(f'{layer_name}_attn_allreduce_group{expert_group}',
                 f'AllReduce_Attn_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//gpus_per_group}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Residual connection after attention
    for gpu_id in range(total_gpus):
        dot.node(f'{layer_name}_residual1_gpu{gpu_id}',
                 f'Residual_Add1_GPU{gpu_id}\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Layer Norm 2 (on all GPUs)
    for gpu_id in range(total_gpus):
        dot.node(f'{layer_name}_ln2_gpu{gpu_id}',
                 f'LayerNorm2_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Gate network (routing)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.node(f'{layer_name}_gate_gpu{gpu_id}',
                     f'Gate_Network_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=64]',
                     shape='rectangle', style='filled,dashed', fillcolor='lightgreen')
    
    # Expert selection and routing
    for expert_group in range(num_expert_groups):
        dot.node(f'{layer_name}_router_group{expert_group}',
                 f'Router_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Expert computation (1 expert per GPU)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            expert_id = gpu_id  # 1 expert per GPU
            
            # Expert MLP with tensor parallelism
            # First linear (column parallel)
            dot.node(f'{layer_name}_expert1_gpu{gpu_id}',
                     f'Expert{expert_id}_Linear1_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim//gpus_per_group}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Activation
            dot.node(f'{layer_name}_expert_act_gpu{gpu_id}',
                     f'Expert{expert_id}_GELU_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim//gpus_per_group}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim//gpus_per_group}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Second linear (row parallel)
            dot.node(f'{layer_name}_expert2_gpu{gpu_id}',
                     f'Expert{expert_id}_Linear2_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim//gpus_per_group}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//gpus_per_group}]',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # All-reduce for expert outputs within group
        dot.node(f'{layer_name}_expert_allreduce_group{expert_group}',
                 f'AllReduce_Expert_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//gpus_per_group}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Final aggregation across all experts
    for expert_group in range(num_expert_groups):
        dot.node(f'{layer_name}_aggregate_group{expert_group}',
                 f'Aggregate_Experts_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Final all-reduce across all expert groups
    dot.node(f'{layer_name}_final_allreduce',
             f'AllReduce_Final_Layer0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Residual connection after MoE
    for gpu_id in range(total_gpus):
        dot.node(f'{layer_name}_residual2_gpu{gpu_id}',
                 f'Residual_Add2_GPU{gpu_id}\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Create summary nodes for remaining layers
    dot.node('layers_1_15_summary',
             'Layers 1-15 (Identical Pattern)\\nEach layer: 128 GPUs, 64 expert groups, TP2\\nSame computation and communication pattern as Layer 0',
             shape='note', style='filled', fillcolor='lightgray')
    
    # Create output node
    dot.node('output',
             f'OUTPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='octagon', style='filled', fillcolor='white')
    
    # Connect Layer 0
    # Input to layer norm 1
    for gpu_id in range(total_gpus):
        dot.edge('input', f'{layer_name}_ln1_gpu{gpu_id}')
    
    # Layer norm 1 to QKV projection
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_ln1_gpu{gpu_id}', f'{layer_name}_qkv_gpu{gpu_id}')
    
    # QKV to attention
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_qkv_gpu{gpu_id}', f'{layer_name}_attn_gpu{gpu_id}')
    
    # Attention to attention output
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_attn_gpu{gpu_id}', f'{layer_name}_attn_out_gpu{gpu_id}')
    
    # Attention output to all-reduce
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_attn_out_gpu{gpu_id}', f'{layer_name}_attn_allreduce_group{expert_group}')
    
    # All-reduce to residual 1
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_attn_allreduce_group{expert_group}', f'{layer_name}_residual1_gpu{gpu_id}')
            # Skip connection
            dot.edge('input', f'{layer_name}_residual1_gpu{gpu_id}')
    
    # Residual 1 to layer norm 2
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_residual1_gpu{gpu_id}', f'{layer_name}_ln2_gpu{gpu_id}')
    
    # Layer norm 2 to gate
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_ln2_gpu{gpu_id}', f'{layer_name}_gate_gpu{gpu_id}')
    
    # Gate to router (dashed line)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        dot.edge(f'{layer_name}_gate_gpu{gpu_base}', f'{layer_name}_router_group{expert_group}', style='dashed')
        if gpus_per_group > 1:
            dot.edge(f'{layer_name}_gate_gpu{gpu_base+1}', f'{layer_name}_router_group{expert_group}', style='dashed')
    
    # Router to expert computation
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_router_group{expert_group}', f'{layer_name}_expert1_gpu{gpu_id}')
    
    # Expert computation chain
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_expert1_gpu{gpu_id}', f'{layer_name}_expert_act_gpu{gpu_id}')
            dot.edge(f'{layer_name}_expert_act_gpu{gpu_id}', f'{layer_name}_expert2_gpu{gpu_id}')
    
    # Expert outputs to all-reduce
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_expert2_gpu{gpu_id}', f'{layer_name}_expert_allreduce_group{expert_group}')
    
    # All-reduce to aggregation
    for expert_group in range(num_expert_groups):
        dot.edge(f'{layer_name}_expert_allreduce_group{expert_group}', f'{layer_name}_aggregate_group{expert_group}')
    
    # Aggregation to final all-reduce
    for expert_group in range(num_expert_groups):
        dot.edge(f'{layer_name}_aggregate_group{expert_group}', f'{layer_name}_final_allreduce')
    
    # Final all-reduce to residual 2
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_final_allreduce', f'{layer_name}_residual2_gpu{gpu_id}')
        # Skip connection
        dot.edge(f'{layer_name}_residual1_gpu{gpu_id}', f'{layer_name}_residual2_gpu{gpu_id}')
    
    # Connect to summary and output
    dot.edge(f'{layer_name}_residual2_gpu0', 'layers_1_15_summary')
    dot.edge('layers_1_15_summary', 'output')
    
    return dot

if __name__ == '__main__':
    # Generate the DAG
    dag = create_simplified_moe_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete.dot')
    
    # Try to save as SVG (may still be large)
    try:
        dag.render('../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete', format='svg', cleanup=True)
        print("SVG image generated successfully!")
    except:
        print("SVG generation failed due to size, but DOT file is complete")
    
    print("Complete DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete.dot")
    print(f"SVG image: ../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete.svg (if generated)")