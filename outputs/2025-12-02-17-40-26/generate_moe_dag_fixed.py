#!/usr/bin/env python3
"""
Generate complete DAG for EP64_TP2 hybrid parallel MoE model - FIXED VERSION
Properly connects all GPU outputs to ensure no dead-end nodes
"""

import graphviz
from graphviz import Digraph

def create_fixed_moe_dag():
    """Create complete DAG with proper connectivity for all GPUs"""
    
    # Create directed graph
    dot = Digraph(comment='EP64_TP2 Hybrid Parallel MoE Model DAG - Fixed Complete View')
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
                 shape='parallelogram', style='filled,dashed', fillcolor='lightyellow')
    
    # Expert computation (one expert per GPU)
    for gpu_id in range(total_gpus):
        # Expert MLP layer 1
        dot.node(f'{layer_name}_expert1_gpu{gpu_id}',
                 f'Expert_MLP1_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert activation
        dot.node(f'{layer_name}_expert_act_gpu{gpu_id}',
                 f'SiLU_Activation_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert MLP layer 2
        dot.node(f'{layer_name}_expert2_gpu{gpu_id}',
                 f'Expert_MLP2_GPU{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={moe_hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Expert all-reduce within groups
    for expert_group in range(num_expert_groups):
        dot.node(f'{layer_name}_expert_allreduce_group{expert_group}',
                 f'AllReduce_Expert_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Expert aggregation within groups
    for expert_group in range(num_expert_groups):
        dot.node(f'{layer_name}_aggregate_group{expert_group}',
                 f'Expert_Aggregation_Group{expert_group}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Final all-reduce across all expert groups
    dot.node(f'{layer_name}_final_allreduce',
             f'Final_AllReduce\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Residual connection after experts
    for gpu_id in range(total_gpus):
        dot.node(f'{layer_name}_residual2_gpu{gpu_id}',
                 f'Residual_Add2_GPU{gpu_id}\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # FIXED: Create proper aggregation layer that combines outputs from all GPUs
    dot.node('layer_0_final_aggregate',
             f'Final_Aggregate_All_Groups\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 128\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Summary node for layers 1-15 (simplified representation)
    dot.node('layers_1_15_summary',
             'Layers_1_15_Summary\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', style='filled,dashed', fillcolor='lightgreen')
    
    # Create output node
    dot.node('output',
             f'OUTPUT\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='octagon', style='filled', fillcolor='white')
    
    # Create connections
    # Input to all LayerNorm1 nodes
    for gpu_id in range(total_gpus):
        dot.edge('input', f'{layer_name}_ln1_gpu{gpu_id}')
    
    # LayerNorm1 to QKV
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_ln1_gpu{gpu_id}', f'{layer_name}_qkv_gpu{gpu_id}')
    
    # QKV to Attention
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_qkv_gpu{gpu_id}', f'{layer_name}_attn_gpu{gpu_id}')
    
    # Attention to Attention Output
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_attn_gpu{gpu_id}', f'{layer_name}_attn_out_gpu{gpu_id}')
    
    # Attention Output to All-reduce (grouped)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_attn_out_gpu{gpu_id}', f'{layer_name}_attn_allreduce_group{expert_group}')
    
    # All-reduce to Residual1
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_attn_allreduce_group{expert_group}', f'{layer_name}_residual1_gpu{gpu_id}')
            dot.edge('input', f'{layer_name}_residual1_gpu{gpu_id}')  # Residual connection
    
    # Residual1 to LayerNorm2
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_residual1_gpu{gpu_id}', f'{layer_name}_ln2_gpu{gpu_id}')
    
    # LayerNorm2 to Gate
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_ln2_gpu{gpu_id}', f'{layer_name}_gate_gpu{gpu_id}')
    
    # Gate to Router (grouped)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_gate_gpu{gpu_id}', f'{layer_name}_router_group{expert_group}')
    
    # Router to Expert MLP1
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_router_group{expert_group}', f'{layer_name}_expert1_gpu{gpu_id}')
    
    # Expert MLP1 to Activation
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_expert1_gpu{gpu_id}', f'{layer_name}_expert_act_gpu{gpu_id}')
    
    # Activation to Expert MLP2
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_expert_act_gpu{gpu_id}', f'{layer_name}_expert2_gpu{gpu_id}')
    
    # Expert MLP2 to Expert All-reduce (grouped)
    for expert_group in range(num_expert_groups):
        gpu_base = expert_group * gpus_per_group
        for gpu_offset in range(gpus_per_group):
            gpu_id = gpu_base + gpu_offset
            dot.edge(f'{layer_name}_expert2_gpu{gpu_id}', f'{layer_name}_expert_allreduce_group{expert_group}')
    
    # Expert All-reduce to Aggregation
    for expert_group in range(num_expert_groups):
        dot.edge(f'{layer_name}_expert_allreduce_group{expert_group}', f'{layer_name}_aggregate_group{expert_group}')
    
    # Aggregation to Final All-reduce
    for expert_group in range(num_expert_groups):
        dot.edge(f'{layer_name}_aggregate_group{expert_group}', f'{layer_name}_final_allreduce')
    
    # Final All-reduce to Residual2
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_final_allreduce', f'{layer_name}_residual2_gpu{gpu_id}')
        dot.edge(f'{layer_name}_residual1_gpu{gpu_id}', f'{layer_name}_residual2_gpu{gpu_id}')  # Residual connection
    
    # FIXED: Connect ALL residual2 outputs to the final aggregation node
    for gpu_id in range(total_gpus):
        dot.edge(f'{layer_name}_residual2_gpu{gpu_id}', 'layer_0_final_aggregate')
    
    # Final aggregation to layers summary
    dot.edge('layer_0_final_aggregate', 'layers_1_15_summary')
    
    # Layers summary to output
    dot.edge('layers_1_15_summary', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_fixed_moe_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete_fixed.dot')
    
    # Save as SVG image
    dag.render('../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete_fixed', format='svg', cleanup=True)
    
    print("Fixed DAG generated successfully!")
    print("Files saved:")
    print("- ../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete_fixed.dot")
    print("- ../outputs/2025-12-02-17-40-26/ep64_tp2_moe_model_complete_fixed.svg")