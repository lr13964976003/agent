#!/usr/bin/env python3

import graphviz
import os

def create_moe_dag():
    """
    Create a comprehensive DAG for the 30B MoE model deployment
    with hybrid parallelism: 4-way tensor, 16-way expert, 4-stage pipeline, 2-way data
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial', fontsize='10')
    
    # Define colors for different GPU groups
    colors = {
        'stage0': '#FFE5CC',  # Light orange
        'stage1': '#E5F3FF',  # Light blue  
        'stage2': '#E5FFE5',  # Light green
        'stage3': '#F3E5FF',  # Light purple
        'communication': '#FFD700',  # Gold
        'routing': '#FFCCCB',  # Light red
        'aggregation': '#DDA0DD'  # Plum
    }
    
    # Model parameters
    batch_size = 128
    seq_len = 1024  # Max sequence length
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    ffn_hidden_size = 2048
    num_experts = 64
    experts_per_gpu = 4
    micro_batch_size = 8
    
    # GPU allocation: 4 stages × 4 tensor parallel = 16 GPUs total
    # Stage 0: GPUs 0-3, Stage 1: GPUs 4-7, Stage 2: GPUs 8-11, Stage 3: GPUs 12-15
    
    def add_input_node(node_id, label, gpu_id, input_dims):
        """Add input node (ellipse shape)"""
        dot.node(node_id, label, shape='ellipse', style='filled', fillcolor='white',
                fontsize='9', fontname='Arial')
        dot.attr('node', shape='rectangle')
    
    def add_communication_node(node_id, label, input_dims, output_dims):
        """Add communication node (ellipse shape)"""
        comm_label = f"{label}\\nInput: {input_dims}\\nOutput: {output_dims}"
        dot.node(node_id, comm_label, shape='ellipse', style='filled', 
                fillcolor=colors['communication'], fontsize='9')
    
    def add_routing_node(node_id, label, input_dims, output_dims):
        """Add routing node (parallelogram shape)"""
        route_label = f"{label}\\nInput: {input_dims}\\nOutput: {output_dims}"
        dot.node(node_id, route_label, shape='parallelogram', style='filled',
                fillcolor=colors['routing'], fontsize='9')
    
    def add_aggregation_node(node_id, label, input_dims, output_dims):
        """Add aggregation node (parallelogram shape)"""
        agg_label = f"{label}\\nInput: {input_dims}\\nOutput: {output_dims}"
        dot.node(node_id, agg_label, shape='parallelogram', style='filled',
                fillcolor=colors['aggregation'], fontsize='9')
    
    def add_computation_node(node_id, label, gpu_id, input_dims, output_dims, color=None):
        """Add computation node with GPU label"""
        if color is None:
            if gpu_id < 4:
                color = colors['stage0']
            elif gpu_id < 8:
                color = colors['stage1']
            elif gpu_id < 12:
                color = colors['stage2']
            else:
                color = colors['stage3']
        
        full_label = f"{label}\\nGPU {gpu_id}\\nInput: {input_dims}\\nOutput: {output_dims}"
        dot.node(node_id, full_label, style='filled', fillcolor=color, fontsize='9')
    
    # Input nodes for data parallelism (2-way)
    add_input_node('input_dp0', 'Input Batch (DP=0)', 0, 
                  f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]')
    add_input_node('input_dp1', 'Input Batch (DP=1)', 1, 
                  f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]')
    
    # Split to micro-batches
    add_aggregation_node('split_mb0', 'Split to Micro-batches', 
                        f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]',
                        f'8×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
    add_aggregation_node('split_mb1', 'Split to Micro-batches', 
                        f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]',
                        f'8×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
    
    dot.edge('input_dp0', 'split_mb0')
    dot.edge('input_dp1', 'split_mb1')
    
    # Process each micro-batch through pipeline stages
    for mb in range(8):  # 8 micro-batches
        prev_node_dp0 = 'split_mb0'
        prev_node_dp1 = 'split_mb1'
        
        # Stage 0: Layers 0-3 (GPUs 0-3)
        for layer in range(4):
            gpu_base = 0  # Stage 0 starts at GPU 0
            
            # Attention Layer - broken down into detailed steps
            # Step 1: QKV Projection (Column-parallel across 4 GPUs)
            for tp in range(4):  # 4-way tensor parallelism
                gpu_id = gpu_base + tp
                
                # QKV Projection
                add_computation_node(f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} QKV Projection', gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]')
                
                add_computation_node(f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} QKV Projection', gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]')
                
                dot.edge(prev_node_dp0, f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}')
                dot.edge(prev_node_dp1, f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}')
            
            # Step 2: Attention Scores and Softmax (distributed)
            for tp in range(4):
                gpu_id = gpu_base + tp
                
                add_computation_node(f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Scores', gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]',
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                
                add_computation_node(f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Scores', gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]',
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                
                dot.edge(f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}')
                dot.edge(f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}')
                
                # Attention Softmax
                add_computation_node(f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Softmax', gpu_id,
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                
                add_computation_node(f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Softmax', gpu_id,
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                
                dot.edge(f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}')
                dot.edge(f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}')
            
            # Step 3: Attention Output (Row-parallel)
            for tp in range(4):
                gpu_id = gpu_base + tp
                
                add_computation_node(f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Output', gpu_id,
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]')
                
                add_computation_node(f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}', 
                                   f'Layer {layer} Attention Output', gpu_id,
                                   f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]')
                
                dot.edge(f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}')
                dot.edge(f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}')
            
            # All-reduce for attention output
            add_communication_node(f'layer{layer}_attn_allreduce_dp0_mb{mb}', 
                                 'All-Reduce Attention Output',
                                 f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]',
                                 f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
            
            add_communication_node(f'layer{layer}_attn_allreduce_dp1_mb{mb}', 
                                 'All-Reduce Attention Output',
                                 f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]',
                                 f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
            
            for tp in range(4):
                gpu_id = gpu_base + tp
                dot.edge(f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_allreduce_dp0_mb{mb}')
                dot.edge(f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_allreduce_dp1_mb{mb}')
            
            # MoE Layer - Expert Parallelism (16-way, 4 experts per GPU)
            # Expert Routing
            add_routing_node(f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}', 
                           f'Layer {layer} Expert Router',
                           f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                           f'[batch={micro_batch_size}, seq={seq_len}, top_k=1]')
            
            add_routing_node(f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}', 
                           f'Layer {layer} Expert Router',
                           f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                           f'[batch={micro_batch_size}, seq={seq_len}, top_k=1]')
            
            dot.edge(f'layer{layer}_attn_allreduce_dp0_mb{mb}', f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}')
            dot.edge(f'layer{layer}_attn_allreduce_dp1_mb{mb}', f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}')
            
            # Expert computation (4 experts per GPU in this stage)
            for expert in range(experts_per_gpu):
                expert_gpu_id = gpu_base + expert  # 4 experts distributed across stage GPUs
                
                # Expert MLP (Column-parallel within each expert)
                add_computation_node(f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}', 
                                   f'Layer {layer} Expert {expert} MLP1', expert_gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]')
                
                add_computation_node(f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}', 
                                   f'Layer {layer} Expert {expert} MLP1', expert_gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]')
                
                dot.edge(f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}', f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}')
                dot.edge(f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}', f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}')
                
                # Expert MLP2
                add_computation_node(f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}', 
                                   f'Layer {layer} Expert {expert} MLP2', expert_gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                add_computation_node(f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}', 
                                   f'Layer {layer} Expert {expert} MLP2', expert_gpu_id,
                                   f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]',
                                   f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                dot.edge(f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}')
                dot.edge(f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}')
            
            # Expert aggregation (all-to-all communication)
            add_communication_node(f'layer{layer}_expert_agg_dp0_mb{mb}', 
                                 'All-to-All Expert Aggregation',
                                 f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                 f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
            
            add_communication_node(f'layer{layer}_expert_agg_dp1_mb{mb}', 
                                 'All-to-All Expert Aggregation',
                                 f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                 f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
            
            for expert in range(experts_per_gpu):
                expert_gpu_id = gpu_base + expert
                dot.edge(f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert_agg_dp0_mb{mb}')
                dot.edge(f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert_agg_dp1_mb{mb}')
            
            prev_node_dp0 = f'layer{layer}_expert_agg_dp0_mb{mb}'
            prev_node_dp1 = f'layer{layer}_expert_agg_dp1_mb{mb}'
    
    # Continue with remaining stages (1, 2, 3) - similar structure but different GPU allocations
    # Stage 1: Layers 4-7 (GPUs 4-7)
    for stage in range(1, 4):
        gpu_base = stage * 4  # Stage 1: GPUs 4-7, Stage 2: GPUs 8-11, Stage 3: GPUs 12-15
        
        for mb in range(8):
            prev_node_dp0 = f'layer3_expert_agg_dp0_mb{mb}' if stage == 1 else f'layer7_expert_agg_dp0_mb{mb}' if stage == 2 else f'layer11_expert_agg_dp0_mb{mb}'
            prev_node_dp1 = f'layer3_expert_agg_dp1_mb{mb}' if stage == 1 else f'layer7_expert_agg_dp1_mb{mb}' if stage == 2 else f'layer11_expert_agg_dp1_mb{mb}'
            
            for layer in range(stage * 4, (stage + 1) * 4):
                # Attention Layer (same structure as stage 0)
                for tp in range(4):
                    gpu_id = gpu_base + tp
                    
                    # QKV Projection
                    add_computation_node(f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} QKV Projection', gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]')
                    
                    add_computation_node(f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} QKV Projection', gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]')
                    
                    dot.edge(prev_node_dp0, f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}')
                    dot.edge(prev_node_dp1, f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}')
                    
                    # Attention Scores and Softmax
                    add_computation_node(f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Scores', gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]',
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                    
                    add_computation_node(f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Scores', gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, heads={num_heads//4}, d_k={head_dim}]',
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                    
                    dot.edge(f'layer{layer}_qkv_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}')
                    dot.edge(f'layer{layer}_qkv_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}')
                    
                    # Attention Softmax
                    add_computation_node(f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Softmax', gpu_id,
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                    
                    add_computation_node(f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Softmax', gpu_id,
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]')
                    
                    dot.edge(f'layer{layer}_attn_scores_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}')
                    dot.edge(f'layer{layer}_attn_scores_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}')
                    
                    # Attention Output
                    add_computation_node(f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Output', gpu_id,
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]')
                    
                    add_computation_node(f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}', 
                                       f'Layer {layer} Attention Output', gpu_id,
                                       f'[batch={micro_batch_size}, heads={num_heads//4}, seq={seq_len}, seq={seq_len}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]')
                    
                    dot.edge(f'layer{layer}_softmax_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}')
                    dot.edge(f'layer{layer}_softmax_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}')
                
                # All-reduce for attention output
                add_communication_node(f'layer{layer}_attn_allreduce_dp0_mb{mb}', 
                                     'All-Reduce Attention Output',
                                     f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]',
                                     f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                add_communication_node(f'layer{layer}_attn_allreduce_dp1_mb{mb}', 
                                     'All-Reduce Attention Output',
                                     f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size//4}]',
                                     f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                for tp in range(4):
                    gpu_id = gpu_base + tp
                    dot.edge(f'layer{layer}_attn_out_dp0_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_allreduce_dp0_mb{mb}')
                    dot.edge(f'layer{layer}_attn_out_dp1_mb{mb}_gpu{gpu_id}', f'layer{layer}_attn_allreduce_dp1_mb{mb}')
                
                # MoE Layer - Expert Parallelism
                # Expert Routing
                add_routing_node(f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}', 
                               f'Layer {layer} Expert Router',
                               f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                               f'[batch={micro_batch_size}, seq={seq_len}, top_k=1]')
                
                add_routing_node(f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}', 
                               f'Layer {layer} Expert Router',
                               f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                               f'[batch={micro_batch_size}, seq={seq_len}, top_k=1]')
                
                dot.edge(f'layer{layer}_attn_allreduce_dp0_mb{mb}', f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}')
                dot.edge(f'layer{layer}_attn_allreduce_dp1_mb{mb}', f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}')
                
                # Expert computation
                for expert in range(experts_per_gpu):
                    expert_gpu_id = (gpu_base + expert) % 16  # Distribute experts across all GPUs
                    
                    # Expert MLP1
                    add_computation_node(f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}', 
                                       f'Layer {layer} Expert {expert} MLP1', expert_gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]')
                    
                    add_computation_node(f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}', 
                                       f'Layer {layer} Expert {expert} MLP1', expert_gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]')
                    
                    dot.edge(f'layer{layer}_router_dp0_mb{mb}_gpu{gpu_base}', f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}')
                    dot.edge(f'layer{layer}_router_dp1_mb{mb}_gpu{gpu_base}', f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}')
                    
                    # Expert MLP2
                    add_computation_node(f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}', 
                                       f'Layer {layer} Expert {expert} MLP2', expert_gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                    
                    add_computation_node(f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}', 
                                       f'Layer {layer} Expert {expert} MLP2', expert_gpu_id,
                                       f'[batch={micro_batch_size}, seq={seq_len}, ffn={ffn_hidden_size}]',
                                       f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                    
                    dot.edge(f'layer{layer}_expert{expert}_mlp1_dp0_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}')
                    dot.edge(f'layer{layer}_expert{expert}_mlp1_dp1_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}')
                
                # Expert aggregation
                add_communication_node(f'layer{layer}_expert_agg_dp0_mb{mb}', 
                                     'All-to-All Expert Aggregation',
                                     f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                     f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                add_communication_node(f'layer{layer}_expert_agg_dp1_mb{mb}', 
                                     'All-to-All Expert Aggregation',
                                     f'4×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                                     f'[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]')
                
                for expert in range(experts_per_gpu):
                    expert_gpu_id = (gpu_base + expert) % 16
                    dot.edge(f'layer{layer}_expert{expert}_mlp2_dp0_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert_agg_dp0_mb{mb}')
                    dot.edge(f'layer{layer}_expert{expert}_mlp2_dp1_mb{mb}_gpu{expert_gpu_id}', f'layer{layer}_expert_agg_dp1_mb{mb}')
                
                prev_node_dp0 = f'layer{layer}_expert_agg_dp0_mb{mb}'
                prev_node_dp1 = f'layer{layer}_expert_agg_dp1_mb{mb}'
    
    # Final output aggregation and gradient synchronization
    add_aggregation_node('final_agg_dp0', 'Final Output Aggregation',
                        f'8×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                        f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]')
    
    add_aggregation_node('final_agg_dp1', 'Final Output Aggregation',
                        f'8×[batch={micro_batch_size}, seq={seq_len}, hidden={hidden_size}]',
                        f'[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]')
    
    # Gradient synchronization (async all-reduce)
    add_communication_node('grad_sync', 'Async All-Reduce Gradients',
                          f'2×[batch={batch_size//2}, seq={seq_len}, hidden={hidden_size}]',
                          f'[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]')
    
    # Connect final layer outputs to aggregation
    for mb in range(8):
        dot.edge(f'layer15_expert_agg_dp0_mb{mb}', 'final_agg_dp0')
        dot.edge(f'layer15_expert_agg_dp1_mb{mb}', 'final_agg_dp1')
    
    dot.edge('final_agg_dp0', 'grad_sync')
    dot.edge('final_agg_dp1', 'grad_sync')
    
    # Output nodes
    add_input_node('output', 'Final Output', 0, 
                  f'[batch={batch_size}, seq={seq_len}, hidden={hidden_size}]')
    dot.edge('grad_sync', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-05-10-26-38/moe_deployment_dag.dot'
    dag.save(dot_file_path)
    print(f"DOT file saved to: {dot_file_path}")
    
    # Render as SVG
    svg_file_path = '../outputs/2025-12-05-10-26-38/moe_deployment_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved to: {svg_file_path}")
    
    # Also render as PNG for easier viewing
    png_file_path = '../outputs/2025-12-05-10-26-38/moe_deployment_dag.png'
    dag.render(png_file_path.replace('.png', ''), format='png', cleanup=True)
    print(f"PNG file saved to: {png_file_path}")
    
    print("DAG generation completed successfully!")