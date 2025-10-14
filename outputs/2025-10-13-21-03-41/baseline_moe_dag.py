#!/usr/bin/env python3
"""
Baseline MoE DAG with TP=8, PP=2
16 GPUs total: 2 pipeline stages × 8 tensor parallel ranks
Each GPU holds 8 experts colocated
"""

import graphviz

def create_baseline_moe_dag():
    """Create complete DAG for baseline MoE with TP=8, PP=2"""
    
    # Create directed graph
    dot = graphviz.Digraph('baseline_moe_tp8_pp2', 
                          comment='Baseline MoE (TP=8, PP=2)')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,40', compound='true')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define colors for different components
    colors = {
        'input': 'lightblue',
        'attention': 'lightgreen',
        'gate': 'lightyellow',
        'expert': 'lightcoral',
        'communication': 'lightgray',
        'aggregation': 'lightpink',
        'output': 'lightblue',
        'pipeline': 'lightsteelblue'
    }
    
    # Model dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    mha_heads = 16
    head_dim = 512
    ffn_hidden = 32768
    tp_degree = 8
    pp_degree = 2
    experts_per_gpu = 8
    
    # Tensor parallel dimensions
    tp_hidden = hidden_dim // tp_degree
    tp_ffn = ffn_hidden // tp_degree
    
    # =================================================================================
    # PIPELINE STAGE 0 (GPUs 0-7)
    # =================================================================================
    
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='rounded', color='black', bgcolor='lightcyan')
        
        # Stage 0 Input
        stage0.node('stage0_input', f'Stage 0 Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: 0-7',
                   shape='ellipse', style='filled', fillcolor=colors['input'])
        
        # =============================================================================
        # LAYER 0 in Stage 0
        # =============================================================================
        
        with stage0.subgraph(name='cluster_layer0_s0') as layer0_s0:
            layer0_s0.attr(label='Layer 0 (Stage 0)', style='rounded', color='blue')
            
            # Tensor parallel attention across 8 GPUs
            for tp_rank in range(8):
                gpu_id = tp_rank
                
                # Input split for tensor parallelism
                layer0_s0.node(f'input_split_l0_tp{tp_rank}',
                             f'Input Split TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                # MHA QKV Linear (column parallel)
                layer0_s0.node(f'mha_qkv_l0_tp{tp_rank}',
                             f'MHA QKV TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                # MHA Attention
                layer0_s0.node(f'mha_attn_l0_tp{tp_rank}',
                             f'MHA Attention TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                # MHA Output Linear (row parallel)
                layer0_s0.node(f'mha_out_l0_tp{tp_rank}',
                             f'MHA Output TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                # All-reduce for attention output
                layer0_s0.node(f'attn_allreduce_l0_tp{tp_rank}',
                             f'Attention All-Reduce TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                # Residual add and layer norm
                layer0_s0.node(f'res_add_l0_tp{tp_rank}',
                             f'Residual Add TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer0_s0.node(f'norm_l0_tp{tp_rank}',
                             f'Layer Norm TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
            
            # Gate network (replicated across all GPUs)
            for tp_rank in range(8):
                gpu_id = tp_rank
                layer0_s0.node(f'gate_l0_tp{tp_rank}',
                             f'Gate Network TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['gate'])
            
            # 8 experts per GPU (colocated)
            for tp_rank in range(8):
                gpu_id = tp_rank
                for expert_idx in range(experts_per_gpu):
                    expert_id = tp_rank * experts_per_gpu + expert_idx
                    
                    layer0_s0.node(f'expert{expert_id}_l0_tp{tp_rank}',
                                 f'Expert {expert_id}\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                                 shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            # Expert aggregation and output
            for tp_rank in range(8):
                gpu_id = tp_rank
                layer0_s0.node(f'expert_agg_l0_tp{tp_rank}',
                             f'Expert Aggregation TP{tp_rank}\nInput: [processed_tokens from 8 experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
                
                layer0_s0.node(f'res_add2_l0_tp{tp_rank}',
                             f'Residual Add 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer0_s0.node(f'norm2_l0_tp{tp_rank}',
                             f'Layer Norm 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # =============================================================================
        # LAYER 1 in Stage 0
        # =============================================================================
        
        with stage0.subgraph(name='cluster_layer1_s0') as layer1_s0:
            layer1_s0.attr(label='Layer 1 (Stage 0)', style='rounded', color='green')
            
            for tp_rank in range(8):
                gpu_id = tp_rank
                
                layer1_s0.node(f'input_split_l1_tp{tp_rank}',
                             f'Input Split TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer1_s0.node(f'mha_qkv_l1_tp{tp_rank}',
                             f'MHA QKV TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'mha_attn_l1_tp{tp_rank}',
                             f'MHA Attention TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'mha_out_l1_tp{tp_rank}',
                             f'MHA Output TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'attn_allreduce_l1_tp{tp_rank}',
                             f'Attention All-Reduce TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer1_s0.node(f'res_add_l1_tp{tp_rank}',
                             f'Residual Add TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'norm_l1_tp{tp_rank}',
                             f'Layer Norm TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'gate_l1_tp{tp_rank}',
                             f'Gate Network TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['gate'])
                
                # 8 experts per GPU
                for expert_idx in range(experts_per_gpu):
                    expert_id = tp_rank * experts_per_gpu + expert_idx
                    
                    layer1_s0.node(f'expert{expert_id}_l1_tp{tp_rank}',
                                 f'Expert {expert_id}\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                                 shape='rectangle', style='filled', fillcolor=colors['expert'])
                
                layer1_s0.node(f'expert_agg_l1_tp{tp_rank}',
                             f'Expert Aggregation TP{tp_rank}\nInput: [processed_tokens from 8 experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
                
                layer1_s0.node(f'res_add2_l1_tp{tp_rank}',
                             f'Residual Add 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer1_s0.node(f'norm2_l1_tp{tp_rank}',
                             f'Layer Norm 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
    
    # =================================================================================
    # PIPELINE STAGE 1 (GPUs 8-15)
    # =================================================================================
    
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='rounded', color='black', bgcolor='lightyellow')
        
        # Pipeline communication between stages
        stage1.node('pipeline_comm_01', f'Pipeline Communication S0→S1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: 0-7 → 8-15',
                   shape='ellipse', style='filled', fillcolor=colors['pipeline'])
        
        # =============================================================================
        # LAYER 2 in Stage 1
        # =============================================================================
        
        with stage1.subgraph(name='cluster_layer2_s1') as layer2_s1:
            layer2_s1.attr(label='Layer 2 (Stage 1)', style='rounded', color='red')
            
            for tp_rank in range(8):
                gpu_id = tp_rank + 8
                
                layer2_s1.node(f'input_split_l2_tp{tp_rank}',
                             f'Input Split TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer2_s1.node(f'mha_qkv_l2_tp{tp_rank}',
                             f'MHA QKV TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'mha_attn_l2_tp{tp_rank}',
                             f'MHA Attention TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'mha_out_l2_tp{tp_rank}',
                             f'MHA Output TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'attn_allreduce_l2_tp{tp_rank}',
                             f'Attention All-Reduce TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer2_s1.node(f'res_add_l2_tp{tp_rank}',
                             f'Residual Add TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'norm_l2_tp{tp_rank}',
                             f'Layer Norm TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'gate_l2_tp{tp_rank}',
                             f'Gate Network TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['gate'])
                
                # 8 experts per GPU (offset by 8 for stage 1)
                for expert_idx in range(experts_per_gpu):
                    expert_id = 8 + tp_rank * experts_per_gpu + expert_idx
                    
                    layer2_s1.node(f'expert{expert_id}_l2_tp{tp_rank}',
                                 f'Expert {expert_id}\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                                 shape='rectangle', style='filled', fillcolor=colors['expert'])
                
                layer2_s1.node(f'expert_agg_l2_tp{tp_rank}',
                             f'Expert Aggregation TP{tp_rank}\nInput: [processed_tokens from 8 experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
                
                layer2_s1.node(f'res_add2_l2_tp{tp_rank}',
                             f'Residual Add 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer2_s1.node(f'norm2_l2_tp{tp_rank}',
                             f'Layer Norm 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # =============================================================================
        # LAYER 3 in Stage 1
        # =============================================================================
        
        with stage1.subgraph(name='cluster_layer3_s1') as layer3_s1:
            layer3_s1.attr(label='Layer 3 (Stage 1)', style='rounded', color='purple')
            
            for tp_rank in range(8):
                gpu_id = tp_rank + 8
                
                layer3_s1.node(f'input_split_l3_tp{tp_rank}',
                             f'Input Split TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer3_s1.node(f'mha_qkv_l3_tp{tp_rank}',
                             f'MHA QKV TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'mha_attn_l3_tp{tp_rank}',
                             f'MHA Attention TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'mha_out_l3_tp{tp_rank}',
                             f'MHA Output TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads//8}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'attn_allreduce_l3_tp{tp_rank}',
                             f'Attention All-Reduce TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={tp_hidden}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='ellipse', style='filled', fillcolor=colors['communication'])
                
                layer3_s1.node(f'res_add_l3_tp{tp_rank}',
                             f'Residual Add TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'norm_l3_tp{tp_rank}',
                             f'Layer Norm TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'gate_l3_tp{tp_rank}',
                             f'Gate Network TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['gate'])
                
                # 8 experts per GPU (offset by 8 for stage 1)
                for expert_idx in range(experts_per_gpu):
                    expert_id = 8 + tp_rank * experts_per_gpu + expert_idx
                    
                    layer3_s1.node(f'expert{expert_id}_l3_tp{tp_rank}',
                                 f'Expert {expert_id}\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                                 shape='rectangle', style='filled', fillcolor=colors['expert'])
                
                layer3_s1.node(f'expert_agg_l3_tp{tp_rank}',
                             f'Expert Aggregation TP{tp_rank}\nInput: [processed_tokens from 8 experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
                
                layer3_s1.node(f'res_add2_l3_tp{tp_rank}',
                             f'Residual Add 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
                
                layer3_s1.node(f'norm2_l3_tp{tp_rank}',
                             f'Layer Norm 2 TP{tp_rank}\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: {gpu_id}',
                             shape='rectangle', style='filled', fillcolor=colors['attention'])
            
            # Final output
            stage1.node('final_output', f'Final Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: 8-15',
                       shape='ellipse', style='filled', fillcolor=colors['output'])
    
    # =================================================================================
    # CONNECTIONS WITHIN LAYERS
    # =================================================================================
    
    # Layer 0 connections
    for tp_rank in range(8):
        dot.edge('stage0_input', f'input_split_l0_tp{tp_rank}')
        dot.edge(f'input_split_l0_tp{tp_rank}', f'mha_qkv_l0_tp{tp_rank}')
        dot.edge(f'mha_qkv_l0_tp{tp_rank}', f'mha_attn_l0_tp{tp_rank}')
        dot.edge(f'mha_attn_l0_tp{tp_rank}', f'mha_out_l0_tp{tp_rank}')
        dot.edge(f'mha_out_l0_tp{tp_rank}', f'attn_allreduce_l0_tp{tp_rank}')
        dot.edge('stage0_input', f'res_add_l0_tp{tp_rank}')
        dot.edge(f'attn_allreduce_l0_tp{tp_rank}', f'res_add_l0_tp{tp_rank}')
        dot.edge(f'res_add_l0_tp{tp_rank}', f'norm_l0_tp{tp_rank}')
        dot.edge(f'norm_l0_tp{tp_rank}', f'gate_l0_tp{tp_rank}')
        
        # Connect to 8 experts per GPU
        for expert_idx in range(experts_per_gpu):
            expert_id = tp_rank * experts_per_gpu + expert_idx
            dot.edge(f'norm_l0_tp{tp_rank}', f'expert{expert_id}_l0_tp{tp_rank}')
            dot.edge(f'expert{expert_id}_l0_tp{tp_rank}', f'expert_agg_l0_tp{tp_rank}')
        
        dot.edge(f'expert_agg_l0_tp{tp_rank}', f'res_add2_l0_tp{tp_rank}')
        dot.edge(f'norm_l0_tp{tp_rank}', f'res_add2_l0_tp{tp_rank}')
        dot.edge(f'res_add2_l0_tp{tp_rank}', f'norm2_l0_tp{tp_rank}')
    
    # Layer 1 connections
    for tp_rank in range(8):
        dot.edge(f'norm2_l0_tp{tp_rank}', f'input_split_l1_tp{tp_rank}')
        dot.edge(f'input_split_l1_tp{tp_rank}', f'mha_qkv_l1_tp{tp_rank}')
        dot.edge(f'mha_qkv_l1_tp{tp_rank}', f'mha_attn_l1_tp{tp_rank}')
        dot.edge(f'mha_attn_l1_tp{tp_rank}', f'mha_out_l1_tp{tp_rank}')
        dot.edge(f'mha_out_l1_tp{tp_rank}', f'attn_allreduce_l1_tp{tp_rank}')
        dot.edge(f'norm2_l0_tp{tp_rank}', f'res_add_l1_tp{tp_rank}')
        dot.edge(f'attn_allreduce_l1_tp{tp_rank}', f'res_add_l1_tp{tp_rank}')
        dot.edge(f'res_add_l1_tp{tp_rank}', f'norm_l1_tp{tp_rank}')
        dot.edge(f'norm_l1_tp{tp_rank}', f'gate_l1_tp{tp_rank}')
        
        for expert_idx in range(experts_per_gpu):
            expert_id = tp_rank * experts_per_gpu + expert_idx
            dot.edge(f'norm_l1_tp{tp_rank}', f'expert{expert_id}_l1_tp{tp_rank}')
            dot.edge(f'expert{expert_id}_l1_tp{tp_rank}', f'expert_agg_l1_tp{tp_rank}')
        
        dot.edge(f'expert_agg_l1_tp{tp_rank}', f'res_add2_l1_tp{tp_rank}')
        dot.edge(f'norm_l1_tp{tp_rank}', f'res_add2_l1_tp{tp_rank}')
        dot.edge(f'res_add2_l1_tp{tp_rank}', f'norm2_l1_tp{tp_rank}')
    
    # Pipeline communication
    for tp_rank in range(8):
        dot.edge(f'norm2_l1_tp{tp_rank}', 'pipeline_comm_01')
        dot.edge('pipeline_comm_01', f'input_split_l2_tp{tp_rank}')
    
    # Layer 2 connections
    for tp_rank in range(8):
        dot.edge(f'input_split_l2_tp{tp_rank}', f'mha_qkv_l2_tp{tp_rank}')
        dot.edge(f'mha_qkv_l2_tp{tp_rank}', f'mha_attn_l2_tp{tp_rank}')
        dot.edge(f'mha_attn_l2_tp{tp_rank}', f'mha_out_l2_tp{tp_rank}')
        dot.edge(f'mha_out_l2_tp{tp_rank}', f'attn_allreduce_l2_tp{tp_rank}')
        dot.edge(f'input_split_l2_tp{tp_rank}', f'res_add_l2_tp{tp_rank}')
        dot.edge(f'attn_allreduce_l2_tp{tp_rank}', f'res_add_l2_tp{tp_rank}')
        dot.edge(f'res_add_l2_tp{tp_rank}', f'norm_l2_tp{tp_rank}')
        dot.edge(f'norm_l2_tp{tp_rank}', f'gate_l2_tp{tp_rank}')
        
        for expert_idx in range(experts_per_gpu):
            expert_id = 8 + tp_rank * experts_per_gpu + expert_idx
            dot.edge(f'norm_l2_tp{tp_rank}', f'expert{expert_id}_l2_tp{tp_rank}')
            dot.edge(f'expert{expert_id}_l2_tp{tp_rank}', f'expert_agg_l2_tp{tp_rank}')
        
        dot.edge(f'expert_agg_l2_tp{tp_rank}', f'res_add2_l2_tp{tp_rank}')
        dot.edge(f'norm_l2_tp{tp_rank}', f'res_add2_l2_tp{tp_rank}')
        dot.edge(f'res_add2_l2_tp{tp_rank}', f'norm2_l2_tp{tp_rank}')
    
    # Layer 3 connections
    for tp_rank in range(8):
        dot.edge(f'norm2_l2_tp{tp_rank}', f'input_split_l3_tp{tp_rank}')
        dot.edge(f'input_split_l3_tp{tp_rank}', f'mha_qkv_l3_tp{tp_rank}')
        dot.edge(f'mha_qkv_l3_tp{tp_rank}', f'mha_attn_l3_tp{tp_rank}')
        dot.edge(f'mha_attn_l3_tp{tp_rank}', f'mha_out_l3_tp{tp_rank}')
        dot.edge(f'mha_out_l3_tp{tp_rank}', f'attn_allreduce_l3_tp{tp_rank}')
        dot.edge(f'input_split_l3_tp{tp_rank}', f'res_add_l3_tp{tp_rank}')
        dot.edge(f'attn_allreduce_l3_tp{tp_rank}', f'res_add_l3_tp{tp_rank}')
        dot.edge(f'res_add_l3_tp{tp_rank}', f'norm_l3_tp{tp_rank}')
        dot.edge(f'norm_l3_tp{tp_rank}', f'gate_l3_tp{tp_rank}')
        
        for expert_idx in range(experts_per_gpu):
            expert_id = 8 + tp_rank * experts_per_gpu + expert_idx
            dot.edge(f'norm_l3_tp{tp_rank}', f'expert{expert_id}_l3_tp{tp_rank}')
            dot.edge(f'expert{expert_id}_l3_tp{tp_rank}', f'expert_agg_l3_tp{tp_rank}')
        
        dot.edge(f'expert_agg_l3_tp{tp_rank}', f'res_add2_l3_tp{tp_rank}')
        dot.edge(f'norm_l3_tp{tp_rank}', f'res_add2_l3_tp{tp_rank}')
        dot.edge(f'res_add2_l3_tp{tp_rank}', f'norm2_l3_tp{tp_rank}')
        dot.edge(f'norm2_l3_tp{tp_rank}', 'final_output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_moe_dag()
    
    # Save DOT file
    dag.save(directory='./outputs/2025-10-13-21-03-41', filename='baseline_moe_tp8_pp2_dag.dot')
    
    # Render to SVG
    dag.render(directory='./outputs/2025-10-13-21-03-41', filename='baseline_moe_tp8_pp2_dag', format='svg', cleanup=True)
    
    print("Baseline MoE DAG generated successfully!")
    print("Files saved:")
    print("- baseline_moe_tp8_pp2_dag.dot")
    print("- baseline_moe_tp8_pp2_dag.svg")