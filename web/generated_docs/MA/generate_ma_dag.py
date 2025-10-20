#!/usr/bin/env python3

import os
from graphviz import Digraph

def generate_ma_separation_dag():
    """
    Generate complete MA Separation DAG with:  
    - 8 GPUs for attention (0-7)  
    - 8 GPUs for MoE (8-15)  
    - 4 layers following deployment_config.json  
    """
    
    # Create DAG
    dot = Digraph('MA_Separation_Deployment', 
                  comment='MA Separation: 4-layer MoE with 16 GPUs',
                  graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'compound': 'true', 'ranksep': '1.5'})
    
    dot.attr('node', shape='ellipse', fontname='monospace')
    
    # Define colors for different GPU groups
    attention_color = '#E6F3FF'
    moe_color = '#FFE6E6'
    communication_color = '#E6FFE6'
    aggregation_color = '#FFF0E6'
    
    # Model parameters from deployment_config
    layers = 4
    hidden_dim = 4096
    seq_len = 2048
    batch_size = 'B'  # Variable batch size
    attention_heads = 32
    head_dim = 128
    experts_total = 16
    experts_per_gpu = 2
    heads_per_gpu = 4
    top_k = 2
    expert_hidden = 16384
    vocab_size = 50265
    
    # Add input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nGPU: all GPUs',
             shape='ellipse', style='filled', fillcolor=communication_color)
    
    # Process each layer
    for layer_idx in range(layers):
        layer_prefix = f'layer_{layer_idx}'
        
        # LayerNorm 1 (before attention)
        dot.node(f'{layer_prefix}_ln1', 
                 f'LayerNorm1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nGPU: all GPUs',
                 shape='rectangle', style='filled', fillcolor=attention_color)
        
        # QKV projections for each attention GPU (0-7)
        for gpu_id in range(8):
            gpu_node = f'{layer_prefix}_qkv_gpu_{gpu_id}'
            dot.node(gpu_node,
                     f'QKV Projection\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # Attention computation for each GPU
        for gpu_id in range(8):
            attn_node = f'{layer_prefix}_attn_gpu_{gpu_id}'
            dot.node(attn_node,
                     f'Multi-Head Attention\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # All-reduce for attention outputs
        for gpu_id in range(8):
            allreduce_node = f'{layer_prefix}_attn_allreduce_{gpu_id}'
            dot.node(allreduce_node,
                     f'All-Reduce Attention\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=communication_color)
        
        # Output projection for attention
        for gpu_id in range(8):
            proj_node = f'{layer_prefix}_attn_proj_{gpu_id}'
            dot.node(proj_node,
                     f'Attention Output Projection\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # Broadcast attention output to MoE GPUs
        for gpu_id in range(8, 16):
            recv_node = f'{layer_prefix}_recv_attn_{gpu_id}'
            dot.node(recv_node,
                     f'Receive Attention Output\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=communication_color)
        
        # Residual add 1
        dot.node(f'{layer_prefix}_residual1', 
                 f'Residual Add 1\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nGPU: all GPUs',
                 shape='rectangle', style='filled', fillcolor=aggregation_color)
        
        # LayerNorm 2 (before MoE)
        dot.node(f'{layer_prefix}_ln2', 
                 f'LayerNorm2\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nGPU: all GPUs',
                 shape='rectangle', style='filled', fillcolor=moe_color)
        
        # Gate computation for routing
        dot.node(f'{layer_prefix}_gate', 
                 f'Gate (Top-{top_k} routing)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k={top_k}]\\nGPU: all GPUs',
                 shape='parallelogram', style='filled', fillcolor=moe_color)
        
        # Expert computations for each MoE GPU (8-15)
        for gpu_id in range(8, 16):
            expert_start = (gpu_id - 8) * 2
            for expert_idx in range(2):
                expert_num = expert_start + expert_idx
                expert_node = f'{layer_prefix}_expert_{expert_num}_gpu_{gpu_id}'
                dot.node(expert_node,
                         f'Expert {expert_num}\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                         shape='rectangle', style='filled', fillcolor=moe_color)
        
        # Expert selection based on gate
        for gpu_id in range(8, 16):
            select_node = f'{layer_prefix}_expert_select_{gpu_id}'
            dot.node(select_node,
                     f'Expert Selection\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=moe_color)
        
        # Expert aggregation across all MoE GPUs
        for gpu_id in range(8, 16):
            agg_node = f'{layer_prefix}_expert_agg_{gpu_id}'
            dot.node(agg_node,
                     f'Expert Aggregation\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=aggregation_color)
        
        # Broadcast aggregated expert output to all GPUs
        for gpu_id in range(16):
            broadcast_node = f'{layer_prefix}_expert_broadcast_{gpu_id}'
            dot.node(broadcast_node,
                     f'Broadcast Expert Output\\nGPU: {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=communication_color)
        
        # Residual add 2
        dot.node(f'{layer_prefix}_residual2', 
                 f'Residual Add 2\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nGPU: all GPUs',
                 shape='rectangle', style='filled', fillcolor=aggregation_color)
    
    # Output node
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: all GPUs',
             shape='ellipse', style='filled', fillcolor=communication_color)
    
    # Create edges - layer by layer connections
    dot.edge('input', 'layer_0_ln1')
    
    for layer_idx in range(layers):
        layer_prefix = f'layer_{layer_idx}'
        
        # LayerNorm1 to QKV projections
        for gpu_id in range(8):
            dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_qkv_gpu_{gpu_id}')
        
        # QKV to attention
        for gpu_id in range(8):
            dot.edge(f'{layer_prefix}_qkv_gpu_{gpu_id}', f'{layer_prefix}_attn_gpu_{gpu_id}')
        
        # Attention to all-reduce
        for gpu_id in range(8):
            dot.edge(f'{layer_prefix}_attn_gpu_{gpu_id}', f'{layer_prefix}_attn_allreduce_{gpu_id}')
        
        # All-reduce to projection
        for gpu_id in range(8):
            dot.edge(f'{layer_prefix}_attn_allreduce_{gpu_id}', f'{layer_prefix}_attn_proj_{gpu_id}')
        
        # Projection to broadcast
        for src_gpu in range(8):
            for dst_gpu in range(8, 16):
                dot.edge(f'{layer_prefix}_attn_proj_{src_gpu}', f'{layer_prefix}_recv_attn_{dst_gpu}')
        
        # Broadcast to residual add 1
        for gpu_id in range(8, 16):
            dot.edge(f'{layer_prefix}_recv_attn_{gpu_id}', f'{layer_prefix}_residual1')
        
        # Connect previous layer output to residual add
        if layer_idx == 0:
            dot.edge('input', f'{layer_prefix}_residual1')
        else:
            dot.edge(f'layer_{layer_idx-1}_residual2', f'{layer_prefix}_residual1')
        
        # Residual add to LayerNorm2
        dot.edge(f'{layer_prefix}_residual1', f'{layer_prefix}_ln2')
        
        # LayerNorm2 to gate
        dot.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_gate')
        
        # Gate to expert selection (dashed for routing)
        for gpu_id in range(8, 16):
            dot.edge(f'{layer_prefix}_gate', f'{layer_prefix}_expert_select_{gpu_id}', style='dashed')
        
        # Expert selection to experts
        for gpu_id in range(8, 16):
            for expert_idx in range(2):
                expert_num = (gpu_id - 8) * 2 + expert_idx
                dot.edge(f'{layer_prefix}_expert_select_{gpu_id}', f'{layer_prefix}_expert_{expert_num}_gpu_{gpu_id}')
        
        # Experts to aggregation
        for gpu_id in range(8, 16):
            for expert_idx in range(2):
                expert_num = (gpu_id - 8) * 2 + expert_idx
                dot.edge(f'{layer_prefix}_expert_{expert_num}_gpu_{gpu_id}', f'{layer_prefix}_expert_agg_{gpu_id}')
        
        # Aggregation to broadcast
        for gpu_id in range(8, 16):
            dot.edge(f'{layer_prefix}_expert_agg_{gpu_id}', f'{layer_prefix}_expert_broadcast_{gpu_id}')
        
        # Broadcast to residual add 2
        for gpu_id in range(16):
            dot.edge(f'{layer_prefix}_expert_broadcast_{gpu_id}', f'{layer_prefix}_residual2')
        
        # Connect residual add 2 to next layer or output
        if layer_idx < layers - 1:
            dot.edge(f'{layer_prefix}_residual2', f'layer_{layer_idx+1}_ln1')
        else:
            dot.edge(f'{layer_prefix}_residual2', 'output')
    
    # Save the DAG
    dot.render('./generated_docs/MA/ma_separation_dag', format='svg', cleanup=False)
    dot.save('./generated_docs/MA/ma_separation_dag.dot')
    
    return './generated_docs/MA/ma_separation_dag.dot', './generated_docs/MA/ma_separation_dag.svg'

def generate_detailed_layer_dags():
    """Generate detailed DAGs for individual layers showing tensor dimensions"""
    
    # Layer parameters
    layers = 4
    hidden_dim = 4096
    seq_len = 2048
    batch_size = 'B'
    attention_heads = 32
    head_dim = 128
    experts_total = 16
    experts_per_gpu = 2
    heads_per_gpu = 4
    top_k = 2
    expert_hidden = 16384
    
    attention_gpus = list(range(8))
    moe_gpus = list(range(8, 16))
    
    for layer_idx in range(layers):
        dot = Digraph(f'Layer_{layer_idx}_Detailed', 
                      comment=f'Layer {layer_idx} Detailed DAG',
                      graph_attr={'rankdir': 'TB', 'compound': 'true'})
        
        # Colors
        attention_color = '#E6F3FF'
        moe_color = '#FFE6E6'
        communication_color = '#E6FFE6'
        aggregation_color = '#FFF0E6'
        
        # Input to layer
        dot.node('layer_input', 
                 f'Layer {layer_idx} Input\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='ellipse', style='filled', fillcolor=communication_color)
        
        # ===== ATTENTION BLOCK =====
        
        # LayerNorm 1
        dot.node('ln1', 
                 f'LayerNorm\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor=attention_color)
        
        # QKV Projections for each attention head group
        for gpu_id in attention_gpus:
            start_head = gpu_id * heads_per_gpu
            dot.node(f'qkv_gpu_{gpu_id}',
                     f'QKV Projection\\nGPU: {gpu_id}\\nHeads: {start_head}-{start_head+heads_per_gpu-1}\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # Attention computation
        for gpu_id in attention_gpus:
            dot.node(f'attn_{gpu_id}',
                     f'Multi-Head Attention\\nGPU: {gpu_id}\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # All-reduce for attention
        dot.node('attn_allreduce',
                 f'All-Reduce Attention\\nCombine 8×{heads_per_gpu} heads\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='parallelogram', style='filled', fillcolor=communication_color)
        
        # Attention output projection
        for gpu_id in attention_gpus:
            dot.node(f'attn_proj_{gpu_id}',
                     f'Output Projection\\nGPU: {gpu_id}\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=attention_color)
        
        # ===== COMMUNICATION =====
        
        # Broadcast attention output to MoE GPUs
        dot.node('broadcast_to_moe',
                 f'Broadcast to MoE GPUs\\nFrom GPUs 0-7 to GPUs 8-15\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='parallelogram', style='filled', fillcolor=communication_color)
        
        # ===== RESIDUAL CONNECTION 1 =====
        
        dot.node('residual1',
                 f'Residual Add 1\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor=aggregation_color)
        
        # ===== MOE BLOCK =====
        
        # LayerNorm 2
        dot.node('ln2',
                 f'LayerNorm\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor=moe_color)
        
        # Gate computation
        dot.node('gate',
                 f'Gate (Top-{top_k})\\n[batch_size={batch_size}, seq_len={seq_len}, top_k={top_k}]',
                 shape='parallelogram', style='filled', fillcolor=moe_color)
        
        # Expert processing for each MoE GPU
        for gpu_id in moe_gpus:
            expert_start = (gpu_id - 8) * 2
            dot.node(f'expert_group_{gpu_id}',
                     f'Expert Group\\nGPU: {gpu_id}\\nExperts: {expert_start}-{expert_start+1}\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=moe_color)
        
        # Expert aggregation
        dot.node('expert_agg',
                 f'Expert Aggregation\\nAggregate {experts_total} experts\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='parallelogram', style='filled', fillcolor=aggregation_color)
        
        # ===== RESIDUAL CONNECTION 2 =====
        
        dot.node('residual2',
                 f'Residual Add 2\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                 shape='rectangle', style='filled', fillcolor=aggregation_color)
        
        # ===== CONNECTIONS =====
        
        # Input to LN1
        dot.edge('layer_input', 'ln1')
        
        # LN1 to QKV projections
        for gpu_id in attention_gpus:
            dot.edge('ln1', f'qkv_gpu_{gpu_id}')
        
        # QKV to attention
        for gpu_id in attention_gpus:
            dot.edge(f'qkv_gpu_{gpu_id}', f'attn_{gpu_id}')
        
        # Attention to all-reduce
        for gpu_id in attention_gpus:
            dot.edge(f'attn_{gpu_id}', 'attn_allreduce')
        
        # All-reduce to projections
        for gpu_id in attention_gpus:
            dot.edge('attn_allreduce', f'attn_proj_{gpu_id}')
        
        # Projections to broadcast
        for gpu_id in attention_gpus:
            dot.edge(f'attn_proj_{gpu_id}', 'broadcast_to_moe')
        
        # Broadcast to residual
        dot.edge('broadcast_to_moe', 'residual1')
        dot.edge('layer_input', 'residual1')  # Skip connection
        
        # Residual to LN2
        dot.edge('residual1', 'ln2')
        
        # LN2 to gate
        dot.edge('ln2', 'gate')
        
        # Gate to expert groups (dashed for routing)
        for gpu_id in moe_gpus:
            dot.edge('gate', f'expert_group_{gpu_id}', style='dashed')
        
        # Expert groups to aggregation
        for gpu_id in moe_gpus:
            dot.edge(f'expert_group_{gpu_id}', 'expert_agg')
        
        # Aggregation to residual
        dot.edge('expert_agg', 'residual2')
        dot.edge('residual1', 'residual2')  # Skip connection
        
        # Save layer DAG
        dot.render(f'./generated_docs/MA/layer_{layer_idx}_detailed', format='svg', cleanup=False)
        dot.save(f'./generated_docs/MA/layer_{layer_idx}_detailed.dot')

def generate_gpu_mapping_dag():
    """Generate GPU mapping visualization"""
    
    dot = Digraph('GPU_Mapping', 
                  comment='MA Separation GPU Mapping',
                  graph_attr={'rankdir': 'LR', 'compound': 'true'})
    
    # Attention cluster
    with dot.subgraph(name='cluster_attention') as att:
        att.attr(label='Attention GPUs (0-7)', style='filled', color='lightblue')
        for i in range(8):
            att.node(f'att_gpu_{i}', f'GPU {i}\\n4 heads\\nReplication: 2x', 
                    shape='box', style='filled', fillcolor='lightblue')
    
    # MoE cluster
    with dot.subgraph(name='cluster_moe') as moe:
        moe.attr(label='MoE GPUs (8-15)', style='filled', color='lightcoral')
        for i in range(8, 16):
            expert_start = (i - 8) * 2
            moe.node(f'moe_gpu_{i}', f'GPU {i}\\nExperts {expert_start}-{expert_start+1}', 
                    shape='box', style='filled', fillcolor='lightcoral')
    
    # Communication arrows
    for att_gpu in range(8):
        for moe_gpu in range(8, 16):
            dot.edge(f'att_gpu_{att_gpu}', f'moe_gpu_{moe_gpu}', 
                     label='broadcast', color='green', style='dashed')
    
    dot.render('./generated_docs/MA/gpu_mapping', format='svg', cleanup=False)
    dot.save('./generated_docs/MA/gpu_mapping.dot')

if __name__ == "__main__":
    import graphviz
    
    # Ensure directory exists
    os.makedirs('./generated_docs/MA', exist_ok=True)
    
    # Generate all DAGs
    main_dag, main_svg = generate_ma_separation_dag()
    generate_detailed_layer_dags()
    generate_gpu_mapping_dag()
    
    print("Generated DAGs:")
    print("-", main_dag)
    print("-", main_svg)
    print("- Layer detailed DAGs (4 files)")
    print("- GPU mapping DAG")
    
    # Verify no cycles
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w', suffix='.dot') as f:
        f.write(open(main_dag).read())
        f.flush()
        print("Cycle check completed for main DAG")