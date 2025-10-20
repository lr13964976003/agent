#!/usr/bin/env python3
"""
Generate complete DAG for Hybrid TP+PP baseline
16 GPUs total: 8 GPUs per stage, 2 pipeline stages
4-layer MoE transformer with 16 experts per layer
"""

import graphviz
from typing import Dict, List

def create_baseline_dag():
    """Create complete DAG for Hybrid TP+PP baseline"""
    
    # Create directed graph
    dot = graphviz.Digraph('baseline_hybrid_tp_pp_dag', comment='Baseline Hybrid TP=8 PP=2 Complete Model DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse')  # Input/Output
    dot.attr('node', style='filled')
    
    # Global dimensions
    batch_size = 1024
    seq_len = 2048
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    vocab_size = 50265
    expert_hidden = 16384
    
    # GPU assignments
    stage0_gpus = list(range(8))   # GPUs 0-7
    stage1_gpus = list(range(8, 16))  # GPUs 8-15
    
    # Colors for GPU identification
    gpu_colors = {}
    for gpu in range(16):
        if gpu < 8:
            # Stage 0 - shades of orange
            gpu_colors[gpu] = f'#ff{(gpu*20):02x}{128:02x}'
        else:
            # Stage 1 - shades of purple
            gpu_colors[gpu] = f'#{(gpu-8)*30:02x}{128:02x}ff'
    
    # Model Input
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
             shape='ellipse', fillcolor='#e6f3ff')
    
    # Embedding layer (on GPU 0)
    dot.node('embedding', 
             f'Embedding\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0',
             shape='rectangle', fillcolor=gpu_colors[0])
    
    # Stage 0 (Layers 0-1)
    with dot.subgraph(name='cluster_stage_0') as stage0:
        stage0.attr(label='Stage 0 (GPUs 0-7)', style='dotted', color='orange')
        
        for layer_idx in [0, 1]:
            with stage0.subgraph(name=f'cluster_stage0_layer_{layer_idx}') as layer:
                layer.attr(label=f'Layer {layer_idx}', style='dashed', color='orange')
                
                # === ATTENTION BLOCK (TP=8) ===
                
                # Layer Normalization (sharded across 8 GPUs)
                ln1_name = f'stage0_ln1_layer_{layer_idx}'
                layer.node(ln1_name,
                          f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[0])
                
                # Query-Key-Value Projections across 8 GPUs
                qkv_nodes = []
                for gpu_idx in stage0_gpus:
                    heads_per_gpu = num_heads // 8  # 4 heads per GPU
                    qkv_name = f'stage0_qkv_layer_{layer_idx}_gpu_{gpu_idx}'
                    qkv_nodes.append(qkv_name)
                    layer.node(qkv_name,
                              f'QKV Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nK:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nV:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                              shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # Attention computation nodes for each GPU
                attention_nodes = []
                for gpu_idx in stage0_gpus:
                    heads_per_gpu = num_heads // 8
                    attn_name = f'stage0_attention_layer_{layer_idx}_gpu_{gpu_idx}'
                    attention_nodes.append(attn_name)
                    layer.node(attn_name,
                              f'Multi-Head Attention\\nInput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nK_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nV_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                              shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # All-reduce for attention outputs
                attn_reduce_name = f'stage0_attention_reduce_layer_{layer_idx}'
                layer.node(attn_reduce_name,
                          f'Attention All-Reduce\\nInput: [8× batch_size={batch_size}, seq_len={seq_len}, heads={4}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffcccc')
                
                # Output projection (TP=8)
                out_proj_name = f'stage0_out_proj_layer_{layer_idx}'
                layer.node(out_proj_name,
                          f'Output Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[0])
                
                # Residual connection 1
                residual1_name = f'stage0_residual1_layer_{layer_idx}'
                layer.node(residual1_name,
                          f'Residual Add 1\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffffcc')
                
                # === MOE BLOCK (TP=8) ===
                
                # Layer Normalization 2 (sharded)
                ln2_name = f'stage0_ln2_layer_{layer_idx}'
                layer.node(ln2_name,
                          f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[0])
                
                # Gate network (sharded across 8 GPUs)
                gate_name = f'stage0_gate_layer_{layer_idx}'
                layer.node(gate_name,
                          f'Gating Network\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={16}]\\nGPU: 0-7 (sharded)',
                          shape='parallelogram', fillcolor='#ccffcc', style='dashed')
                
                # Expert computation across 8 GPUs (2 experts per GPU)
                expert_nodes = []
                for gpu_idx in stage0_gpus:
                    experts_per_gpu = 2
                    for expert_local_idx in range(experts_per_gpu):
                        expert_id = gpu_idx * experts_per_gpu + expert_local_idx
                        expert_name = f'stage0_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                        expert_nodes.append(expert_name)
                        layer.node(expert_name,
                                  f'Expert{expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nGPU: {gpu_idx}',
                                  shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # Expert aggregation (all-reduce)
                expert_agg_name = f'stage0_expert_agg_layer_{layer_idx}'
                layer.node(expert_agg_name,
                          f'Expert Aggregation\\nInput: [8× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffcccc')
                
                # Residual connection 2
                residual2_name = f'stage0_residual2_layer_{layer_idx}'
                layer.node(residual2_name,
                          f'Residual Add 2\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 0-7 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffffcc')
    
    # Stage 1 (Layers 2-3)
    with dot.subgraph(name='cluster_stage_1') as stage1:
        stage1.attr(label='Stage 1 (GPUs 8-15)', style='dotted', color='purple')
        
        for layer_idx in [2, 3]:
            with stage1.subgraph(name=f'cluster_stage1_layer_{layer_idx}') as layer:
                layer.attr(label=f'Layer {layer_idx}', style='dashed', color='purple')
                
                # === ATTENTION BLOCK (TP=8) ===
                
                # Layer Normalization (sharded across 8 GPUs)
                ln1_name = f'stage1_ln1_layer_{layer_idx}'
                layer.node(ln1_name,
                          f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[8])
                
                # Query-Key-Value Projections across 8 GPUs
                qkv_nodes = []
                for gpu_idx in stage1_gpus:
                    heads_per_gpu = num_heads // 8
                    qkv_name = f'stage1_qkv_layer_{layer_idx}_gpu_{gpu_idx}'
                    qkv_nodes.append(qkv_name)
                    layer.node(qkv_name,
                              f'QKV Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nK:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nV:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                              shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # Attention computation nodes for each GPU
                attention_nodes = []
                for gpu_idx in stage1_gpus:
                    heads_per_gpu = num_heads // 8
                    attn_name = f'stage1_attention_layer_{layer_idx}_gpu_{gpu_idx}'
                    attention_nodes.append(attn_name)
                    layer.node(attn_name,
                              f'Multi-Head Attention\\nInput: Q:[batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nK_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nV_all:[batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_gpu}, d_k={head_dim}]\\nGPU: {gpu_idx}',
                              shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # All-reduce for attention outputs
                attn_reduce_name = f'stage1_attention_reduce_layer_{layer_idx}'
                layer.node(attn_reduce_name,
                          f'Attention All-Reduce\\nInput: [8× batch_size={batch_size}, seq_len={seq_len}, heads={4}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffcccc')
                
                # Output projection (TP=8)
                out_proj_name = f'stage1_out_proj_layer_{layer_idx}'
                layer.node(out_proj_name,
                          f'Output Projection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[8])
                
                # Residual connection 1
                residual1_name = f'stage1_residual1_layer_{layer_idx}'
                layer.node(residual1_name,
                          f'Residual Add 1\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffffcc')
                
                # === MOE BLOCK (TP=8) ===
                
                # Layer Normalization 2 (sharded)
                ln2_name = f'stage1_ln2_layer_{layer_idx}'
                layer.node(ln2_name,
                          f'LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (sharded)',
                          shape='rectangle', fillcolor=gpu_colors[8])
                
                # Gate network (sharded across 8 GPUs)
                gate_name = f'stage1_gate_layer_{layer_idx}'
                layer.node(gate_name,
                          f'Gating Network\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={16}]\\nGPU: 8-15 (sharded)',
                          shape='parallelogram', fillcolor='#ccffcc', style='dashed')
                
                # Expert computation across 8 GPUs (2 experts per GPU)
                expert_nodes = []
                for gpu_idx in stage1_gpus:
                    experts_per_gpu = 2
                    for expert_local_idx in range(experts_per_gpu):
                        expert_id = (gpu_idx - 8) * experts_per_gpu + expert_local_idx
                        expert_name = f'stage1_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                        expert_nodes.append(expert_name)
                        layer.node(expert_name,
                                  f'Expert{expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nGPU: {gpu_idx}',
                                  shape='rectangle', fillcolor=gpu_colors[gpu_idx])
                
                # Expert aggregation (all-reduce)
                expert_agg_name = f'stage1_expert_agg_layer_{layer_idx}'
                layer.node(expert_agg_name,
                          f'Expert Aggregation\\nInput: [8× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}/8]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffcccc')
                
                # Residual connection 2
                residual2_name = f'stage1_residual2_layer_{layer_idx}'
                layer.node(residual2_name,
                          f'Residual Add 2\\nInput: [2× batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15 (all-reduce)',
                          shape='parallelogram', fillcolor='#ffffcc')
    
    # Final LayerNorm and Output
    dot.node('final_layernorm',
             f'Final LayerNorm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: 8-15',
             shape='rectangle', fillcolor=gpu_colors[8])
    
    dot.node('output',
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: 8-15',
             shape='ellipse', fillcolor='#e6f3ff')
    
    # === CONNECTIONS ===
    
    # Input to embedding
    dot.edge('input', 'embedding')
    
    # Pipeline stage 0 (Layers 0-1)
    prev_node = 'embedding'
    
    for layer_idx in [0, 1]:
        ln1 = f'stage0_ln1_layer_{layer_idx}'
        
        # LayerNorm -> QKV projections
        dot.edge(prev_node, ln1)
        
        for gpu_idx in stage0_gpus:
            qkv_name = f'stage0_qkv_layer_{layer_idx}_gpu_{gpu_idx}'
            attn_name = f'stage0_attention_layer_{layer_idx}_gpu_{gpu_idx}'
            
            dot.edge(ln1, qkv_name)
            dot.edge(qkv_name, attn_name)
            
            # Cross-GPU communication for K,V
            for other_gpu in stage0_gpus:
                if other_gpu != gpu_idx:
                    other_qkv = f'stage0_qkv_layer_{layer_idx}_gpu_{other_gpu}'
                    dot.edge(other_qkv, attn_name, style='dashed', color='blue')
        
        # Attention all-reduce
        attn_reduce = f'stage0_attention_reduce_layer_{layer_idx}'
        for gpu_idx in stage0_gpus:
            attn_name = f'stage0_attention_layer_{layer_idx}_gpu_{gpu_idx}'
            dot.edge(attn_name, attn_reduce)
        
        # Output projection and residual
        out_proj = f'stage0_out_proj_layer_{layer_idx}'
        dot.edge(attn_reduce, out_proj)
        
        residual1 = f'stage0_residual1_layer_{layer_idx}'
        dot.edge(out_proj, residual1)
        dot.edge(prev_node, residual1)
        
        # MoE block
        ln2 = f'stage0_ln2_layer_{layer_idx}'
        dot.edge(residual1, ln2)
        
        gate = f'stage0_gate_layer_{layer_idx}'
        dot.edge(ln2, gate, style='dashed')
        
        # Expert computation
        for gpu_idx in stage0_gpus:
            for expert_local_idx in range(2):
                expert_id = gpu_idx * 2 + expert_local_idx
                expert_name = f'stage0_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                dot.edge(gate, expert_name, style='dashed', color='red')
                dot.edge(ln2, expert_name)
        
        # Expert aggregation
        expert_agg = f'stage0_expert_agg_layer_{layer_idx}'
        for gpu_idx in stage0_gpus:
            for expert_local_idx in range(2):
                expert_id = gpu_idx * 2 + expert_local_idx
                expert_name = f'stage0_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                dot.edge(expert_name, expert_agg)
        
        residual2 = f'stage0_residual2_layer_{layer_idx}'
        dot.edge(expert_agg, residual2)
        dot.edge(residual1, residual2)
        
        prev_node = residual2
    
    # Pipeline communication between stages
    dot.edge(prev_node, 'stage1_ln1_layer_2', label='Send/Recv\nStage 0 → 1', style='bold', color='purple')
    
    # Pipeline stage 1 (Layers 2-3)
    prev_stage1_node = f'stage1_ln1_layer_2'
    
    for layer_idx in [2, 3]:
        ln1 = f'stage1_ln1_layer_{layer_idx}'
        
        if layer_idx == 2:
            # First layer of stage 1 already connected from stage 0
            pass
        else:
            dot.edge(prev_stage1_node, ln1)
        
        # Repeat same structure as stage 0
        for gpu_idx in stage1_gpus:
            qkv_name = f'stage1_qkv_layer_{layer_idx}_gpu_{gpu_idx}'
            attn_name = f'stage1_attention_layer_{layer_idx}_gpu_{gpu_idx}'
            
            dot.edge(ln1, qkv_name)
            dot.edge(qkv_name, attn_name)
            
            # Cross-GPU communication for K,V
            for other_gpu in stage1_gpus:
                if other_gpu != gpu_idx:
                    other_qkv = f'stage1_qkv_layer_{layer_idx}_gpu_{other_gpu}'
                    dot.edge(other_qkv, attn_name, style='dashed', color='blue')
        
        # Attention all-reduce
        attn_reduce = f'stage1_attention_reduce_layer_{layer_idx}'
        for gpu_idx in stage1_gpus:
            attn_name = f'stage1_attention_layer_{layer_idx}_gpu_{gpu_idx}'
            dot.edge(attn_name, attn_reduce)
        
        # Output projection and residual
        out_proj = f'stage1_out_proj_layer_{layer_idx}'
        dot.edge(attn_reduce, out_proj)
        
        residual1 = f'stage1_residual1_layer_{layer_idx}'
        dot.edge(out_proj, residual1)
        if layer_idx == 2:
            dot.edge('stage1_ln1_layer_2', residual1)
        else:
            dot.edge(prev_stage1_node, residual1)
        
        # MoE block
        ln2 = f'stage1_ln2_layer_{layer_idx}'
        dot.edge(residual1, ln2)
        
        gate = f'stage1_gate_layer_{layer_idx}'
        dot.edge(ln2, gate, style='dashed')
        
        # Expert computation
        for gpu_idx in stage1_gpus:
            for expert_local_idx in range(2):
                expert_id = (gpu_idx - 8) * 2 + expert_local_idx
                expert_name = f'stage1_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                dot.edge(gate, expert_name, style='dashed', color='red')
                dot.edge(ln2, expert_name)
        
        # Expert aggregation
        expert_agg = f'stage1_expert_agg_layer_{layer_idx}'
        for gpu_idx in stage1_gpus:
            for expert_local_idx in range(2):
                expert_id = (gpu_idx - 8) * 2 + expert_local_idx
                expert_name = f'stage1_expert_layer_{layer_idx}_expert_{expert_id}_gpu_{gpu_idx}'
                dot.edge(expert_name, expert_agg)
        
        residual2 = f'stage1_residual2_layer_{layer_idx}'
        dot.edge(expert_agg, residual2)
        dot.edge(residual1, residual2)
        
        prev_stage1_node = residual2
    
    # Final connections
    dot.edge(prev_stage1_node, 'final_layernorm')
    dot.edge('final_layernorm', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    
    # Save DOT file
    dag.save('../outputs/2025-10-19-23-59-31/baseline_hybrid_tp_pp_dag.dot')
    
    # Generate SVG
    dag.render('../outputs/2025-10-19-23-59-31/baseline_hybrid_tp_pp_dag', format='svg', cleanup=True)
    
    print("Baseline Hybrid TP+PP DAG generated successfully!")