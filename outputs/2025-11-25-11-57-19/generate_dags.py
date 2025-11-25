#!/usr/bin/env python3
"""
Generate DAGs for baseline (TP=8, PP=2) and proposed (EP=16) MoE deployments
"""

import os
from graphviz import Digraph

def generate_baseline_dag():
    """Generate baseline DAG with TP=8, PP=2"""
    dot = Digraph(name='baseline_moe_dag', 
                  comment='Baseline MoE with TP=8, PP=2, colocated experts')
    dot.attr(rankdir='TB', splines='ortho', ranksep='2.0', nodesep='0.8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal')
    
    # Global configuration
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    ffn_hidden = 16384
    
    # Input
    dot.node('input', f'Total Input\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (layers 0-7)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0\\nLayers 0-7\\nGPUs: 0-7', style='dashed', color='red')
        
        for layer in range(8):
            layer_name = f'layer_{layer}'
            
            # Multi-Head Attention
            mha_name = f'{layer_name}_mha'
            c.node(mha_name, f'MHA Layer {layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 0-7 (TP=8)')
            
            # MHA components
            qkv_name = f'{layer_name}_qkv'
            attn_name = f'{layer_name}_attn'
            proj_name = f'{layer_name}_proj'
            add_norm_name = f'{layer_name}_add_norm1'
            
            c.node(qkv_name, f'QKV Linear\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 0-7 (TP=8)')
            c.node(attn_name, f'Attention\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 0-7 (TP=8)')
            c.node(proj_name, f'Projection\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 0-7 (TP=8)')
            c.node(add_norm_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 0-7 (TP=8)')
            
            # MoE FFN
            moe_name = f'{layer_name}_moe'
            gate_name = f'{layer_name}_gate'
            expert_prefix = f'{layer_name}_expert'
            
            c.node(gate_name, f'Gating\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, topk=2]\\nGPU: 0-7', shape='parallelogram', fillcolor='yellow')
            
            # Create expert nodes (2 experts per GPU)
            for gpu in range(8):
                for expert_id in range(2):
                    expert_idx = gpu * 2 + expert_id
                    expert_name = f'{expert_prefix}_{expert_idx}'
                    c.node(expert_name, f'Expert {expert_idx}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: {gpu}')
            
            moe_agg_name = f'{layer_name}_moe_agg'
            c.node(moe_agg_name, f'MoE Aggregation\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 0-7 (TP=8)')
            
            add_norm2_name = f'{layer_name}_add_norm2'
            c.node(add_norm2_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 0-7 (TP=8)')
            
            # Connect MHA components
            if layer == 0:
                dot.edge('input', qkv_name)
            else:
                dot.edge(f'layer_{layer-1}_add_norm2', qkv_name)
            
            dot.edge(qkv_name, attn_name)
            dot.edge(attn_name, proj_name)
            dot.edge(proj_name, add_norm_name)
            if layer == 0:
                dot.edge('input', add_norm_name, style='dashed', label='residual')
            else:
                dot.edge(f'layer_{layer-1}_add_norm2', add_norm_name, style='dashed', label='residual')
            
            # Connect MoE components
            dot.edge(add_norm_name, gate_name)
            dot.edge(add_norm_name, moe_agg_name)
            
            # Connect experts
            for gpu in range(8):
                for expert_id in range(2):
                    expert_idx = gpu * 2 + expert_id
                    expert_name = f'{expert_prefix}_{expert_idx}'
                    dot.edge(gate_name, expert_name, style='dashed', label='routing')
                    dot.edge(add_norm_name, expert_name)
                    dot.edge(expert_name, moe_agg_name)
            
            dot.edge(moe_agg_name, add_norm2_name)
            dot.edge(add_norm_name, add_norm2_name, style='dashed', label='residual')
    
    # Pipeline communication
    comm_stage = dot.node('comm_stage_0_1', 'Pipeline Communication\\nStage 0 → Stage 1\\nGPU: 7 → 8', 
                          shape='ellipse', fillcolor='lightgray')
    
    # Pipeline Stage 1 (layers 8-15)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1\\nLayers 8-15\\nGPUs: 8-15', style='dashed', color='blue')
        
        for layer in range(8, 16):
            layer_name = f'layer_{layer}'
            
            # Multi-Head Attention
            mha_name = f'{layer_name}_mha'
            c.node(mha_name, f'MHA Layer {layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 8-15 (TP=8)')
            
            # MHA components
            qkv_name = f'{layer_name}_qkv'
            attn_name = f'{layer_name}_attn'
            proj_name = f'{layer_name}_proj'
            add_norm_name = f'{layer_name}_add_norm1'
            
            c.node(qkv_name, f'QKV Linear\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 8-15 (TP=8)')
            c.node(attn_name, f'Attention\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 8-15 (TP=8)')
            c.node(proj_name, f'Projection\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 8-15 (TP=8)')
            c.node(add_norm_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 8-15 (TP=8)')
            
            # MoE FFN
            moe_name = f'{layer_name}_moe'
            gate_name = f'{layer_name}_gate'
            expert_prefix = f'{layer_name}_expert'
            
            c.node(gate_name, f'Gating\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, topk=2]\\nGPU: 8-15', shape='parallelogram', fillcolor='yellow')
            
            # Create expert nodes (2 experts per GPU)
            for gpu in range(8, 16):
                for expert_id in range(2):
                    expert_idx = (gpu - 8) * 2 + expert_id
                    expert_name = f'{expert_prefix}_{expert_idx}'
                    c.node(expert_name, f'Expert {expert_idx}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: {gpu}')
            
            moe_agg_name = f'{layer_name}_moe_agg'
            c.node(moe_agg_name, f'MoE Aggregation\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 8-15 (TP=8)')
            
            add_norm2_name = f'{layer_name}_add_norm2'
            c.node(add_norm2_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: 8-15 (TP=8)')
            
            # Connect MHA components
            if layer == 8:
                comm_stage = 'comm_stage_0_1'
            else:
                comm_stage = f'layer_{layer-1}_add_norm2'
            
            dot.edge(comm_stage, qkv_name)
            dot.edge(qkv_name, attn_name)
            dot.edge(attn_name, proj_name)
            dot.edge(proj_name, add_norm_name)
            dot.edge(comm_stage, add_norm_name, style='dashed', label='residual')
            
            # Connect MoE components
            dot.edge(add_norm_name, gate_name)
            dot.edge(add_norm_name, moe_agg_name)
            
            # Connect experts
            for gpu in range(8, 16):
                for expert_id in range(2):
                    expert_idx = (gpu - 8) * 2 + expert_id
                    expert_name = f'{expert_prefix}_{expert_idx}'
                    dot.edge(gate_name, expert_name, style='dashed', label='routing')
                    dot.edge(add_norm_name, expert_name)
                    dot.edge(expert_name, moe_agg_name)
            
            dot.edge(moe_agg_name, add_norm2_name)
            dot.edge(add_norm_name, add_norm2_name, style='dashed', label='residual')
    
    # Output
    dot.node('output', f'Total Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    dot.edge('layer_15_add_norm2', 'output')
    
    return dot

def generate_proposed_dag():
    """Generate proposed DAG with EP=16, one expert per GPU"""
    dot = Digraph(name='proposed_moe_dag', 
                  comment='Proposed MoE with EP=16, single expert per GPU')
    dot.attr(rankdir='TB', splines='ortho', ranksep='3.0', nodesep='1.0')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal')
    
    # Global configuration
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    num_heads = 32
    head_dim = 128
    ffn_hidden = 16384
    
    # Input
    dot.node('input', f'Total Input\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Process all 16 layers with EP=16
    for layer in range(16):
        layer_name = f'layer_{layer}'
        gpu_offset = layer * 16  # Each layer uses 16 GPUs
        
        with dot.subgraph(name=f'cluster_layer_{layer}') as c:
            c.attr(label=f'Layer {layer}\\nEP=16 (16 GPUs)', style='dashed', color='purple')
            
            # Multi-Head Attention
            mha_name = f'{layer_name}_mha'
            c.node(mha_name, f'MHA Layer {layer}\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: {gpu_offset}')
            
            # MHA components
            qkv_name = f'{layer_name}_qkv'
            attn_name = f'{layer_name}_attn'
            proj_name = f'{layer_name}_proj'
            add_norm_name = f'{layer_name}_add_norm1'
            
            c.node(qkv_name, f'QKV Linear\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: {gpu_offset}')
            c.node(attn_name, f'Attention\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: {gpu_offset}')
            c.node(proj_name, f'Projection\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: {gpu_offset}')
            c.node(add_norm_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: {gpu_offset}')
            
            # MoE with EP=16
            gate_name = f'{layer_name}_gate'
            scatter_name = f'{layer_name}_scatter'
            gather_name = f'{layer_name}_gather'
            
            c.node(gate_name, f'Gating\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, topk=2]\\nGPU: {gpu_offset}', shape='parallelogram', fillcolor='yellow')
            c.node(scatter_name, f'Token Scatter\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: All 16', shape='ellipse', fillcolor='lightgray')
            c.node(gather_name, f'Token Gather\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: All 16', shape='ellipse', fillcolor='lightgray')
            
            add_norm2_name = f'{layer_name}_add_norm2'
            c.node(add_norm2_name, f'Residual+Norm\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: {gpu_offset}')
            
            # Create 16 expert nodes (one per GPU)
            for expert_id in range(16):
                gpu_id = gpu_offset + expert_id
                expert_name = f'{layer_name}_expert_{expert_id}'
                c.node(expert_name, f'Expert {expert_id}\\nInput: [tokens_per_expert, hidden={hidden_size}]\\nOutput: [tokens_per_expert, hidden={hidden_size}]\\nGPU: {gpu_id}')
            
            # Connect MHA components
            if layer == 0:
                dot.edge('input', qkv_name)
            else:
                dot.edge(f'layer_{layer-1}_add_norm2', qkv_name)
            
            dot.edge(qkv_name, attn_name)
            dot.edge(attn_name, proj_name)
            dot.edge(proj_name, add_norm_name)
            if layer == 0:
                dot.edge('input', add_norm_name, style='dashed', label='residual')
            else:
                dot.edge(f'layer_{layer-1}_add_norm2', add_norm_name, style='dashed', label='residual')
            
            # Connect MoE components
            dot.edge(add_norm_name, gate_name)
            dot.edge(add_norm_name, scatter_name)
            dot.edge(gate_name, scatter_name, style='dashed', label='routing_info')
            
            # Connect experts with scatter/gather
            for expert_id in range(16):
                gpu_id = gpu_offset + expert_id
                expert_name = f'{layer_name}_expert_{expert_id}'
                dot.edge(scatter_name, expert_name, label=f'to_GPU_{gpu_id}')
                dot.edge(expert_name, gather_name, label=f'from_GPU_{gpu_id}')
            
            dot.edge(gather_name, add_norm2_name)
            dot.edge(add_norm_name, add_norm2_name, style='dashed', label='residual')
    
    # Output
    dot.node('output', f'Total Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPU: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    dot.edge('layer_15_add_norm2', 'output')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-11-25-11-57-19', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = generate_baseline_dag()
    baseline_dag.save('../outputs/2025-11-25-11-57-19/baseline_moe_dag.dot')
    baseline_dag.render('../outputs/2025-11-25-11-57-19/baseline_moe_dag', format='svg', cleanup=True)
    
    # Generate proposed DAG
    proposed_dag = generate_proposed_dag()
    proposed_dag.save('../outputs/2025-11-25-11-57-19/proposed_moe_dag.dot')
    proposed_dag.render('../outputs/2025-11-25-11-57-19/proposed_moe_dag', format='svg', cleanup=True)
    
    print("DAGs generated successfully!")
    print("Files created:")
    print("- baseline_moe_dag.dot")
    print("- baseline_moe_dag.svg") 
    print("- proposed_moe_dag.dot")
    print("- proposed_moe_dag.svg")

if __name__ == "__main__":
    main()