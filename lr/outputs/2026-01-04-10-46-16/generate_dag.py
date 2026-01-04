#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Qwen3-235B Parallel Strategy Deployment DAG')
dot.attr(rankdir='TB')
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

# Model configuration
batch_size = 128
seq_len = 2048
hidden_dim = 4096
num_heads = 64
head_dim = hidden_dim // num_heads
num_experts = 128

# Add input node
dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', 
         shape='ellipse', fillcolor='white')

# Pipeline Stage 0: GPUs 0-7 (Layers 0-22)
with dot.subgraph(name='cluster_stage0') as stage0:
    stage0.attr(label='Pipeline Stage 0: GPUs 0-7 (Layers 0-22)', style='rounded,filled', fillcolor='lightblue')
    
    # Embedding layer
    stage0.node('embed_0', f'Embedding (GPU-0)\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]')
    
    # Layer 0-22 with experts and attention
    for layer in range(23):
        # Attention computation - split across GPUs 0-1 (TP=2)
        stage0.node(f'attn_qkv_{layer}_0', f'Attention QKV Proj (GPU-0)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage0.node(f'attn_qkv_{layer}_1', f'Attention QKV Proj (GPU-1)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        # Attention computation
        stage0.node(f'attn_score_{layer}_0', f'Attention Scores (GPU-0)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage0.node(f'attn_score_{layer}_1', f'Attention Scores (GPU-1)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        # Attention output projection
        stage0.node(f'attn_out_{layer}_0', f'Attention Output Proj (GPU-0)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        stage0.node(f'attn_out_{layer}_1', f'Attention Output Proj (GPU-1)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        
        # Gate mechanism - routing (dashed line)
        stage0.node(f'gate_{layer}', f'Gate Routing (GPU-2)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]', shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        # Expert computation - all 128 experts on each GPU (EP=1)
        for expert_id in range(4):  # Show first 4 experts for clarity
            stage0.node(f'expert_{layer}_{expert_id}', f'Expert {expert_id} (GPU-{expert_id%4})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightcoral')
        
        # Expert aggregation
        stage0.node(f'expert_agg_{layer}', f'Expert Aggregation (GPU-4)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', shape='parallelogram', fillcolor='lightyellow')

# Pipeline Stage 1: GPUs 8-15 (Layers 23-46)
with dot.subgraph(name='cluster_stage1') as stage1:
    stage1.attr(label='Pipeline Stage 1: GPUs 8-15 (Layers 23-46)', style='rounded,filled', fillcolor='lightgreen')
    
    for layer in range(23, 47):
        # Similar structure as stage 0
        stage1.node(f'attn_qkv_{layer}_0', f'Attention QKV Proj (GPU-8)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage1.node(f'attn_qkv_{layer}_1', f'Attention QKV Proj (GPU-9)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage1.node(f'attn_score_{layer}_0', f'Attention Scores (GPU-8)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage1.node(f'attn_score_{layer}_1', f'Attention Scores (GPU-9)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage1.node(f'attn_out_{layer}_0', f'Attention Output Proj (GPU-8)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        stage1.node(f'attn_out_{layer}_1', f'Attention Output Proj (GPU-9)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        
        stage1.node(f'gate_{layer}', f'Gate Routing (GPU-10)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]', shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            stage1.node(f'expert_{layer}_{expert_id}', f'Expert {expert_id} (GPU-{8+expert_id%4})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightcoral')
        
        stage1.node(f'expert_agg_{layer}', f'Expert Aggregation (GPU-12)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', shape='parallelogram', fillcolor='lightyellow')

# Pipeline Stage 2: GPUs 16-23 (Layers 47-69)
with dot.subgraph(name='cluster_stage2') as stage2:
    stage2.attr(label='Pipeline Stage 2: GPUs 16-23 (Layers 47-69)', style='rounded,filled', fillcolor='lightyellow')
    
    for layer in range(47, 70):
        stage2.node(f'attn_qkv_{layer}_0', f'Attention QKV Proj (GPU-16)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage2.node(f'attn_qkv_{layer}_1', f'Attention QKV Proj (GPU-17)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage2.node(f'attn_score_{layer}_0', f'Attention Scores (GPU-16)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage2.node(f'attn_score_{layer}_1', f'Attention Scores (GPU-17)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage2.node(f'attn_out_{layer}_0', f'Attention Output Proj (GPU-16)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        stage2.node(f'attn_out_{layer}_1', f'Attention Output Proj (GPU-17)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        
        stage2.node(f'gate_{layer}', f'Gate Routing (GPU-18)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]', shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            stage2.node(f'expert_{layer}_{expert_id}', f'Expert {expert_id} (GPU-{16+expert_id%4})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightcoral')
        
        stage2.node(f'expert_agg_{layer}', f'Expert Aggregation (GPU-20)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', shape='parallelogram', fillcolor='lightyellow')

# Pipeline Stage 3: GPUs 24-31 (Layers 70-93)
with dot.subgraph(name='cluster_stage3') as stage3:
    stage3.attr(label='Pipeline Stage 3: GPUs 24-31 (Layers 70-93)', style='rounded,filled', fillcolor='lightcoral')
    
    for layer in range(70, 94):
        stage3.node(f'attn_qkv_{layer}_0', f'Attention QKV Proj (GPU-24)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage3.node(f'attn_qkv_{layer}_1', f'Attention QKV Proj (GPU-25)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage3.node(f'attn_score_{layer}_0', f'Attention Scores (GPU-24)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        stage3.node(f'attn_score_{layer}_1', f'Attention Scores (GPU-25)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]', fillcolor='lightgreen')
        
        stage3.node(f'attn_out_{layer}_0', f'Attention Output Proj (GPU-24)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        stage3.node(f'attn_out_{layer}_1', f'Attention Output Proj (GPU-25)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=32, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightgreen')
        
        stage3.node(f'gate_{layer}', f'Gate Routing (GPU-26)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]', shape='parallelogram', fillcolor='lightyellow', style='dashed')
        
        for expert_id in range(4):
            stage3.node(f'expert_{layer}_{expert_id}', f'Expert {expert_id} (GPU-{24+expert_id%4})\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', fillcolor='lightcoral')
        
        stage3.node(f'expert_agg_{layer}', f'Expert Aggregation (GPU-28)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', shape='parallelogram', fillcolor='lightyellow')

# Output node
dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]', 
         shape='ellipse', fillcolor='white')

# Add edges (connections)
# Input to embedding
dot.edge('input', 'embed_0')

# Connections within layers (showing first layer as example)
for layer in range(94):
    gpu_base = (layer // 24) * 8  # GPU base for each stage
    
    if layer == 0:
        dot.edge('embed_0', f'attn_qkv_{layer}_0')
        dot.edge('embed_0', f'attn_qkv_{layer}_1')
    else:
        prev_gpu_base = ((layer-1) // 24) * 8
        dot.edge(f'expert_agg_{layer-1}', f'attn_qkv_{layer}_0')
        dot.edge(f'expert_agg_{layer-1}', f'attn_qkv_{layer}_1')
    
    # Attention connections
    dot.edge(f'attn_qkv_{layer}_0', f'attn_score_{layer}_0')
    dot.edge(f'attn_qkv_{layer}_1', f'attn_score_{layer}_1')
    dot.edge(f'attn_score_{layer}_0', f'attn_out_{layer}_0')
    dot.edge(f'attn_score_{layer}_1', f'attn_out_{layer}_1')
    
    # Add communication edges for attention (dashed)
    dot.edge(f'attn_qkv_{layer}_0', f'attn_score_{layer}_1', style='dashed', color='red', label='TP Comm')
    dot.edge(f'attn_qkv_{layer}_1', f'attn_score_{layer}_0', style='dashed', color='red', label='TP Comm')
    
    # Gate routing (dashed)
    dot.edge(f'attn_out_{layer}_0', f'gate_{layer}', style='dashed')
    dot.edge(f'attn_out_{layer}_1', f'gate_{layer}', style='dashed')
    
    # Expert connections
    for expert_id in range(4):
        dot.edge(f'gate_{layer}', f'expert_{layer}_{expert_id}', style='dashed', color='blue', label='routing')
        dot.edge(f'expert_{layer}_{expert_id}', f'expert_agg_{layer}')

# Final output
dot.edge('expert_agg_93', 'output')

# Save the DOT file and generate SVG
dot.render('./outputs/2026-01-04-10-46-16/qwen3_235b_parallel_dag', format='svg', cleanup=False)
dot.render('./outputs/2026-01-04-10-46-16/qwen3_235b_parallel_dag', format='dot', cleanup=False)

print("DAG generated successfully as SVG and DOT files!")
print("Files saved:")
print("- ./outputs/2026-01-04-10-46-16/qwen3_235b_parallel_dag.svg")
print("- ./outputs/2026-01-04-10-46-16/qwen3_235b_parallel_dag.dot")