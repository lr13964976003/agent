#!/usr/bin/env python3

import graphviz

# Create prefill DAG
dot = graphviz.Digraph(comment='Prefill Phase DAG - Qwen3-235B', format='svg')
dot.attr(rankdir='TB', size='20,30')
dot.attr('node', shape='rectangle', style='filled')

# Define colors for different GPU groups
gpu_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
gpu_labels = ['GPUs 0-8', 'GPUs 9-17', 'GPUs 18-26', 'GPUs 27-34']

# Input node
dot.node('input', 'Input Embedding\nGPU: 0-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', 
         shape='ellipse', fillcolor='lightgray')

# Stage 0: Layers 0-23 (GPUs 0-8)
with dot.subgraph(name='cluster_stage0') as stage0:
    stage0.attr(rank='same', style='rounded', fillcolor=gpu_colors[0], label='Stage 0: Layers 0-23\n' + gpu_labels[0])
    
    # Layer 0 operations
    stage0.node('stage0_layer0_attn', 'Layer 0 Attention\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[0])
    stage0.node('stage0_layer0_gate', 'Gate Selection (Top-8)\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage0.node('stage0_layer0_moe', 'Layer 0 MoE (128 experts)\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[0])
    
    # Layer 23 operations (last layer in stage 0)
    stage0.node('stage0_layer23_attn', 'Layer 23 Attention\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[0])
    stage0.node('stage0_layer23_gate', 'Gate Selection (Top-8)\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage0.node('stage0_layer23_moe', 'Layer 23 MoE (128 experts)\nGPU: 0-8\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[0])

# Stage 1: Layers 24-47 (GPUs 9-17)
with dot.subgraph(name='cluster_stage1') as stage1:
    stage1.attr(rank='same', style='rounded', fillcolor=gpu_colors[1], label='Stage 1: Layers 24-47\n' + gpu_labels[1])
    
    # Layer 24 operations (first layer in stage 1)
    stage1.node('stage1_layer24_attn', 'Layer 24 Attention\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[1])
    stage1.node('stage1_layer24_gate', 'Gate Selection (Top-8)\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage1.node('stage1_layer24_moe', 'Layer 24 MoE (128 experts)\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[1])
    
    # Layer 47 operations (last layer in stage 1)
    stage1.node('stage1_layer47_attn', 'Layer 47 Attention\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[1])
    stage1.node('stage1_layer47_gate', 'Gate Selection (Top-8)\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage1.node('stage1_layer47_moe', 'Layer 47 MoE (128 experts)\nGPU: 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[1])

# Stage 2: Layers 48-71 (GPUs 18-26)
with dot.subgraph(name='cluster_stage2') as stage2:
    stage2.attr(rank='same', style='rounded', fillcolor=gpu_colors[2], label='Stage 2: Layers 48-71\n' + gpu_labels[2])
    
    # Layer 48 operations (first layer in stage 2)
    stage2.node('stage2_layer48_attn', 'Layer 48 Attention\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[2])
    stage2.node('stage2_layer48_gate', 'Gate Selection (Top-8)\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage2.node('stage2_layer48_moe', 'Layer 48 MoE (128 experts)\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[2])
    
    # Layer 71 operations (last layer in stage 2)
    stage2.node('stage2_layer71_attn', 'Layer 71 Attention\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[2])
    stage2.node('stage2_layer71_gate', 'Gate Selection (Top-8)\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage2.node('stage2_layer71_moe', 'Layer 71 MoE (128 experts)\nGPU: 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[2])

# Stage 3: Layers 72-93 (GPUs 27-34)
with dot.subgraph(name='cluster_stage3') as stage3:
    stage3.attr(rank='same', style='rounded', fillcolor=gpu_colors[3], label='Stage 3: Layers 72-93\n' + gpu_labels[3])
    
    # Layer 72 operations (first layer in stage 3)
    stage3.node('stage3_layer72_attn', 'Layer 72 Attention\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[3])
    stage3.node('stage3_layer72_gate', 'Gate Selection (Top-8)\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage3.node('stage3_layer72_moe', 'Layer 72 MoE (128 experts)\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[3])
    
    # Layer 93 operations (last layer in stage 3)
    stage3.node('stage3_layer93_attn', 'Layer 93 Attention\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[3])
    stage3.node('stage3_layer93_gate', 'Gate Selection (Top-8)\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, top_k=8]', shape='parallelogram', fillcolor='orange')
    stage3.node('stage3_layer93_moe', 'Layer 93 MoE (128 experts)\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', fillcolor=gpu_colors[3])

# Output node
dot.node('output', 'Output Projection\nGPU: 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, vocab_size=151936]', 
         shape='ellipse', fillcolor='lightgray')

# Communication nodes between stages
dot.node('comm_0_1', 'Pipeline Communication\nGPUs: 0-8 → 9-17\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', 
         shape='ellipse', fillcolor='pink', style='dashed')

dot.node('comm_1_2', 'Pipeline Communication\nGPUs: 9-17 → 18-26\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', 
         shape='ellipse', fillcolor='pink', style='dashed')

dot.node('comm_2_3', 'Pipeline Communication\nGPUs: 18-26 → 27-34\nInput: [batch_size=128, seq_len=2048, d_model=4096]\nOutput: [batch_size=128, seq_len=2048, d_model=4096]', 
         shape='ellipse', fillcolor='pink', style='dashed')

# Connections - Input to Stage 0
dot.edge('input', 'stage0_layer0_attn')
dot.edge('stage0_layer0_attn', 'stage0_layer0_gate')
dot.edge('stage0_layer0_gate', 'stage0_layer0_moe', style='dashed', label='Top-8 expert selection')

# Connections within Stage 0 (simplified - showing key connections)
dot.edge('stage0_layer0_moe', 'stage0_layer23_attn', label='Layers 1-22')
dot.edge('stage0_layer23_attn', 'stage0_layer23_gate')
dot.edge('stage0_layer23_gate', 'stage0_layer23_moe', style='dashed', label='Top-8 expert selection')

# Communication between stages
dot.edge('stage0_layer23_moe', 'comm_0_1')
dot.edge('comm_0_1', 'stage1_layer24_attn')

# Stage 1 connections
dot.edge('stage1_layer24_attn', 'stage1_layer24_gate')
dot.edge('stage1_layer24_gate', 'stage1_layer24_moe', style='dashed', label='Top-8 expert selection')
dot.edge('stage1_layer47_attn', 'stage1_layer47_gate')
dot.edge('stage1_layer47_gate', 'stage1_layer47_moe', style='dashed', label='Top-8 expert selection')

# Communication between stages 1-2
dot.edge('stage1_layer47_moe', 'comm_1_2')
dot.edge('comm_1_2', 'stage2_layer48_attn')

# Stage 2 connections
dot.edge('stage2_layer48_attn', 'stage2_layer48_gate')
dot.edge('stage2_layer48_gate', 'stage2_layer48_moe', style='dashed', label='Top-8 expert selection')
dot.edge('stage2_layer71_attn', 'stage2_layer71_gate')
dot.edge('stage2_layer71_gate', 'stage2_layer71_moe', style='dashed', label='Top-8 expert selection')

# Communication between stages 2-3
dot.edge('stage2_layer71_moe', 'comm_2_3')
dot.edge('comm_2_3', 'stage3_layer72_attn')

# Stage 3 connections
dot.edge('stage3_layer72_attn', 'stage3_layer72_gate')
dot.edge('stage3_layer72_gate', 'stage3_layer72_moe', style='dashed', label='Top-8 expert selection')
dot.edge('stage3_layer93_attn', 'stage3_layer93_gate')
dot.edge('stage3_layer93_gate', 'stage3_layer93_moe', style='dashed', label='Top-8 expert selection')

# Final output
dot.edge('stage3_layer93_moe', 'output')

# Save the DOT file
dot.save('./outputs/2026-01-04-16-26-17/prefill_dag.dot')

# Render to SVG
dot.render('./outputs/2026-01-04-16-26-17/prefill_dag', format='svg', cleanup=True)

print("Prefill DAG generated successfully!")
print(f"DOT file saved: ./outputs/2026-01-04-16-26-17/prefill_dag.dot")
print(f"SVG file saved: ./outputs/2026-01-04-16-26-17/prefill_dag.svg")