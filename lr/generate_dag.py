#!/usr/bin/env python3

import os
from graphviz import Digraph

# Create the output directory
os.makedirs('./outputs/2025-12-26-11-27-33', exist_ok=True)

# Create the DAG
dd = Digraph(comment='LLM Parallel Strategy Deployment DAG')
dd.attr(rankdir='TB', size='100,200', ranksep='2', nodesep='1')
dd.attr('node', fontname='Arial', fontsize='10')

# Define node styles
comp_style = 'shape=rectangle,style=filled,fillcolor=lightblue'
comm_style = 'shape=ellipse,style=filled,fillcolor=lightgreen'
route_style = 'shape=parallelogram,style=filled,fillcolor=lightyellow'
gate_style = 'shape=diamond,style=filled,fillcolor=pink,peripheries=2'

# Input dimensions
batch_size = 128
seq_len = 512
heads = 16
dk = 32
hidden = 1024

# Create the DAG for a single micro-batch through the system
# We'll show the complete flow through all 4 pipeline stages

# Stage 0: Layers 0-3 (GPUs 0-15)
with dd.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 (Layers 0-3)\nGPUs 0-15', style='rounded,filled', fillcolor='lightgray')
    
    # Input node
    c.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Layer 0
    c.node('layer0_attn_gpu0', f'Layer 0 Attention\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', comp_style)
    c.node('layer0_moe_route_gpu0', f'Layer 0 MoE Router\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]', route_style)
    
    # Expert processing for layer 0 (experts distributed across GPUs 0-3)
    for gpu_id in range(4):
        c.node(f'layer0_experts_gpu{gpu_id}', f'Layer 0 Experts (4/16)\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', comp_style)
    
    c.node('layer0_moe_agg_gpu0', f'Layer 0 MoE Aggregate\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', route_style)
    
    # Communication between experts
    for gpu_id in range(4):
        c.node(f'layer0_expert_comm_gpu{gpu_id}', f'Expert All-to-aq\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, tokens=?]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, tokens=?]', comm_style)

# Stage 1: Layers 4-7 (GPUs 16-31)  
with dd.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 (Layers 4-7)\nGPUs 16-31', style='rounded,filled', fillcolor='lightgray')
    
    # Layer 4
    c.node('layer4_attn_gpu16', f'Layer 4 Attention\\nGPU 16\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', comp_style)
    
    # Expert processing for layer 4 (experts distributed across GPUs 16-19)
    for gpu_id in range(16, 20):
        c.node(f'layer4_experts_gpu{gpu_id}', f'Layer 4 Experts (4/16)\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', comp_style)

# Stage 2: Layers 8-11 (GPUs 32-47)
with dd.subgraph(name='cluster_stage2') as c:
    c.attr(label='Pipeline Stage 2 (Layers 8-11)\nGPUs 32-47', style='rounded,filled', fillcolor='lightgray')
    
    # Layer 8
    c.node('layer8_attn_gpu32', f'Layer 8 Attention\\nGPU 32\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', comp_style)

# Stage 3: Layers 12-15 (GPUs 48-63)
with dd.subgraph(name='cluster_stage3') as c:
    c.attr(label='Pipeline Stage 3 (Layers 12-15)\nGPUs 48-63', style='rounded,filled', fillcolor='lightgray')
    
    # Layer 12
    c.node('layer12_attn_gpu48', f'Layer 12 Attention\\nGPU 48\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', comp_style)
    
    # Output node
    c.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='ellipse', style='filled', fillcolor='lightcoral')

# Add gate selection nodes (diamond shape, dashed connections)
for layer in range(0, 16, 4):  # Only show gates for first layer of each stage
    for gpu_id in [0, 16, 32, 48]:
        dd.node(f'gate_layer{layer}_gpu{gpu_id}', f'Gate Selection\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]', gate_style)

# Add pipeline communication nodes
for stage in range(3):
    start_gpu = stage * 16
    end_gpu = (stage + 1) * 16
    dd.node(f'pipe_comm_stage{stage}_to_{stage+1}', f'Pipeline Communication\\nStage {stage} to {stage+1}\\nGPUs {start_gpu}-{end_gpu-1} to GPUs {end_gpu}-{end_gpu+15}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', comm_style)

# Add data parallel communication nodes
for dp_group in range(4):
    start_gpu = dp_group * 16
    dd.node(f'dp_comm_group{dp_group}', f'Data Parallel All-Reduce\\nDP Group {dp_group}\\nGPUs {start_gpu}-{start_gpu+15}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', comm_style)

# Define edges
# Input to first layer
dd.edge('input', 'layer0_attn_gpu0')
dd.edge('layer0_attn_gpu0', 'layer0_moe_route_gpu0')

# Gate selections (dashed lines)
for gpu_id in range(4):
    dd.edge('layer0_moe_route_gpu0', f'gate_layer0_gpu{gpu_id}', style='dashed')
    dd.edge(f'gate_layer0_gpu{gpu_id}', f'layer0_experts_gpu{gpu_id}', style='dashed')

# Expert processing
for gpu_id in range(4):
    dd.edge(f'layer0_experts_gpu{gpu_id}', f'layer0_expert_comm_gpu{gpu_id}')
    dd.edge(f'layer0_expert_comm_gpu{gpu_id}', 'layer0_moe_agg_gpu0')

# Continue with simplified connections for remaining stages
dd.edge('layer0_moe_agg_gpu0', 'pipe_comm_stage0_to_1')
dd.edge('pipe_comm_stage0_to_1', 'layer4_attn_gpu16')

# Add more connections for other stages
dd.edge('layer4_attn_gpu16', 'pipe_comm_stage1_to_2')
dd.edge('pipe_comm_stage1_to_2', 'layer8_attn_gpu32')
dd.edge('layer8_attn_gpu32', 'pipe_comm_stage2_to_3')
dd.edge('pipe_comm_stage2_to_3', 'layer12_attn_gpu48')
dd.edge('layer12_attn_gpu48', 'output')

# Add data parallel communications at the end
dd.edge('output', 'dp_comm_group0')
dd.edge('output', 'dp_comm_group1')
dd.edge('output', 'dp_comm_group2')
dd.edge('output', 'dp_comm_group3')

# Save the DOT file
dd.save('./outputs/2025-12-26-11-27-33/llm_parallel_dag.dot')

# Render to SVG
dd.format = 'svg'
dd.render('./outputs/2025-12-26-11-27-33/llm_parallel_dag', cleanup=True)

print("DAG generated successfully!")
print(f"DOT file saved to: ./outputs/2025-12-26-11-27-33/llm_parallel_dag.dot")
print(f"SVG file saved to: ./outputs/2025-12-26-11-27-33/llm_parallel_dag.svg")