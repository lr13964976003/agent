#!/usr/bin/env python3

import os
from graphviz import Digraph

# Create the output directory
os.makedirs('./outputs/2025-12-26-11-27-33', exist_ok=True)

# Create the DAG
dd = Digraph(comment='LLM Parallel Strategy Deployment DAG')
dd.attr(rankdir='TB', size='100,200', ranksep='2', nodesep='1')
dd.attr('node', fontname='Arial', fontsize='10')

# Define dimensions
batch_size = 128
seq_len = 512
heads = 16
dk = 32
hidden = 1024

# Create subgraph for Pipeline Stage 0 (GPUs 0-15)
with dd.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 (Layers 0-3)\nGPUs 0-15', style='rounded,filled', fillcolor='lightgray')
    
    # Input node
    c.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Layer 0 - Attention on GPU 0
    c.node('layer0_attn_gpu0', f'Layer 0 Attention\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Layer 0 - MoE Router on GPU 0
    c.node('layer0_moe_route_gpu0', f'Layer 0 MoE Router\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]', 
           shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Gate selection for Layer 0
    c.node('gate_layer0', f'Gate Selection\\nLayer 0\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]', 
           shape='diamond', style='filled', fillcolor='pink', peripheries='2')
    
    # Expert processing - distributed across GPUs 0-3
    for gpu_id in range(4):
        c.node(f'layer0_experts_gpu{gpu_id}', f'Layer 0 Experts (4/16)\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
               shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert communication
    for gpu_id in range(4):
        c.node(f'layer0_expert_comm_gpu{gpu_id}', f'Expert All-to-all\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, tokens=?]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, tokens=?]', 
               shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # MoE aggregation
    c.node('layer0_moe_agg_gpu0', f'Layer 0 MoE Aggregate\\nGPU 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
           shape='parallelogram', style='filled', fillcolor='lightyellow')

# Create subgraph for Pipeline Stage 1 (GPUs 16-31)
with dd.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 (Layers 4-7)\nGPUs 16-31', style='rounded,filled', fillcolor='lightgray')
    
    # Pipeline communication from stage 0 to 1
    c.node('pipe_comm_0_1', f'Pipeline Communication\\nStage 0 to 1\\nGPUs 0-15 to GPUs 16-31\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
           shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Layer 4 - Attention on GPU 16
    c.node('layer4_attn_gpu16', f'Layer 4 Attention\\nGPU 16\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert processing for layer 4 - distributed across GPUs 16-19
    for gpu_id in range(16, 20):
        c.node(f'layer4_experts_gpu{gpu_id}', f'Layer 4 Experts (4/16)\\nGPU {gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts=2]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
               shape='rectangle', style='filled', fillcolor='lightblue')

# Create subgraph for Pipeline Stage 2 (GPUs 32-47)
with dd.subgraph(name='cluster_stage2') as c:
    c.attr(label='Pipeline Stage 2 (Layers 8-11)\nGPUs 32-47', style='rounded,filled', fillcolor='lightgray')
    
    # Pipeline communication from stage 1 to 2
    c.node('pipe_comm_1_2', f'Pipeline Communication\\nStage 1 to 2\\nGPUs 16-31 to GPUs 32-47\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
           shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Layer 8 - Attention on GPU 32
    c.node('layer8_attn_gpu32', f'Layer 8 Attention\\nGPU 32\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='rectangle', style='filled', fillcolor='lightblue')

# Create subgraph for Pipeline Stage 3 (GPUs 48-63)
with dd.subgraph(name='cluster_stage3') as c:
    c.attr(label='Pipeline Stage 3 (Layers 12-15)\nGPUs 48-63', style='rounded,filled', fillcolor='lightgray')
    
    # Pipeline communication from stage 2 to 3
    c.node('pipe_comm_2_3', f'Pipeline Communication\\nStage 2 to 3\\nGPUs 32-47 to GPUs 48-63\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
           shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Layer 12 - Attention on GPU 48
    c.node('layer12_attn_gpu48', f'Layer 12 Attention\\nGPU 48\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Output node
    c.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={dk}]', 
           shape='ellipse', style='filled', fillcolor='lightcoral')

# Add data parallel communication nodes
for dp_group in range(4):
    start_gpu = dp_group * 16
    dd.node(f'dp_comm_group{dp_group}', f'Data Parallel All-Reduce\\nDP Group {dp_group}\\nGPUs {start_gpu}-{start_gpu+15}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden}]', 
           shape='ellipse', style='filled', fillcolor='lightgreen')

# Define edges showing the flow
# Input to first attention layer
dd.edge('input', 'layer0_attn_gpu0')
dd.edge('layer0_attn_gpu0', 'layer0_moe_route_gpu0')

# Gate selection with dashed lines
dd.edge('layer0_moe_route_gpu0', 'gate_layer0', style='dashed')

# Expert distribution
dd.edge('gate_layer0', 'layer0_experts_gpu0', style='dashed')
dd.edge('gate_layer0', 'layer0_experts_gpu1', style='dashed')
dd.edge('gate_layer0', 'layer0_experts_gpu2', style='dashed')
dd.edge('gate_layer0', 'layer0_experts_gpu3', style='dashed')

# Expert processing and communication
for gpu_id in range(4):
    dd.edge(f'layer0_experts_gpu{gpu_id}', f'layer0_expert_comm_gpu{gpu_id}')
    dd.edge(f'layer0_expert_comm_gpu{gpu_id}', 'layer0_moe_agg_gpu0')

# Continue to next stage
dd.edge('layer0_moe_agg_gpu0', 'pipe_comm_0_1')
dd.edge('pipe_comm_0_1', 'layer4_attn_gpu16')

# Continue through stages
dd.edge('layer4_attn_gpu16', 'pipe_comm_1_2')
dd.edge('pipe_comm_1_2', 'layer8_attn_gpu32')
dd.edge('layer8_attn_gpu32', 'pipe_comm_2_3')
dd.edge('pipe_comm_2_3', 'layer12_attn_gpu48')
dd.edge('layer12_attn_gpu48', 'output')

# Data parallel communications
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