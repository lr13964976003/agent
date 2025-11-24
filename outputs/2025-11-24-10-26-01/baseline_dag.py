#!/usr/bin/env python3
import graphviz

# Create baseline DAG for TP=8, PP=2
dot = graphviz.Digraph('baseline_tp8_pp2', comment='Dense 16-layer model with TP=8, PP=2')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')

# Define node shapes and styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')

# Input node
dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]', 
         shape='parallelogram', fillcolor='lightgreen')

# Stage 0: layers 0-7 on GPUs 0-7
with dot.subgraph(name='cluster_stage0') as stage0:
    stage0.attr(label='Stage 0 (Layers 0-7)\\nGPUs 0-7 (TP=8)', style='rounded,dashed', color='red')
    
    # Add input to stage0
    stage0.node('stage0_input', 'Input to Stage 0\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]', 
                shape='parallelogram', fillcolor='lightyellow')
    
    # Process layers 0-7
    for layer in range(8):
        with stage0.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer}\\nGPUs 0-7', style='dotted')
            
            # Attention components with tensor parallelism
            layer_cluster.node(f'lay{layer}_qkv_proj', 
                             f'QKV Projection (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightcoral')
            
            layer_cluster.node(f'lay{layer}_attn', 
                             f'Multi-Head Attention (TP=8)\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightpink')
            
            layer_cluster.node(f'lay{layer}_attn_out', 
                             f'Attention Output Projection (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightcoral')
            
            layer_cluster.node(f'lay{layer}_res1', 
                             f'Residual Add 1\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 0-7', 
                             shape='ellipse', fillcolor='lightgray')
            
            # MLP components with tensor parallelism
            layer_cluster.node(f'lay{layer}_mlp_gate', 
                             f'MLP Gate (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_mlp_up', 
                             f'MLP Up (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_mlp_act', 
                             f'GELU Activation\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightblue')
            
            layer_cluster.node(f'lay{layer}_mlp_down', 
                             f'MLP Down (TP=8)\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nGPUs 0-7', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_res2', 
                             f'Residual Add 2\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 0-7', 
                             shape='ellipse', fillcolor='lightgray')

# Stage 1: layers 8-15 on GPUs 8-15
with dot.subgraph(name='cluster_stage1') as stage1:
    stage1.attr(label='Stage 1 (Layers 8-15)\\nGPUs 8-15 (TP=8)', style='rounded,dashed', color='blue')
    
    # Add transition between stages
    stage1.node('stage1_input', 'Input to Stage 1\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]', 
                shape='parallelogram', fillcolor='lightyellow')
    
    # Process layers 8-15
    for layer in range(8, 16):
        with stage1.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer}\\nGPUs 8-15', style='dotted')
            
            # Attention components with tensor parallelism
            layer_cluster.node(f'lay{layer}_qkv_proj', 
                             f'QKV Projection (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightcoral')
            
            layer_cluster.node(f'lay{layer}_attn', 
                             f'Multi-Head Attention (TP=8)\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightpink')
            
            layer_cluster.node(f'lay{layer}_attn_out', 
                             f'Attention Output Projection (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightcoral')
            
            layer_cluster.node(f'lay{layer}_res1', 
                             f'Residual Add 1\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 8-15', 
                             shape='ellipse', fillcolor='lightgray')
            
            # MLP components with tensor parallelism
            layer_cluster.node(f'lay{layer}_mlp_gate', 
                             f'MLP Gate (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_mlp_up', 
                             f'MLP Up (TP=8)\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_mlp_act', 
                             f'GELU Activation\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightblue')
            
            layer_cluster.node(f'lay{layer}_mlp_down', 
                             f'MLP Down (TP=8)\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nGPUs 8-15', 
                             shape='rectangle', fillcolor='lightseagreen')
            
            layer_cluster.node(f'lay{layer}_res2', 
                             f'Residual Add 2\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPUs 8-15', 
                             shape='ellipse', fillcolor='lightgray')

# Output node
dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]', 
         shape='parallelogram', fillcolor='lightgreen')

# Communication nodes between tensor parallel groups
dot.node('tp_allreduce', 'Tensor Parallel All-Reduce\\nAcross GPUs 0-7 or 8-15\\nSize: 4096 elements per token', 
         shape='ellipse', fillcolor='yellow', style='dashed')

# Connections for Stage 0
dot.edge('input', 'stage0_input', label='Full Model Input')
dot.edge('stage0_input', 'lay0_qkv_proj')

# Layer 0 connections
for layer in range(8):
    dot.edge(f'lay{layer}_qkv_proj', f'lay{layer}_attn')
    dot.edge(f'lay{layer}_attn', f'lay{layer}_attn_out')
    dot.edge(f'lay{layer}_attn_out', f'lay{layer}_res1')
    dot.edge(f'lay{layer}_res1', f'lay{layer}_mlp_gate')
    dot.edge(f'lay{layer}_res1', f'lay{layer}_mlp_up')
    dot.edge(f'lay{layer}_mlp_gate', f'lay{layer}_mlp_act')
    dot.edge(f'lay{layer}_mlp_up', f'lay{layer}_mlp_act')
    dot.edge(f'lay{layer}_mlp_act', f'lay{layer}_mlp_down')
    dot.edge(f'lay{layer}_mlp_down', f'lay{layer}_res2')
    
    if layer < 7:
        dot.edge(f'lay{layer}_res2', f'lay{layer+1}_qkv_proj')
    else:
        dot.edge('lay7_res2', 'stage1_input', label='Pipeline Transfer\\nSize: 5.24GB')

# Connections for Stage 1
dot.edge('stage1_input', 'lay8_qkv_proj')

for layer in range(8, 16):
    dot.edge(f'lay{layer}_qkv_proj', f'lay{layer}_attn')
    dot.edge(f'lay{layer}_attn', f'lay{layer}_attn_out')
    dot.edge(f'lay{layer}_attn_out', f'lay{layer}_res1')
    dot.edge(f'lay{layer}_res1', f'lay{layer}_mlp_gate')
    dot.edge(f'lay{layer}_res1', f'lay{layer}_mlp_up')
    dot.edge(f'lay{layer}_mlp_gate', f'lay{layer}_mlp_act')
    dot.edge(f'lay{layer}_mlp_up', f'lay{layer}_mlp_act')
    dot.edge(f'lay{layer}_mlp_act', f'lay{layer}_mlp_down')
    dot.edge(f'lay{layer}_mlp_down', f'lay{layer}_res2')
    
    if layer < 15:
        dot.edge(f'lay{layer}_res2', f'lay{layer+1}_qkv_proj')
    else:
        dot.edge('lay15_res2', 'output')

# All-reduce operations for tensor parallelism
for layer in range(16):
    for op in ['attn_out', 'mlp_down']:
        dot.edge(f'lay{layer}_{op}', 'tp_allreduce', style='dashed', constraint='false')
        dot.edge('tp_allreduce', f'lay{layer}_{op}', style='dashed', constraint='false')

# Save the DOT file
dot.format = 'dot'
dot.render('../outputs/2025-11-24-10-26-01/baseline_dag')

# Save as SVG
dot.format = 'svg'
dot.render('../outputs/2025-11-24-10-26-01/baseline_dag')

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-11-24-10-26-01/baseline_dag.dot")
print("- ../outputs/2025-11-24-10-26-01/baseline_dag.svg")