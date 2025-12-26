#!/usr/bin/env python3

import os
from graphviz import Digraph

# Create the DAG
dot = Digraph(comment='10B MoE Transformer 4D Hybrid Parallelism DAG')

# Graph settings
dot.attr(nodesep='0.8')
dot.attr(rankdir='TB')
dot.attr(ranksep='1.2')
dot.attr(splines='ortho')

# Node style definitions
dot.attr('node', fillcolor='lightblue', shape='ellipse', style='filled')  # Communication
dot.attr('node', fillcolor='lightgreen', shape='rectangle', style='filled')  # Computation
dot.attr('node', fillcolor='lightyellow', shape='parallelogram', style='filled')  # Routing/Aggregation

# Input cluster
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Layer', style='rounded', fillcolor='lightgray')
    c.node('input', 'Input\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           shape='box', fillcolor='white', style='rounded,filled')

# Pipeline Stage 1: Layers 0-7 (GPUs 0-127)
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1: Layers 0-7 (GPUs 0-127)', style='rounded', fillcolor='lightcyan')
    
    # Layer 0
    c.node('layernorm0', 'LayerNorm (Layer 0)\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    # Attention block
    c.node('qkv_proj', 'QKV Projection TP=4\nGPU: 0-127 (TP groups)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, heads=4, d_k=32]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('qkv_allreduce', 'TP All-Reduce QKV\nGPU: 0-127 (TP groups)\nInput: [batch_size=32, seq_len=1024, heads=4, d_k=32]\nOutput: [batch_size=32, seq_len=1024, heads=16, d_k=32]', 
           fillcolor='lightblue', shape='ellipse')
    
    c.node('attention', 'Multi-Head Attention\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, heads=16, d_k=32]\nOutput: [batch_size=32, seq_len=1024, heads=16, d_k=32]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('attn_out', 'Attention Output Projection TP=4\nGPU: 0-127 (TP groups)\nInput: [batch_size=32, seq_len=1024, heads=16, d_k=32]\nOutput: [batch_size=32, seq_len=1024, token_dim=128]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('attn_allreduce', 'TP All-Reduce Attention\nGPU: 0-127 (TP groups)\nInput: [batch_size=32, seq_len=1024, token_dim=128]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightblue', shape='ellipse')
    
    c.node('residual1', 'Residual Connection\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    # MoE block
    c.node('moe_gate', 'MoE Gate (Expert Selection)\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, num_experts=16]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('expert_route', 'Expert Routing EP=8\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, experts=2]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('expert_0', 'Expert 0 Computation\nGPU: 0-127 (2 experts per GPU)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('expert_1', 'Expert 1 Computation\nGPU: 0-127 (2 experts per GPU)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('expert_combine', 'Expert Combination EP=8\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, experts=2]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('moe_residual', 'MoE Residual Connection\nGPU: 0-127\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')

# Pipeline communication between stages
dot.node('stage1_to_stage2', 'Pipeline Communication PP=2\nGPU: 0-127 → 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
         fillcolor='lightblue', shape='ellipse')

# Pipeline Stage 2: Layers 8-15 (GPUs 128-255)
with dot.subgraph(name='cluster_stage2') as c:
    c.attr(label='Pipeline Stage 2: Layers 8-15 (GPUs 128-255)', style='rounded', fillcolor='lightpink')
    
    # Layer 8
    c.node('layernorm8', 'LayerNorm (Layer 8)\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    # Attention block
    c.node('qkv_proj2', 'QKV Projection TP=4\nGPU: 128-255 (TP groups)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, heads=4, d_k=32]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('qkv_allreduce2', 'TP All-Reduce QKV\nGPU: 128-255 (TP groups)\nInput: [batch_size=32, seq_len=1024, heads=4, d_k=32]\nOutput: [batch_size=32, seq_len=1024, heads=16, d_k=32]', 
           fillcolor='lightblue', shape='ellipse')
    
    c.node('attention2', 'Multi-Head Attention\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, heads=16, d_k=32]\nOutput: [batch_size=32, seq_len=1024, heads=16, d_k=32]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('attn_out2', 'Attention Output Projection TP=4\nGPU: 128-255 (TP groups)\nInput: [batch_size=32, seq_len=1024, heads=16, d_k=32]\nOutput: [batch_size=32, seq_len=1024, token_dim=128]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('attn_allreduce2', 'TP All-Reduce Attention\nGPU: 128-255 (TP groups)\nInput: [batch_size=32, seq_len=1024, token_dim=128]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightblue', shape='ellipse')
    
    c.node('residual2', 'Residual Connection\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    # MoE block (MISSING IN PREVIOUS VERSION)
    c.node('moe_gate2', 'MoE Gate (Expert Selection)\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, num_experts=16]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('expert_route2', 'Expert Routing EP=8\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, experts=2]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('expert_2', 'Expert 2 Computation\nGPU: 128-255 (2 experts per GPU)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('expert_3', 'Expert 3 Computation\nGPU: 128-255 (2 experts per GPU)\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('expert_combine2', 'Expert Combination EP=8\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, experts=2]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightyellow', shape='parallelogram')
    
    c.node('moe_residual2', 'MoE Residual Connection\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')
    
    c.node('final_layernorm', 'Final LayerNorm\nGPU: 128-255\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, token_dim=512]', 
           fillcolor='lightgreen', shape='rectangle')

# Output cluster
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer', style='rounded', fillcolor='lightgray')
    c.node('output', 'Output\nInput: [batch_size=32, seq_len=1024, token_dim=512]\nOutput: [batch_size=32, seq_len=1024, vocab_size=V]', 
           shape='box', fillcolor='white', style='rounded,filled')

# Edges connecting the nodes
dot.edge('input', 'layernorm0')
dot.edge('layernorm0', 'qkv_proj')
dot.edge('qkv_proj', 'qkv_allreduce')
dot.edge('qkv_allreduce', 'attention')
dot.edge('attention', 'attn_out')
dot.edge('attn_out', 'attn_allreduce')
dot.edge('attn_allreduce', 'residual1')
dot.edge('residual1', 'moe_gate')
dot.edge('moe_gate', 'expert_route')
dot.edge('expert_route', 'expert_0', style='dashed')
dot.edge('expert_route', 'expert_1', style='dashed')
dot.edge('expert_0', 'expert_combine')
dot.edge('expert_1', 'expert_combine')
dot.edge('expert_combine', 'moe_residual')
dot.edge('moe_residual', 'stage1_to_stage2')

dot.edge('stage1_to_stage2', 'layernorm8')
dot.edge('layernorm8', 'qkv_proj2')
dot.edge('qkv_proj2', 'qkv_allreduce2')
dot.edge('qkv_allreduce2', 'attention2')
dot.edge('attention2', 'attn_out2')
dot.edge('attn_out2', 'attn_allreduce2')
dot.edge('attn_allreduce2', 'residual2')
dot.edge('residual2', 'moe_gate2')  # NEW: Connect to MoE block
dot.edge('moe_gate2', 'expert_route2')
dot.edge('expert_route2', 'expert_2', style='dashed')
dot.edge('expert_route2', 'expert_3', style='dashed')
dot.edge('expert_2', 'expert_combine2')
dot.edge('expert_3', 'expert_combine2')
dot.edge('expert_combine2', 'moe_residual2')
dot.edge('moe_residual2', 'final_layernorm')
dot.edge('final_layernorm', 'output')

# Save the DOT file
dot.save('./outputs/2025-12-26-10-50-02/parallel_strategy_deployment.dot')

# Render to SVG
dot.render('./outputs/2025-12-26-10-50-02/parallel_strategy_deployment', format='svg', cleanup=True)

print("DAG generated successfully!")
print("Files saved:")
print("- ./outputs/2025-12-26-10-50-02/parallel_strategy_deployment.dot")
print("- ./outputs/2025-12-26-10-50-02/parallel_strategy_deployment.svg")