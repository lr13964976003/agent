import graphviz

# Create baseline DAG with TP=8, PP=2
dot = graphviz.Digraph(comment='Baseline MoE DAG (TP=8, PP=2)', format='svg')
dot.attr(rankdir='TB', size='20,20')

# Define colors for different GPUs
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan', 'lightgray', 'lightseagreen']

# Input node
dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, hidden_size=8192]', shape='ellipse', style='filled', fillcolor='white')

# Layer 1 - First pipeline stage (GPUs 0-7)
with dot.subgraph(name='cluster_layer1_stage1') as c:
    c.attr(label='Layer 1 - Pipeline Stage 1 (GPUs 0-7)', style='dashed')
    
    # LayerNorm on GPU 0
    c.node('l1_ln0', 'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: 0', shape='rectangle', style='filled', fillcolor=colors[0])
    
    # Multi-Head Attention - Tensor Parallel across 8 GPUs
    c.node('l1_qkv0', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 0', shape='rectangle', style='filled', fillcolor=colors[0])
    c.node('l1_qkv1', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 1', shape='rectangle', style='filled', fillcolor=colors[1])
    c.node('l1_qkv2', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 2', shape='rectangle', style='filled', fillcolor=colors[2])
    c.node('l1_qkv3', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 3', shape='rectangle', style='filled', fillcolor=colors[3])
    c.node('l1_qkv4', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 4', shape='rectangle', style='filled', fillcolor=colors[4])
    c.node('l1_qkv5', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 5', shape='rectangle', style='filled', fillcolor=colors[5])
    c.node('l1_qkv6', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 6', shape='rectangle', style='filled', fillcolor=colors[6])
    c.node('l1_qkv7', 'QKV Linear\nInput: [1024,10000,8192]\nOutput: [1024,10000,96,128]\nGPU: 7', shape='rectangle', style='filled', fillcolor=colors[7])
    
    c.node('l1_attention', 'Multi-Head Attention\nInput: [1024,10000,96,128]×8\nOutput: [1024,10000,8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightsteelblue')
    c.node('l1_residual1', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nGPU: 0', shape='parallelogram', style='filled', fillcolor='white')
    
    # MoE Layer - 8 experts on 8 GPUs (1 expert per GPU)
    c.node('l1_gate', 'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: 0', shape='parallelogram', style='filled', fillcolor=colors[0])
    
    for i in range(8):
        c.node(f'l1_expert{i}', f'Expert {i}\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {i}', shape='rectangle', style='filled', fillcolor=colors[i])
    
    c.node('l1_moe_agg', 'MoE Aggregate\nInput: [1024,10000,8192]×8\nOutput: [1024,10000,8192]\nGPU: 0', shape='parallelogram', style='filled', fillcolor='white')
    c.node('l1_residual2', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nGPU: 0', shape='parallelogram', style='filled', fillcolor='white')

# Communication between pipeline stages
dot.node('pipeline_comm', 'Pipeline Communication\n[1024,10000,8192]\nGPU: 0→8', shape='ellipse', style='dashed', fillcolor='yellow')

# Layer 1 - Second pipeline stage (GPUs 8-15)
with dot.subgraph(name='cluster_layer1_stage2') as c:
    c.attr(label='Layer 1 - Pipeline Stage 2 (GPUs 8-15)', style='dashed')
    
    # Similar structure for second stage
    c.node('l1_ln1_stage2', 'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: 8', shape='rectangle', style='filled', fillcolor=colors[0])
    
    # MoE Layer - 8 experts on 8 GPUs
    c.node('l1_gate_stage2', 'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: 8', shape='parallelogram', style='filled', fillcolor=colors[0])
    
    for i in range(8, 16):
        c.node(f'l1_expert{i}', f'Expert {i}\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {i}', shape='rectangle', style='filled', fillcolor=colors[i-8])
    
    c.node('l1_moe_agg_stage2', 'MoE Aggregate\nInput: [1024,10000,8192]×8\nOutput: [1024,10000,8192]\nGPU: 8', shape='parallelogram', style='filled', fillcolor='white')
    c.node('l1_residual2_stage2', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nGPU: 8', shape='parallelogram', style='filled', fillcolor='white')

# Repeat for layers 2, 3, 4 (simplified representation)
for layer in range(2, 5):
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer} (Similar Structure)', style='dashed')
        c.node(f'layer{layer}', f'Layer {layer}\nSame as Layer 1\n16 GPUs total', shape='rectangle', style='filled', fillcolor='lightgray')

# Output node
dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, hidden_size=8192]', shape='ellipse', style='filled', fillcolor='white')

# Connect nodes
# Layer 1 connections
dot.edge('input', 'l1_ln0')
dot.edge('l1_ln0', 'l1_qkv0')
dot.edge('l1_ln0', 'l1_qkv1')
dot.edge('l1_ln0', 'l1_qkv2')
dot.edge('l1_ln0', 'l1_qkv3')
dot.edge('l1_ln0', 'l1_qkv4')
dot.edge('l1_ln0', 'l1_qkv5')
dot.edge('l1_ln0', 'l1_qkv6')
dot.edge('l1_ln0', 'l1_qkv7')
dot.edge('l1_qkv0', 'l1_attention')
dot.edge('l1_qkv1', 'l1_attention')
dot.edge('l1_qkv2', 'l1_attention')
dot.edge('l1_qkv3', 'l1_attention')
dot.edge('l1_qkv4', 'l1_attention')
dot.edge('l1_qkv5', 'l1_attention')
dot.edge('l1_qkv6', 'l1_attention')
dot.edge('l1_qkv7', 'l1_attention')
dot.edge('l1_attention', 'l1_residual1')
dot.edge('input', 'l1_residual1')  # Residual connection

dot.edge('l1_residual1', 'l1_gate')
for i in range(8):
    dot.edge('l1_gate', f'l1_expert{i}', style='dashed')
    dot.edge('l1_residual1', f'l1_expert{i}')
    dot.edge(f'l1_expert{i}', 'l1_moe_agg')

dot.edge('l1_moe_agg', 'l1_residual2')
dot.edge('l1_residual1', 'l1_residual2')  # Residual connection

# Pipeline communication
dot.edge('l1_residual2', 'pipeline_comm')
dot.edge('pipeline_comm', 'l1_ln1_stage2')

# Layer 1 stage 2 connections
dot.edge('l1_ln1_stage2', 'l1_gate_stage2')
for i in range(8, 16):
    dot.edge('l1_gate_stage2', f'l1_expert{i}', style='dashed')
    dot.edge('l1_ln1_stage2', f'l1_expert{i}')
    dot.edge(f'l1_expert{i}', 'l1_moe_agg_stage2')

dot.edge('l1_moe_agg_stage2', 'l1_residual2_stage2')
dot.edge('l1_ln1_stage2', 'l1_residual2_stage2')  # Residual connection

# Connect through layers 2-4 (simplified)
dot.edge('l1_residual2_stage2', 'layer2')
dot.edge('layer2', 'layer3')
dot.edge('layer3', 'layer4')
dot.edge('layer4', 'output')

# Save files
dot.render('./outputs/2025-10-13-16-10-29/baseline_dag', format='dot')
dot.render('./outputs/2025-10-13-16-10-29/baseline_dag', format='svg')

print("Baseline DAG generated successfully")