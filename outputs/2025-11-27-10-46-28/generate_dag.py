import graphviz
import os

# Create DAG for large-scale MoE deployment with integrated parallel strategies
# Representing 3 representative layers: 1 dense layer and 2 MoE layers with integrated DP, TP, PP

dot = graphviz.Digraph(comment='Large-Scale MoE Cross-Node Expert Parallelism DAG')
dot.attr(rankdir='false')

# Node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
dot.attr('edge', style='solid')

# Define tensor shapes
batch_size = 4
seq_len = 2048
hidden_dim = 7168
vocab_size = 128256  # Typical size for large models
num_heads = 128
head_dim = 56
ffn_hidden = 18432
num_experts = 16

# Input node
dot.node('input', f'''Input Layer\nGPU: META\nInput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='ellipse', fillcolor='lightgreen')

# Dense Layer 1 (Embedding)
# DP splits batch across GPUs
dot.node('dense1_dp_split', f'''DP Split\nGPU: META\nInput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, vocab_size={vocab_size}]''', 
         shape='parallelogram', fillcolor='yellow')

# TP for embedding layer across 2 GPUs
dot.node('embed_gpu0', f'''Embedding\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, vocab_size={vocab_size}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]''', 
         shape='rectangle', fillcolor='lightblue')

dot.node('embed_gpu1', f'''Embedding\nGPU: 1\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, vocab_size={vocab_size}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]''', 
         shape='rectangle', fillcolor='lightblue')

dot.node('embed_concat', f'''TP Concat\nGPU: 0-1\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim//2}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='parallelogram', fillcolor='yellow')

# Dense Layer 2 (LayerNorm + Multi-Head Attention)
dot.node('ln1_gpu0', f'''LayerNorm\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='rectangle', fillcolor='lightblue')

dot.node('qkv_proj_gpu0', f'''QKV Projection\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim*3}]''', 
         shape='rectangle', fillcolor='lightblue')

# MLA computation across multiple GPUs
dot.node('mla_split', f'''Head Split\nGPU: 0-7\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim*3}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, heads_per_gpu={num_heads//8}, head_dim={head_dim*3}]''', 
         shape='parallelogram', fillcolor='yellow')

# Attention heads distributed across GPUs 0-7
for i in range(8):
    dot.node(f'mha_gpu{i}', f'''MHA Head Group\nGPU: {i}\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, heads_per_gpu={num_heads//8}, head_dim={head_dim*3}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, heads_per_gpu={num_heads//8}, head_dim={head_dim}]''', 
             shape='rectangle', fillcolor='lightblue')

# Attention aggregation
dot.node('attn_concat', f'''Attention Concat\nGPU: 0-7\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, heads_per_gpu={num_heads//8}, head_dim={head_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='parallelogram', fillcolor='yellow')

# Output projection
dot.node('out_proj_gpu0', f'''Output Projection\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='rectangle', fillcolor='lightblue')

# Residual connection
dot.node('residual1', f'''Residual Add\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='rectangle', fillcolor='lightblue')

# MoE Layer 1 - Expert routing and selection
dot.node('moe_ln1', f'''Pre-MoE LayerNorm\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='rectangle', fillcolor='lightblue')

# Gate computation
dot.node('gate_gpu0', f'''Expert Gate\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, num_experts={num_experts}]''', 
         shape='rectangle', fillcolor='lightblue')

# Expert selection and routing
dot.node('expert_select', f'''Expert Selection\nGPU: 0\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, num_experts={num_experts}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, selected_experts=2]''', 
         shape='parallelogram', fillcolor='yellow')

# Expert distribution - 16 experts across 16 GPUs
experts_gpus = list(range(100, 116))  # GPU IDs for experts 0-15
for i, gpu_id in enumerate(experts_gpus):
    dot.node(f'expert_{i}', f'''Expert {i} MLP\nGPU: {gpu_id}\nInput: [tokens_per_expert=variable, hidden_dim={hidden_dim}]\nOutput: [tokens_per_expert=variable, hidden_dim={hidden_dim}]''', 
             shape='rectangle', fillcolor='lightblue')

# Token routing to experts (dashed lines for communication)
for i, gpu_id in enumerate(experts_gpus):
    dot.edge('expert_select', f'expert_{i}', style='dashed', label=f'route tokens to expert {i}')

# Expert outputs aggregation
dot.node('expert_agg', f'''Expert Output Aggregation\nGPU: 116\nInput: [tokens_from_experts=variable, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='parallelogram', fillcolor='yellow')

# MoE Layer 2 - Second MoE layer
dot.node('moe2_ln1', f'''Pre-MoE2 LayerNorm\nGPU: 116\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='rectangle', fillcolor='lightblue')

dot.node('gate2_gpu116', f'''Expert Gate 2\nGPU: 116\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, num_experts={num_experts}]''', 
         shape='rectangle', fillcolor='lightblue')

# Second MoE expert selection
dot.node('expert2_select', f'''Expert Selection 2\nGPU: 116\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, num_experts={num_experts}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, selected_experts=2]''', 
         shape='parallelogram', fillcolor='yellow')

# Second set of experts on different GPUs
experts2_gpus = list(range(200, 216))  # GPU IDs for second layer experts
for i, gpu_id in enumerate(experts2_gpus):
    dot.node(f'expert2_{i}', f'''Expert {i} MLP 2\nGPU: {gpu_id}\nInput: [tokens_per_expert=variable, hidden_dim={hidden_dim}]\nOutput: [tokens_per_expert=variable, hidden_dim={hidden_dim}]''', 
             shape='rectangle', fillcolor='lightblue')

# Second layer routing (dashed lines)
for i, gpu_id in enumerate(experts2_gpus):
    dot.edge('expert2_select', f'expert2_{i}', style='dashed', label=f'route tokens to expert2 {i}')

# Final aggregation
dot.node('expert2_agg', f'''Expert Output Aggregation 2\nGPU: 216\nInput: [tokens_from_experts=variable, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='parallelogram', fillcolor='yellow')

# DP aggregation
dot.node('dp_agg', f'''DP Aggregation\nGPU: META\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]''', 
         shape='parallelogram', fillcolor='yellow')

# Output layer
dot.node('output', f'''Output Layer\nGPU: META\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]''', 
         shape='ellipse', fillcolor='lightgreen')

# Connect all nodes
# Input flow
dot.edge('input', 'dense1_dp_split')
dot.edge('dense1_dp_split', 'embed_gpu0')
dot.edge('dense1_dp_split', 'embed_gpu1')
dot.edge('embed_gpu0', 'embed_concat')
dot.edge('embed_gpu1', 'embed_concat')
dot.edge('embed_concat', 'ln1_gpu0')

# Attention flow
dot.edge('ln1_gpu0', 'qkv_proj_gpu0')
dot.edge('qkv_proj_gpu0', 'mla_split')
for i in range(8):
    dot.edge('mla_split', f'mha_gpu{i}')
for i in range(8):
    dot.edge(f'mha_gpu{i}', 'attn_concat')
dot.edge('attn_concat', 'out_proj_gpu0')
dot.edge('out_proj_gpu0', 'residual1')

# MoE Layer 1 flow
dot.edge('residual1', 'moe_ln1')
dot.edge('moe_ln1', 'gate_gpu0')
dot.edge('gate_gpu0', 'expert_select')
for i, gpu_id in enumerate(experts_gpus):
    dot.edge('expert_select', f'expert_{i}')
    dot.edge(f'expert_{i}', 'expert_agg')
dot.edge('expert_agg', 'moe2_ln1')

# MoE Layer 2 flow
dot.edge('moe2_ln1', 'gate2_gpu116')
dot.edge('gate2_gpu116', 'expert2_select')
for i, gpu_id in enumerate(experts2_gpus):
    dot.edge('expert2_select', f'expert2_{i}')
    dot.edge(f'expert2_{i}', 'expert2_agg')
dot.edge('expert2_agg', 'dp_agg')
dot.edge('dp_agg', 'output')

# Save the DAG
dot.format = 'svg'
dot.render('../outputs/2025-11-27-10-46-28/moe_deployment_dag', cleanup=False)

# Also save as DOT file
dot.save('../outputs/2025-11-27-10-46-28/moe_deployment_dag.dot')

# Create a more detailed version with communication patterns
detailed_dot = graphviz.Digraph(comment='Detailed MoE Parallelism with Communication')

# Add subgraphs for better organization
with detailed_dot.subgraph(name='cluster_dp') as dp:
    dp.attr(label='Data Parallelism')
    dp.node('dp_split', 'DP Split', shape='parallelogram')
    dp.node('dp_agg', 'DP Aggregate', shape='parallelogram')

with detailed_dot.subgraph(name='cluster_tp') as tp:
    tp.attr(label='Tensor Parallelism')
    tp.node('tp_split', 'TP Split', shape='parallelogram')
    tp.node('tp_concat', 'TP Concat', shape='parallelogram')

with detailed_dot.subgraph(name='cluster_ep') as ep:
    ep.attr(label='Expert Parallelism')
    for i in range(16):
        ep.node(f'ep_expert_{i}', f'Expert {i}', shape='rectangle')

# Save detailed version
detailed_dot.save('../outputs/2025-11-27-10-46-28/moe_detailed_dag.dot')

print("DAG files generated:")
print("- moe_deployment_dag.svg")
print("- moe_deployment_dag.dot")
print("- moe_detailed_dag.dot")