import graphviz

# Create proposed DAG with EP=16 (1 expert per GPU)
dot = graphviz.Digraph(comment='Proposed MoE DAG (EP=16)', format='svg')
dot.attr(rankdir='TB', size='30,30')

# Define colors for different GPUs
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan', 'lightgray', 'lightseagreen',
          'lightsteelblue', 'lightgoldenrod', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightcoral', 'lightgreen', 'lightyellow']

# Input node
dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, hidden_size=8192]', shape='ellipse', style='filled', fillcolor='white')

# Layer 1 - Full expert parallelism across 16 GPUs
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1 - Expert Parallelism (16 GPUs)', style='dashed')
    
    # LayerNorm - replicated on all GPUs
    for gpu in range(16):
        c.node(f'l1_ln_{gpu}', f'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
    
    # Multi-Head Attention - tensor parallel across 16 GPUs (simplified)
    for gpu in range(16):
        c.node(f'l1_mha_{gpu}', f'MHA Part\nInput: [1024,10000,8192]\nOutput: [1024,10000,512]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
    
    c.node('l1_mha_agg', 'MHA Aggregate\nInput: [1024,10000,512]×16\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='lightsteelblue')
    
    # Residual connection after attention
    c.node('l1_residual1', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')
    
    # Gate - determines routing
    for gpu in range(16):
        c.node(f'l1_gate_{gpu}', f'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: {gpu}', 
               shape='parallelogram', style='filled', fillcolor=colors[gpu])
    
    # Expert routing and computation - 1 expert per GPU
    for expert_id in range(16):
        gpu = expert_id
        c.node(f'l1_expert{expert_id}', f'Expert {expert_id}\nInput: [tokens_per_expert,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        
        # Token routing from gate to expert
        c.node(f'l1_route_{expert_id}', f'Route Tokens\nInput: [1024,10000,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
        
        # Token aggregation from expert
        c.node(f'l1_gather_{expert_id}', f'Gather Tokens\nInput: [tokens_per_expert,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
    
    # MoE aggregation
    c.node('l1_moe_agg', 'MoE Aggregate\nInput: [1024,10000,8192]×16\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')
    
    # Final residual connection
    c.node('l1_residual2', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')

# Layer 2 - Similar structure
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2 - Expert Parallelism (16 GPUs)', style='dashed')
    
    for gpu in range(16):
        c.node(f'l2_ln_{gpu}', f'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l2_gate_{gpu}', f'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: {gpu}', 
               shape='parallelogram', style='filled', fillcolor=colors[gpu])
    
    for expert_id in range(16):
        gpu = expert_id
        c.node(f'l2_expert{expert_id}', f'Expert {expert_id+16}\nInput: [tokens_per_expert,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l2_route_{expert_id}', f'Route Tokens\nInput: [1024,10000,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
        c.node(f'l2_gather_{expert_id}', f'Gather Tokens\nInput: [tokens_per_expert,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
    
    c.node('l2_moe_agg', 'MoE Aggregate\nInput: [1024,10000,8192]×16\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')
    c.node('l2_residual', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')

# Layer 3 - Similar structure
with dot.subgraph(name='cluster_layer3') as c:
    c.attr(label='Layer 3 - Expert Parallelism (16 GPUs)', style='dashed')
    
    for gpu in range(16):
        c.node(f'l3_ln_{gpu}', f'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l3_gate_{gpu}', f'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: {gpu}', 
               shape='parallelogram', style='filled', fillcolor=colors[gpu])
    
    for expert_id in range(16):
        gpu = expert_id
        c.node(f'l3_expert{expert_id}', f'Expert {expert_id+32}\nInput: [tokens_per_expert,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l3_route_{expert_id}', f'Route Tokens\nInput: [1024,10000,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
        c.node(f'l3_gather_{expert_id}', f'Gather Tokens\nInput: [tokens_per_expert,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
    
    c.node('l3_moe_agg', 'MoE Aggregate\nInput: [1024,10000,8192]×16\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')
    c.node('l3_residual', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')

# Layer 4 - Similar structure
with dot.subgraph(name='cluster_layer4') as c:
    c.attr(label='Layer 4 - Expert Parallelism (16 GPUs)', style='dashed')
    
    for gpu in range(16):
        c.node(f'l4_ln_{gpu}', f'LayerNorm\nInput: [1024,10000,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l4_gate_{gpu}', f'Gate\nInput: [1024,10000,8192]\nOutput: [1024,10000,16]\nGPU: {gpu}', 
               shape='parallelogram', style='filled', fillcolor=colors[gpu])
    
    for expert_id in range(16):
        gpu = expert_id
        c.node(f'l4_expert{expert_id}', f'Expert {expert_id+48}\nInput: [tokens_per_expert,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='rectangle', style='filled', fillcolor=colors[gpu])
        c.node(f'l4_route_{expert_id}', f'Route Tokens\nInput: [1024,10000,8192]\nOutput: [tokens_per_expert,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
        c.node(f'l4_gather_{expert_id}', f'Gather Tokens\nInput: [tokens_per_expert,8192]\nOutput: [1024,10000,8192]\nGPU: {gpu}', 
               shape='ellipse', style='dashed', fillcolor=colors[gpu])
    
    c.node('l4_moe_agg', 'MoE Aggregate\nInput: [1024,10000,8192]×16\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')
    c.node('l4_residual', 'Residual Add\nInput: [1024,10000,8192]×2\nOutput: [1024,10000,8192]\nAll GPUs', 
           shape='parallelogram', style='filled', fillcolor='white')

# Output node
dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, hidden_size=8192]', shape='ellipse', style='filled', fillcolor='white')

# Connect Layer 1
dot.edge('input', 'l1_ln_0')
for gpu in range(16):
    if gpu > 0:
        dot.edge('input', f'l1_ln_{gpu}')
    dot.edge(f'l1_ln_{gpu}', f'l1_mha_{gpu}')
    dot.edge(f'l1_mha_{gpu}', 'l1_mha_agg')

dot.edge('l1_mha_agg', 'l1_residual1')
dot.edge('input', 'l1_residual1')  # Residual connection

dot.edge('l1_residual1', 'l1_gate_0')
for gpu in range(16):
    if gpu > 0:
        dot.edge('l1_residual1', f'l1_gate_{gpu}')
    dot.edge(f'l1_gate_{gpu}', f'l1_route_{gpu}', style='dashed')
    dot.edge('l1_residual1', f'l1_route_{gpu}')
    dot.edge(f'l1_route_{gpu}', f'l1_expert{gpu}')
    dot.edge(f'l1_expert{gpu}', f'l1_gather_{gpu}')
    dot.edge(f'l1_gather_{gpu}', 'l1_moe_agg')

dot.edge('l1_moe_agg', 'l1_residual2')
dot.edge('l1_residual1', 'l1_residual2')  # Residual connection

# Connect Layer 2
dot.edge('l1_residual2', 'l2_ln_0')
for gpu in range(16):
    if gpu > 0:
        dot.edge('l1_residual2', f'l2_ln_{gpu}')
    dot.edge(f'l2_ln_{gpu}', f'l2_gate_{gpu}')
    dot.edge(f'l2_gate_{gpu}', f'l2_route_{gpu}', style='dashed')
    dot.edge('l2_ln_{gpu}', f'l2_route_{gpu}')
    dot.edge(f'l2_route_{gpu}', f'l2_expert{gpu}')
    dot.edge(f'l2_expert{gpu}', f'l2_gather_{gpu}')
    dot.edge(f'l2_gather_{gpu}', 'l2_moe_agg')

dot.edge('l2_moe_agg', 'l2_residual')
dot.edge('l2_ln_0', 'l2_residual')  # Residual connection (simplified)

# Connect Layer 3
dot.edge('l2_residual', 'l3_ln_0')
for gpu in range(16):
    if gpu > 0:
        dot.edge('l2_residual', f'l3_ln_{gpu}')
    dot.edge(f'l3_ln_{gpu}', f'l3_gate_{gpu}')
    dot.edge(f'l3_gate_{gpu}', f'l3_route_{gpu}', style='dashed')
    dot.edge('l3_ln_{gpu}', f'l3_route_{gpu}')
    dot.edge(f'l3_route_{gpu}', f'l3_expert{gpu}')
    dot.edge(f'l3_expert{gpu}', f'l3_gather_{gpu}')
    dot.edge(f'l3_gather_{gpu}', 'l3_moe_agg')

dot.edge('l3_moe_agg', 'l3_residual')
dot.edge('l3_ln_0', 'l3_residual')  # Residual connection (simplified)

# Connect Layer 4
dot.edge('l3_residual', 'l4_ln_0')
for gpu in range(16):
    if gpu > 0:
        dot.edge('l3_residual', f'l4_ln_{gpu}')
    dot.edge(f'l4_ln_{gpu}', f'l4_gate_{gpu}')
    dot.edge(f'l4_gate_{gpu}', f'l4_route_{gpu}', style='dashed')
    dot.edge('l4_ln_{gpu}', f'l4_route_{gpu}')
    dot.edge(f'l4_route_{gpu}', f'l4_expert{gpu}')
    dot.edge(f'l4_expert{gpu}', f'l4_gather_{gpu}')
    dot.edge(f'l4_gather_{gpu}', 'l4_moe_agg')

dot.edge('l4_moe_agg', 'l4_residual')
dot.edge('l4_ln_0', 'l4_residual')  # Residual connection (simplified)
dot.edge('l4_residual', 'output')

# Save files
dot.render('./outputs/2025-10-13-16-10-29/proposed_dag', format='dot')
dot.render('./outputs/2025-10-13-16-10-29/proposed_dag', format='svg')

print("Proposed DAG generated successfully")