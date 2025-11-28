import graphviz
from graphviz import Digraph

# Create a new directed graph with horizontal layout
dot = Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
dot.attr(rankdir='LR')  # Left to right layout
dot.attr('node', fontname='Arial', fontsize='10')

# Input layer - distributing tokens across GPUs
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Distribution Layer')
    c.node('input_tokens', 'Input Tokens\\n[batch_size=?, seq_len=?]\\nGPU: N/A', 
           shape='egg', style='filled', fillcolor='lightgray')
    c.node('token_split', 'Token Split\\n[batch_size=?, seq_len=?] → [batch_size=?, seq_len=?/256]\\nGPU: 0-255', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    c.edge('input_tokens', 'token_split', label='broadcast')

# Layer 1: Attention + Routing (representative layer)
with dot.subgraph(name='cluster_layer1') as c:
    c.attr(label='Layer 1: MLA + Expert Routing')
    
    # MLA computation on GPU 0
    c.node('mla_0', 'MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Routing/Gating on GPU 0
    c.node('gate_0', 'Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 0', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Add residual connection input
    c.node('residual_add_0', 'Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    c.edge('token_split', 'mla_0', label='tokens for GPU 0')
    c.edge('mla_0', 'gate_0')
    c.edge('gate_0', 'residual_add_0')
    c.edge('token_split', 'residual_add_0', style='dashed', label='residual')

# Expert routing and communication to experts
with dot.subgraph(name='cluster_routing') as c:
    c.attr(label='Expert Routing Communication')
    
    # Communication from gate to experts
    for i in [0, 64, 128, 192, 255]:  # Show representative experts
        c.node(f'comm_to_expert_{i}', f'Token Routing\\n[batch_size=?, seq_len=?, dim=7168]\\n→ GPU: {i}', 
               shape='ellipse', style='filled', fillcolor='lightyellow')
        if i == 0:
            c.edge('residual_add_0', f'comm_to_expert_{i}')
        
        # Expert computation nodes
        c.node(f'expert_{i}', f'Expert MLP {i}\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: {i}', 
               shape='rectangle', style='filled', fillcolor='lightblue')
        c.edge(f'comm_to_expert_{i}', f'expert_{i}')

# Layer 2: Intermediate layer (representative)
with dot.subgraph(name='cluster_layer2') as c:
    c.attr(label='Layer 2: Intermediate Processing')
    
    # Show MLA on GPU 64
    c.node('mla_64', 'MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 64', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert gating on GPU 64
    c.node('gate_64', 'Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 64', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    c.node('residual_add_64', 'Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 64', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Connect from expert 64
    c.edge('expert_64', 'mla_64', label='processed tokens')
    c.edge('mla_64', 'gate_64')
    c.edge('gate_64', 'residual_add_64')
    c.edge('expert_64', 'residual_add_64', style='dashed', label='residual')

# Layer 3: Output layer (representative)
with dot.subgraph(name='cluster_layer3') as c:
    c.attr(label='Layer 3: Output Processing')
    
    # Final processing on GPU 255
    c.node('mla_255', 'MLA\\n[batch_size=?, seq_len=?, heads=128, d_k=56]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 255', 
           shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Final routing
    c.node('gate_255', 'Expert Gating\\n[batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, top_k=2]\\nGPU: 255', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    c.node('residual_add_255', 'Residual Add\\n[batch_size=?, seq_len=?, dim=7168] + [batch_size=?, seq_len=?, dim=7168]\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 255', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Connect from expert 255
    c.edge('expert_255', 'mla_255', label='processed tokens')
    c.edge('mla_255', 'gate_255')
    c.edge('gate_255', 'residual_add_255')
    c.edge('expert_255', 'residual_add_255', style='dashed', label='residual')

# Aggregation and final output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Aggregation')
    
    # Collect results from all experts
    c.node('collect_res', 'Gather Results\\n[batch_size=?, seq_len=?, dim=7168] from all GPUs\\n→ [batch_size=?, seq_len=?, dim=7168]\\nGPU: 0-255', 
           shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Final aggregation
    c.node('final_out', 'Final Output\\n[batch_size=?, seq_len=?, dim=7168]\\nGPU: 0', 
           shape='egg', style='filled', fillcolor='lightgray')
    
    c.edge('residual_add_0', 'collect_res', label='from GPU 0')
    c.edge('residual_add_64', 'collect_res', label='from GPU 64')
    c.edge('residual_add_255', 'collect_res', label='from GPU 255')
    c.edge('collect_res', 'final_out')

# Add communication edges between representative experts
with dot.subgraph(name='cluster_communication') as c:
    c.attr(label='Cross-Node Communication')
    
    # Show inter-expert communication
    c.node('comm_0_64', 'Async Transfer\\nGPU: 0 ↔ GPU: 64', 
           shape='ellipse', style='filled', fillcolor='lightyellow')
    c.node('comm_64_255', 'Async Transfer\\nGPU: 64 ↔ GPU: 255', 
           shape='ellipse', style='filled', fillcolor='lightyellow')
    
    c.edge('gate_0', 'comm_0_64', style='dashed', label='token routing')
    c.edge('gate_64', 'comm_64_255', style='dashed', label='token routing')

# Save the DOT file
dot.save('../outputs/2025-11-28-16-11-17/large_ep_dag.dot')

# Generate SVG
dot.render('../outputs/2025-11-28-16-11-17/large_ep_dag', format='svg', cleanup=True)

print("DAG generated successfully!")
print("DOT file: ../outputs/2025-11-28-16-11-17/large_ep_dag.dot")
print("SVG file: ../outputs/2025-11-28-16-11-17/large_ep_dag.svg")