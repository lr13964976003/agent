import graphviz

# Create proposed layer-wise deployment DAG with 1 layer per GPU
dot = graphviz.Digraph(comment='Proposed Layer-wise Deployment (16 GPUs)', format='svg')
dot.attr(rankdir='TB', size='25,20', fontname='Arial')

# Define consistent attributes
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='Legend', style='dashed', color='gray')
    c.node('legend_compute', 'Computation', shape='rectangle', style='filled', color='lightblue')
    c.node('legend_comm', 'Communication', shape='ellipse', style='filled', color='lightgreen')
    c.node('legend_route', 'Routing/Aggregation', shape='parallelogram', style='filled', color='lightyellow')
    c.node('legend_split', 'Split', shape='parallelogram', style='filled', color='orange')
    c.node('legend_gather', 'Gather', shape='parallelogram', style='filled', color='purple')

# Input layer
dot.node('input', 'Input Embedding\nInput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-0\nLayer: Embedding+Pre-process', 
         shape='parallelogram', style='filled', color='lightyellow')

# Create clusters for each GPU and its layer
for gpu_id in range(16):
    layer_id = gpu_id
    
    with dot.subgraph(name=f'cluster_gpu{gpu_id}') as c:
        c.attr(label=f'GPU-{gpu_id} (Layer {layer_id})', style='rounded', color=f'color{gpu_id%8+1}')
        
        # Multi-Head Attention Q computation
        q_label = f'GPU{gpu_id}_Layer{layer_id}_MHA_Q\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: GPU-{gpu_id}'
        c.node(f'mha_q_{layer_id}', q_label, shape='rectangle', style='filled', color='lightblue')
        
        # Multi-Head Attention K computation
        k_label = f'GPU{gpu_id}_Layer{layer_id}_MHA_K\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: GPU-{gpu_id}'
        c.node(f'mha_k_{layer_id}', k_label, shape='rectangle', style='filled', color='lightblue')
        
        # Multi-Head Attention V computation
        v_label = f'GPU{gpu_id}_Layer{layer_id}_MHA_V\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: GPU-{gpu_id}'
        c.node(f'mha_v_{layer_id}', v_label, shape='rectangle', style='filled', color='lightblue')
        
        # Attention score computation
        score_label = f'GPU{gpu_id}_Layer{layer_id}_Attn_Score\nInput1: [batch=128, seq=10000, heads=32, d_k=128]\nInput2: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, seq=10000]\nDevice: GPU-{gpu_id}'
        c.node(f'attn_score_{layer_id}', score_label, shape='rectangle', style='filled', color='lightblue')
        
        # Attention output projection
        attn_out_label = f'GPU{gpu_id}_Layer{layer_id}_Attn_Out\nInput1: [batch=128, seq=10000, heads=32, seq=10000]\nInput2: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'attn_out_{layer_id}', attn_out_label, shape='rectangle', style='filled', color='lightblue')
        
        # Attention residual add
        attn_residual_label = f'GPU{gpu_id}_Layer{layer_id}_Attn_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'attn_residual_{layer_id}', attn_residual_label, shape='parallelogram', style='filled', color='lightyellow')
        
        # LayerNorm after attention
        layernorm1_label = f'GPU{gpu_id}_Layer{layer_id}_LayerNorm1\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'layernorm1_{layer_id}', layernorm1_label, shape='rectangle', style='filled', color='lightblue')
        
        # FFN Up projection
        ffn_up_label = f'GPU{gpu_id}_Layer{layer_id}_FFN_Up\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: GPU-{gpu_id}'
        c.node(f'ffn_up_{layer_id}', ffn_up_label, shape='rectangle', style='filled', color='lightblue')
        
        # FFN Gate projection
        ffn_gate_label = f'GPU{gpu_id}_Layer{layer_id}_FFN_Gate\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: GPU-{gpu_id}'
        c.node(f'ffn_gate_{layer_id}', ffn_gate_label, shape='rectangle', style='filled', color='lightblue')
        
        # FFN activation (GELU)
        ffn_act_label = f'GPU{gpu_id}_Layer{layer_id}_FFN_Act\nInput: [batch=128, seq=10000, ffn=16384]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: GPU-{gpu_id}'
        c.node(f'ffn_act_{layer_id}', ffn_act_label, shape='rectangle', style='filled', color='lightblue')
        
        # FFN Down projection
        ffn_down_label = f'GPU{gpu_id}_Layer{layer_id}_FFN_Down\nInput: [batch=128, seq=10000, ffn=16384]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'ffn_down_{layer_id}', ffn_down_label, shape='rectangle', style='filled', color='lightblue')
        
        # FFN residual add
        ffn_residual_label = f'GPU{gpu_id}_Layer{layer_id}_FFN_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'ffn_residual_{layer_id}', ffn_residual_label, shape='parallelogram', style='filled', color='lightyellow')
        
        # LayerNorm after FFN
        layernorm2_label = f'GPU{gpu_id}_Layer{layer_id}_LayerNorm2\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: GPU-{gpu_id}'
        c.node(f'layernorm2_{layer_id}', layernorm2_label, shape='rectangle', style='filled', color='lightblue')

# Communication nodes between GPUs
for gpu_id in range(15):  # Only need 15 communication nodes for 16 GPUs
    comm_label = f'GPU{gpu_id}_to_GPU{gpu_id+1}\nTransfer: [batch=128, seq=10000, hidden=4096]\nBandwidth: NVLink 900GB/s'
    dot.node(f'comm_{gpu_id}', comm_label, shape='ellipse', style='filled', color='lightgreen')

# Output layer (final projection)
dot.node('output', 'Final Output Projection\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, vocab_size=128256]\nDevice: GPU-15', 
         shape='parallelogram', style='filled', color='purple')

# Connect the flow through all layers
prev_node = 'input'

for layer_id in range(16):
    gpu_id = layer_id
    
    # Attention computation flow
    dot.edge(prev_node, f'mha_q_{layer_id}')
    dot.edge(prev_node, f'mha_k_{layer_id}')
    dot.edge(prev_node, f'mha_v_{layer_id}')
    
    # Attention score computation
    dot.edge(f'mha_q_{layer_id}', f'attn_score_{layer_id}')
    dot.edge(f'mha_k_{layer_id}', f'attn_score_{layer_id}')
    
    # Attention output
    dot.edge(f'attn_score_{layer_id}', f'attn_out_{layer_id}')
    dot.edge(f'mha_v_{layer_id}', f'attn_out_{layer_id}')
    
    # Attention residual
    dot.edge(f'attn_out_{layer_id}', f'attn_residual_{layer_id}')
    dot.edge(prev_node, f'attn_residual_{layer_id}')  # Residual connection
    
    # LayerNorm after attention
    dot.edge(f'attn_residual_{layer_id}', f'layernorm1_{layer_id}')
    
    # FFN computation
    dot.edge(f'layernorm1_{layer_id}', f'ffn_up_{layer_id}')
    dot.edge(f'layernorm1_{layer_id}', f'ffn_gate_{layer_id}')
    dot.edge(f'ffn_up_{layer_id}', f'ffn_act_{layer_id}')
    dot.edge(f'ffn_gate_{layer_id}', f'ffn_act_{layer_id}')
    dot.edge(f'ffn_act_{layer_id}', f'ffn_down_{layer_id}')
    
    # FFN residual
    dot.edge(f'ffn_down_{layer_id}', f'ffn_residual_{layer_id}')
    dot.edge(f'layernorm1_{layer_id}', f'ffn_residual_{layer_id}')  # Residual connection
    
    # LayerNorm after FFN
    dot.edge(f'ffn_residual_{layer_id}', f'layernorm2_{layer_id}')
    
    # Communication to next GPU
    if layer_id < 15:
        dot.edge(f'layernorm2_{layer_id}', f'comm_{layer_id}')
        prev_node = f'comm_{layer_id}'
    else:
        # Last layer connects to output
        dot.edge(f'layernorm2_{layer_id}', 'output')

# Save files
dot.render('../outputs/2025-11-21-15-21-34/proposed_dag', format='dot')
dot.render('../outputs/2025-11-21-15-21-34/proposed_dag', format='svg')

print("Proposed layer-wise DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-11-21-15-21-34/proposed_dag.dot")
print("- ../outputs/2025-11-21-15-21-34/proposed_dag.svg")