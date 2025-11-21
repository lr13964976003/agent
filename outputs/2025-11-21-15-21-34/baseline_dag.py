import graphviz

# Create baseline DAG for TP=8, PP=2 deployment
dot = graphviz.Digraph(comment='Baseline Model Deployment (TP=8, PP=2)', format='svg')
dot.attr(rankdir='TB', size='20,15', fontname='Arial')

# Define consistent attributes
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='Legend', style='dashed', color='gray')
    c.node('legend_compute', 'Computation', shape='rectangle', style='filled', color='lightblue')
    c.node('legend_comm', 'Communication', shape='ellipse', style='filled', color='lightgreen')
    c.node('legend_route', 'Routing/Aggregation', shape='parallelogram', style='filled', color='lightyellow')
    c.node('legend_gather', 'All-Gather', shape='ellipse', style='filled', color='orange')
    c.node('legend_reduce', 'All-Reduce', shape='ellipse', style='filled', color='red')

# Pipeline Stage 0 (Devices 0-7)
with dot.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 (Devices 0-7)', style='rounded')
    
    # Input to Stage 0
    c.node('input', 'Input Embedding\nInput: [batch=128, seq=10000, hidden=4096]\nDevice: All GPUs\nLayer: Embedding', 
           shape='parallelogram', style='filled', color='lightyellow')
    
    # Layers 0-7 distributed with TP=8
    for layer in range(8):
        with c.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (TP=8 across devices 0-7)', style='dashed')
            
            # Multi-Head Attention within layer
            attn_label = f'Layer{layer}_MHA_Q\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 0-7 (TP)'
            c.node(f'mha_q_{layer}', attn_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_k_label = f'Layer{layer}_MHA_K\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 0-7 (TP)'
            c.node(f'mha_k_{layer}', attn_k_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_v_label = f'Layer{layer}_MHA_V\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 0-7 (TP)'
            c.node(f'mha_v_{layer}', attn_v_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_score_label = f'Layer{layer}_Attn_Score\nInput: [batch=128, seq=10000, heads=32, seq=10000]\nOutput: [batch=128, seq=10000, heads=32, seq=10000]\nDevice: 0-7 (TP)'
            c.node(f'attn_score_{layer}', attn_score_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_out_label = f'Layer{layer}_Attn_Out\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7 (TP)'
            c.node(f'attn_out_{layer}', attn_out_label, shape='rectangle', style='filled', color='lightblue')
            
            # All-Reduce for attention output
            attn_allreduce_label = f'Layer{layer}_Attn_AllReduce\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7 (All-Reduce)'
            c.node(f'attn_allreduce_{layer}', attn_allreduce_label, shape='ellipse', style='filled', color='red')
            
            # Residual Add for attention
            attn_residual_label = f'Layer{layer}_Attn_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7'
            c.node(f'attn_residual_{layer}', attn_residual_label, shape='parallelogram', style='filled', color='lightyellow')
            
            # MLP (FFN) within layer
            ffn_up_label = f'Layer{layer}_FFN_Up\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: 0-7 (TP)'
            c.node(f'ffn_up_{layer}', ffn_up_label, shape='rectangle', style='filled', color='lightblue')
            
            ffn_gate_label = f'Layer{layer}_FFN_Gate\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: 0-7 (TP)'
            c.node(f'ffn_gate_{layer}', ffn_gate_label, shape='rectangle', style='filled', color='lightblue')
            
            ffn_down_label = f'Layer{layer}_FFN_Down\nInput: [batch=128, seq=10000, ffn=16384]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7 (TP)'
            c.node(f'ffn_down_{layer}', ffn_down_label, shape='rectangle', style='filled', color='lightblue')
            
            # All-Reduce for FFN output
            ffn_allreduce_label = f'Layer{layer}_FFN_AllReduce\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7 (All-Reduce)'
            c.node(f'ffn_allreduce_{layer}', ffn_allreduce_label, shape='ellipse', style='filled', color='red')
            
            # Residual Add for FFN
            ffn_residual_label = f'Layer{layer}_FFN_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 0-7'
            c.node(f'ffn_residual_{layer}', ffn_residual_label, shape='parallelogram', style='filled', color='lightyellow')

# Pipeline Stage 1 (Devices 8-15)
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 (Devices 8-15)', style='rounded')
    
    # Layers 8-15 distributed with TP=8
    for layer in range(8, 16):
        with c.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (TP=8 across devices 8-15)', style='dashed')
            
            # Multi-Head Attention within layer
            attn_label = f'Layer{layer}_MHA_Q\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 8-15 (TP)'
            c.node(f'mha_q_{layer}', attn_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_k_label = f'Layer{layer}_MHA_K\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 8-15 (TP)'
            c.node(f'mha_k_{layer}', attn_k_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_v_label = f'Layer{layer}_MHA_V\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, heads=32, d_k=128]\nDevice: 8-15 (TP)'
            c.node(f'mha_v_{layer}', attn_v_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_score_label = f'Layer{layer}_Attn_Score\nInput: [batch=128, seq=10000, heads=32, seq=10000]\nOutput: [batch=128, seq=10000, heads=32, seq=10000]\nDevice: 8-15 (TP)'
            c.node(f'attn_score_{layer}', attn_score_label, shape='rectangle', style='filled', color='lightblue')
            
            attn_out_label = f'Layer{layer}_Attn_Out\nInput: [batch=128, seq=10000, heads=32, d_k=128]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15 (TP)'
            c.node(f'attn_out_{layer}', attn_out_label, shape='rectangle', style='filled', color='lightblue')
            
            # All-Reduce for attention output
            attn_allreduce_label = f'Layer{layer}_Attn_AllReduce\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15 (All-Reduce)'
            c.node(f'attn_allreduce_{layer}', attn_allreduce_label, shape='ellipse', style='filled', color='red')
            
            # Residual Add for attention
            attn_residual_label = f'Layer{layer}_Attn_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15'
            c.node(f'attn_residual_{layer}', attn_residual_label, shape='parallelogram', style='filled', color='lightyellow')
            
            # MLP (FFN) within layer
            ffn_up_label = f'Layer{layer}_FFN_Up\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: 8-15 (TP)'
            c.node(f'ffn_up_{layer}', ffn_up_label, shape='rectangle', style='filled', color='lightblue')
            
            ffn_gate_label = f'Layer{layer}_FFN_Gate\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, ffn=16384]\nDevice: 8-15 (TP)'
            c.node(f'ffn_gate_{layer}', ffn_gate_label, shape='rectangle', style='filled', color='lightblue')
            
            ffn_down_label = f'Layer{layer}_FFN_Down\nInput: [batch=128, seq=10000, ffn=16384]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15 (TP)'
            c.node(f'ffn_down_{layer}', ffn_down_label, shape='rectangle', style='filled', color='lightblue')
            
            # All-Reduce for FFN output
            ffn_allreduce_label = f'Layer{layer}_FFN_AllReduce\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15 (All-Reduce)'
            c.node(f'ffn_allreduce_{layer}', ffn_allreduce_label, shape='ellipse', style='filled', color='red')
            
            # Residual Add for FFN
            ffn_residual_label = f'Layer{layer}_FFN_Residual\nInput1: [batch=128, seq=10000, hidden=4096]\nInput2: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, hidden=4096]\nDevice: 8-15'
            c.node(f'ffn_residual_{layer}', ffn_residual_label, shape='parallelogram', style='filled', color='lightyellow')

# Pipeline communication between stages
dot.node('pipeline_comm_0', 'Pipeline Communication\nTransfer: [batch=128, seq=10000, hidden=4096]\nDevices: 0-7 → 8-15', 
         shape='ellipse', style='filled', color='lightgreen')

# Output layer
dot.node('output', 'Output Projection\nInput: [batch=128, seq=10000, hidden=4096]\nOutput: [batch=128, seq=10000, vocab_size]\nDevice: 8-15 (TP)', 
         shape='parallelogram', style='filled', color='lightyellow')

# Connect nodes
current = 'input'
for layer in range(8):
    # Attention flow
    dot.edge(current, f'mha_q_{layer}')
    dot.edge(current, f'mha_k_{layer}')
    dot.edge(current, f'mha_v_{layer}')
    dot.edge(f'mha_q_{layer}', f'attn_score_{layer}')
    dot.edge(f'mha_k_{layer}', f'attn_score_{layer}')
    dot.edge(f'attn_v_{layer}', f'attn_out_{layer}')
    dot.edge(f'attn_score_{layer}', f'attn_out_{layer}')
    dot.edge(f'attn_out_{layer}', f'attn_allreduce_{layer}')
    dot.edge(f'attn_allreduce_{layer}', f'attn_residual_{layer}')
    dot.edge(current, f'attn_residual_{layer}')  # Residual connection
    
    # FFN flow
    dot.edge(f'attn_residual_{layer}', f'ffn_up_{layer}')
    dot.edge(f'attn_residual_{layer}', f'ffn_gate_{layer}')
    dot.edge(f'ffn_up_{layer}', f'ffn_down_{layer}')
    dot.edge(f'ffn_gate_{layer}', f'ffn_down_{layer}')
    dot.edge(f'ffn_down_{layer}', f'ffn_allreduce_{layer}')
    dot.edge(f'ffn_allreduce_{layer}', f'ffn_residual_{layer}')
    dot.edge(f'attn_residual_{layer}', f'ffn_residual_{layer}')  # Residual connection
    
    current = f'ffn_residual_{layer}'

# Pipeline communication
dot.edge(current, 'pipeline_comm_0')
current = 'pipeline_comm_0'

# Stage 1 layers
for layer in range(8, 16):
    # Attention flow
    dot.edge(current, f'mha_q_{layer}')
    dot.edge(current, f'mha_k_{layer}')
    dot.edge(current, f'mha_v_{layer}')
    dot.edge(f'mha_q_{layer}', f'attn_score_{layer}')
    dot.edge(f'mha_k_{layer}', f'attn_score_{layer}')
    dot.edge(f'attn_v_{layer}', f'attn_out_{layer}')
    dot.edge(f'attn_score_{layer}', f'attn_out_{layer}')
    dot.edge(f'attn_out_{layer}', f'attn_allreduce_{layer}')
    dot.edge(f'attn_allreduce_{layer}', f'attn_residual_{layer}')
    dot.edge(current, f'attn_residual_{layer}')  # Residual connection
    
    # FFN flow
    dot.edge(f'attn_residual_{layer}', f'ffn_up_{layer}')
    dot.edge(f'attn_residual_{layer}', f'ffn_gate_{layer}')
    dot.edge(f'ffn_up_{layer}', f'ffn_down_{layer}')
    dot.edge(f'ffn_gate_{layer}', f'ffn_down_{layer}')
    dot.edge(f'ffn_down_{layer}', f'ffn_allreduce_{layer}')
    dot.edge(f'ffn_allreduce_{layer}', f'ffn_residual_{layer}')
    dot.edge(f'attn_residual_{layer}', f'ffn_residual_{layer}')  # Residual connection
    
    current = f'ffn_residual_{layer}'

# Final output
dot.edge(current, 'output')

# Save files
dot.render('../outputs/2025-11-21-15-21-34/baseline_dag', format='dot')
dot.render('../outputs/2025-11-21-15-21-34/baseline_dag', format='svg')

print("Baseline DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-11-21-15-21-34/baseline_dag.dot")
print("- ../outputs/2025-11-21-15-21-34/baseline_dag.svg")