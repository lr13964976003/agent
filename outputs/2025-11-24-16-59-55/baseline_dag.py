import graphviz

# Create baseline DAG: Tensor Parallel + Pipeline Parallel
dot = graphviz.Digraph('baseline_tp_pp', comment='Baseline: Tensor Parallel + Pipeline Parallel')

# Set graph attributes
dot.attr(rankdir='TB', ranksep='1.5', nodesep='0.8')
dot.attr('node', shape='rectangle', style='filled')

# Input node
dot.node('input', 'Input Embedding', 
         shape='ellipse', 
         fillcolor='lightblue',
         xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]')

# Pipeline Stage 0 (Devices 0-7)
with dot.subgraph(name='cluster_pipeline_0') as p0:
    p0.attr(label='Pipeline Stage 0 (Devices 0-7)', style='dashed', color='blue')
    
    # Layer 0-7 processing
    for layer in range(8):
        layer_id = f'layer_{layer}'
        
        # Multi-Head Attention Block
        with p0.subgraph(name=f'cluster_mha_{layer}') as mha:
            mha.attr(label=f'Layer {layer} - MHA Block', style='rounded,dashed', color='green')
            
            # QKV Projection - Column Parallel
            qkv_node = f'{layer_id}_qkv'
            mha.node(qkv_node, f'Layer {layer}\\nQKV Projection\\n(Column Parallel)', 
                     fillcolor='lightgreen',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, heads=32, qkv_dim=128]\\nDevices: [0,1,2,3] vs [4,5,6,7]')
            
            # Attention Computation - All devices
            attn_node = f'{layer_id}_attn'
            mha.node(attn_node, f'Layer {layer}\\nMulti-Head Attention\\n(All Devices)', 
                     fillcolor='yellow',
                     xlabel='Input: [batch_size=128, seq_len=100000, heads=32, qkv_dim=128]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nAll GPUs: 0-7')
            
            # Attention Output - Row Parallel
            attn_out_node = f'{layer_id}_attn_out'
            mha.node(attn_out_node, f'Layer {layer}\\nAttention Output\\n(Row Parallel)', 
                     fillcolor='lightgreen',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nDevices: [0,1,2,3] vs [4,5,6,7]')
            
            # Residual Connection
            residual_node = f'{layer_id}_residual'
            mha.node(residual_node, f'Layer {layer}\\nResidual Add', 
                     shape='parallelogram', 
                     fillcolor='lightgray',
                     xlabel='Input1: [batch_size=128, seq_len=100000, d_model=4096]\\nInput2: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]')

        # MLP Block
        with p0.subgraph(name=f'cluster_mlp_{layer}') as mlp:
            mlp.attr(label=f'Layer {layer} - MLP Block', style='rounded,dashed', color='red')
            
            # MLP Gate - Column Parallel
            gate_node = f'{layer_id}_gate'
            mlp.node(gate_node, f'Layer {layer}\\nMLP Gate\\n(Column Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nDevices: [0,1,2,3] vs [4,5,6,7]')
            
            # MLP Up - Column Parallel
            up_node = f'{layer_id}_up'
            mlp.node(up_node, f'Layer {layer}\\nMLP Up\\n(Column Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nDevices: [0,1,2,3] vs [4,5,6,7]')
            
            # MLP Down - Row Parallel
            down_node = f'{layer_id}_down'
            mlp.node(down_node, f'Layer {layer}\\nMLP Down\\n(Row Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nDevices: [0,1,2,3] vs [4,5,6,7]')
            
            # MLP Residual
            mlp_residual = f'{layer_id}_mlp_residual'
            mlp.node(mlp_residual, f'Layer {layer}\\nMLP Residual Add', 
                     shape='parallelogram', 
                     fillcolor='lightgray',
                     xlabel='Input1: [batch_size=128, seq_len=100000, d_model=4096]\\nInput2: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]')

# Pipeline Stage 1 (Devices 8-15)
with dot.subgraph(name='cluster_pipeline_1') as p1:
    p1.attr(label='Pipeline Stage 1 (Devices 8-15)', style='dashed', color='purple')
    
    # Layer 8-15 processing
    for layer in range(8, 16):
        layer_id = f'layer_{layer}'
        
        # Multi-Head Attention Block
        with p1.subgraph(name=f'cluster_mha_{layer}') as mha:
            mha.attr(label=f'Layer {layer} - MHA Block', style='rounded,dashed', color='green')
            
            # QKV Projection - Column Parallel
            qkv_node = f'{layer_id}_qkv'
            mha.node(qkv_node, f'Layer {layer}\\nQKV Projection\\n(Column Parallel)', 
                     fillcolor='lightgreen',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, heads=32, qkv_dim=128]\\nDevices: [8,9,10,11] vs [12,13,14,15]')
            
            # Attention Computation - All devices
            attn_node = f'{layer_id}_attn'
            mha.node(attn_node, f'Layer {layer}\\nMulti-Head Attention\\n(All Devices)', 
                     fillcolor='yellow',
                     xlabel='Input: [batch_size=128, seq_len=100000, heads=32, qkv_dim=128]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nAll GPUs: 8-15')
            
            # Attention Output - Row Parallel
            attn_out_node = f'{layer_id}_attn_out'
            mha.node(attn_out_node, f'Layer {layer}\\nAttention Output\\n(Row Parallel)', 
                     fillcolor='lightgreen',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nDevices: [8,9,10,11] vs [12,13,14,15]')
            
            # Residual Connection
            residual_node = f'{layer_id}_residual'
            mha.node(residual_node, f'Layer {layer}\\nResidual Add', 
                     shape='parallelogram', 
                     fillcolor='lightgray',
                     xlabel='Input1: [batch_size=128, seq_len=100000, d_model=4096]\\nInput2: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]')

        # MLP Block
        with p1.subgraph(name=f'cluster_mlp_{layer}') as mlp:
            mlp.attr(label=f'Layer {layer} - MLP Block', style='rounded,dashed', color='red')
            
            # MLP Gate - Column Parallel
            gate_node = f'{layer_id}_gate'
            mlp.node(gate_node, f'Layer {layer}\\nMLP Gate\\n(Column Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nDevices: [8,9,10,11] vs [12,13,14,15]')
            
            # MLP Up - Column Parallel
            up_node = f'{layer_id}_up'
            mlp.node(up_node, f'Layer {layer}\\nMLP Up\\n(Column Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nDevices: [8,9,10,11] vs [12,13,14,15]')
            
            # MLP Down - Row Parallel
            down_node = f'{layer_id}_down'
            mlp.node(down_node, f'Layer {layer}\\nMLP Down\\n(Row Parallel)', 
                     fillcolor='lightcoral',
                     xlabel='Input: [batch_size=128, seq_len=100000, mlp_hidden=8192]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\nDevices: [8,9,10,11] vs [12,13,14,15]')
            
            # MLP Residual
            mlp_residual = f'{layer_id}_mlp_residual'
            mlp.node(mlp_residual, f'Layer {layer}\\nMLP Residual Add', 
                     shape='parallelogram', 
                     fillcolor='lightgray',
                     xlabel='Input1: [batch_size=128, seq_len=100000, d_model=4096]\\nInput2: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]')

# Pipeline Communication
pipeline_comm = 'pipeline_communication'
dot.node(pipeline_comm, 'Pipeline Communication\\n(Micro-batch Transfer)', 
         shape='ellipse', 
         fillcolor='orange',
         xlabel='Transfer micro-batches\\nFrom Stage 0 to Stage 1\\nDevices: 0-7 → 8-15')

# Output node
output_node = 'output'
dot.node(output_node, 'Output Layer', 
         shape='ellipse', 
         fillcolor='lightblue',
         xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, vocab_size]')

# Connections - simplified for clarity
# Input to first layer
for layer in range(16):
    layer_id = f'layer_{layer}'
    if layer == 0:
        dot.edge('input', f'{layer_id}_qkv')
    
    # QKV -> Attention -> Output -> Residual
    dot.edge(f'{layer_id}_qkv', f'{layer_id}_attn')
    dot.edge(f'{layer_id}_attn', f'{layer_id}_attn_out')
    dot.edge(f'{layer_id}_attn_out', f'{layer_id}_residual')
    
    # Residual -> MLP
    dot.edge(f'{layer_id}_residual', f'{layer_id}_gate')
    dot.edge(f'{layer_id}_gate', f'{layer_id}_up')
    dot.edge(f'{layer_id}_up', f'{layer_id}_down')
    dot.edge(f'{layer_id}_down', f'{layer_id}_mlp_residual')
    
    # Pipeline communication after layer 7
    if layer == 7:
        dot.edge('layer_7_mlp_residual', 'pipeline_commulation')
    elif layer == 8:
        dot.edge('pipeline_communication', 'layer_8_qkv')
    elif layer < 15:
        dot.edge(f'layer_{layer}_mlp_residual', f'layer_{layer+1}_qkv')
    else:
        dot.edge('layer_15_mlp_residual', 'output')

# Save the DAG
dot.render('../outputs/2025-11-24-16-59-55/baseline_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-11-24-16-59-55/baseline_dag', format='svg', cleanup=False)

print("Baseline DAG generated successfully")
print("Files saved:")
print("- ../outputs/2025-11-24-16-59-55/baseline_dag.dot")
print("- ../outputs/2025-11-24-16-59-55/baseline_dag.svg")