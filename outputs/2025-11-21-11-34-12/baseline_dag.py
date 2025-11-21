import graphviz

# Create baseline DAG for Tensor Parallel + Pipeline Parallel
baseline = graphviz.Digraph('baseline_transformer_dag', format='svg')
baseline.attr(rankdir='TB', size='20,30')

# Define styles
baseline.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
communication = {'style': 'filled', 'fillcolor': 'lightyellow', 'shape': 'ellipse'}
computation = {'style': 'filled', 'fillcolor': 'lightblue', 'shape': 'box'}
routing = {'style': 'filled', 'fillcolor': 'lightgreen', 'shape': 'parallelogram'}

# Input node
baseline.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: Host', **routing)

# Pipeline Stage 0 (Layers 0-7)
baseline.node('stage0_start', 'Pipeline Stage 0\\nLayers 0-7\\nDevices: 0-7', **routing)
baseline.edge('input', 'stage0_start')

# Tensor Parallelism within Stage 0
for layer in range(8):
    layer_id = f'layer_{layer}'
    
    # LayerNorm (replicated across TP group)
    norm_node = f'{layer_id}_norm'
    baseline.node(norm_node, f'LayerNorm (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (all)', **computation)
    
    if layer == 0:
        baseline.edge('stage0_start', norm_node)
    else:
        prev_layer = f'layer_{layer-1}_mlp'
        baseline.edge(prev_layer, norm_node)
    
    # Multi-Head Attention (TP across 8 devices)
    # Each device handles 4 heads (32/8 = 4 heads per device)
    attn_node = f'{layer_id}_attn'
    baseline.node(attn_node, f'MHA-TP (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nPer GPU: 4 heads, head_dim=128\\nGPU: 0-7 (TP)', **computation)
    baseline.edge(norm_node, attn_node)
    
    # All-reduce for attention output
    attn_reduce = f'{layer_id}_attn_reduce'
    baseline.node(attn_reduce, f'All-Reduce Attention\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (TP)', **communication)
    baseline.edge(attn_node, attn_reduce)
    
    # Residual connection
    residual_add1 = f'{layer_id}_residual1'
    baseline.node(residual_add1, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096], [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (all)', **computation)
    baseline.edge(norm_node, residual_add1)
    baseline.edge(attn_reduce, residual_add1)
    
    # Second LayerNorm
    norm2_node = f'{layer_id}_norm2'
    baseline.node(norm2_node, f'LayerNorm (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (all)', **computation)
    baseline.edge(residual_add1, norm2_node)
    
    # MLP (TP across 8 devices)
    # Hidden size 16384 split across 8 devices = 2048 per device
    mlp_node = f'{layer_id}_mlp'
    baseline.node(mlp_node, f'MLP-TP (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nGPU: 0-7 (TP)', **computation)
    baseline.edge(norm2_node, mlp_node)
    
    # All-reduce for MLP output
    mlp_reduce = f'{layer_id}_mlp_reduce'
    baseline.node(mlp_reduce, f'All-Reduce MLP\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (TP)', **communication)
    baseline.edge(mlp_node, mlp_reduce)
    
    # Second residual connection
    residual_add2 = f'{layer_id}_residual2'
    baseline.node(residual_add2, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096], [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 (all)', **computation)
    baseline.edge(residual_add1, residual_add2)
    baseline.edge(mlp_reduce, residual_add2)

# Pipeline communication between stages
baseline.node('pipeline_send', 'Send Pipeline\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-7 → 8-15', **communication)
baseline.edge('layer_7_residual2', 'pipeline_send')

# Pipeline Stage 1 (Layers 8-15)
baseline.node('stage1_start', 'Pipeline Stage 1\\nLayers 8-15\\nDevices: 8-15', **routing)
baseline.edge('pipeline_send', 'stage1_start')

# Tensor Parallelism within Stage 1
for layer in range(8, 16):
    layer_id = f'layer_{layer}'
    
    # LayerNorm (replicated across TP group)
    norm_node = f'{layer_id}_norm'
    baseline.node(norm_node, f'LayerNorm (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (all)', **computation)
    
    if layer == 8:
        baseline.edge('stage1_start', norm_node)
    else:
        prev_layer = f'layer_{layer-1}_mlp'
        baseline.edge(prev_layer, norm_node)
    
    # Multi-Head Attention (TP across 8 devices)
    attn_node = f'{layer_id}_attn'
    baseline.node(attn_node, f'MHA-TP (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nPer GPU: 4 heads, head_dim=128\\nGPU: 8-15 (TP)', **computation)
    baseline.edge(norm_node, attn_node)
    
    # All-reduce for attention output
    attn_reduce = f'{layer_id}_attn_reduce'
    baseline.node(attn_reduce, f'All-Reduce Attention\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (TP)', **communication)
    baseline.edge(attn_node, attn_reduce)
    
    # Residual connection
    residual_add1 = f'{layer_id}_residual1'
    baseline.node(residual_add1, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096], [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (all)', **computation)
    baseline.edge(norm_node, residual_add1)
    baseline.edge(attn_reduce, residual_add1)
    
    # Second LayerNorm
    norm2_node = f'{layer_id}_norm2'
    baseline.node(norm2_node, f'LayerNorm (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (all)', **computation)
    baseline.edge(residual_add1, norm2_node)
    
    # MLP (TP across 8 devices)
    mlp_node = f'{layer_id}_mlp'
    baseline.node(mlp_node, f'MLP-TP (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nGPU: 8-15 (TP)', **computation)
    baseline.edge(norm2_node, mlp_node)
    
    # All-reduce for MLP output
    mlp_reduce = f'{layer_id}_mlp_reduce'
    baseline.node(mlp_reduce, f'All-Reduce MLP\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (TP)', **communication)
    baseline.edge(mlp_node, mlp_reduce)
    
    # Second residual connection
    residual_add2 = f'{layer_id}_residual2'
    baseline.node(residual_add2, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096], [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15 (all)', **computation)
    baseline.edge(residual_add1, residual_add2)
    baseline.edge(mlp_reduce, residual_add2)

# Output node
baseline.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 8-15', **routing)
baseline.edge('layer_15_residual2', 'output')

# Save the files
baseline.render('../outputs/2025-11-21-11-34-12/baseline_dag', format='dot', cleanup=False)
baseline.render('../outputs/2025-11-21-11-34-12/baseline_dag', format='svg', cleanup=False)

print("Baseline DAG files generated successfully!")