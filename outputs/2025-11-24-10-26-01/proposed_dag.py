#!/usr/bin/env python3
import graphviz

# Create proposed DAG for layer-wise cache-aware partitioning
dot = graphviz.Digraph('proposed_layer_wise', comment='Dense 16-layer model with layer-wise partitioning (1 layer per GPU)')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')

# Define node shapes and styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')

# Input node
dot.node('input', 'Model Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]', 
         shape='parallelogram', fillcolor='lightgreen')

# Process each layer on separate GPU
for layer in range(16):
    gpu_id = layer
    
    # Create subgraph for each layer on its dedicated GPU
    with dot.subgraph(name=f'cluster_layer{layer}_gpu{gpu_id}') as layer_cluster:
        layer_cluster.attr(label=f'Layer {layer} on GPU {gpu_id}\\nCache-capacity: 11.8GB', 
                          style='rounded,dashed', color='green', fillcolor='lightyellow')
        
        # Input transfer for this layer
        if layer == 0:
            layer_cluster.node(f'layer{layer}_input', 
                             f'Input Transfer\\nFrom: Input\\nSize: 524MB\\nGPU: {gpu_id}', 
                             shape='parallelogram', fillcolor='lightyellow')
        else:
            layer_cluster.node(f'layer{layer}_input', 
                             f'Input Transfer\\nFrom: GPU {layer-1}\\nSize: 524MB\\nGPU: {gpu_id}', 
                             shape='parallelogram', fillcolor='lightyellow')
        
        # Attention components (no tensor parallelism, full layer)
        layer_cluster.node(f'lay{layer}_qkv_proj', 
                         f'QKV Projection\\nInput: [128,10000,4096]\\nOutput: [128,10000,32,128]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightcoral')
        
        layer_cluster.node(f'lay{layer}_attn_score', 
                         f'Scaled Dot-Product Attention\\nInput: [128,10000,32,128], [128,10000,32,128]\\nOutput: [128,10000,32,128]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightpink')
        
        layer_cluster.node(f'lay{layer}_attn_concat', 
                         f'Concat Heads\\nInput: [128,10000,32,128]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightpink')
        
        layer_cluster.node(f'lay{layer}_attn_out', 
                         f'Attention Output Projection\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightcoral')
        
        layer_cluster.node(f'lay{layer}_res1', 
                         f'Residual Add 1\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='ellipse', fillcolor='lightgray')
        
        # Layer normalization after attention
        layer_cluster.node(f'lay{layer}_ln1', 
                         f'LayerNorm 1\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightsteelblue')
        
        # MLP components (full layer on single GPU)
        layer_cluster.node(f'lay{layer}_mlp_gate', 
                         f'MLP Gate Projection\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightseagreen')
        
        layer_cluster.node(f'lay{layer}_mlp_up', 
                         f'MLP Up Projection\\nInput: [128,10000,4096]\\nOutput: [128,10000,16384]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightseagreen')
        
        layer_cluster.node(f'lay{layer}_mlp_act', 
                         f'GELU Activation\\nInput: [128,10000,16384]\\nOutput: [128,10000,16384]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightblue')
        
        layer_cluster.node(f'lay{layer}_mlp_mul', 
                         f'Element-wise Mul\\nInput: [128,10000,16384], [128,10000,16384]\\nOutput: [128,10000,16384]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightblue')
        
        layer_cluster.node(f'lay{layer}_mlp_down', 
                         f'MLP Down Projection\\nInput: [128,10000,16384]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightseagreen')
        
        layer_cluster.node(f'lay{layer}_res2', 
                         f'Residual Add 2\\nInput: [128,10000,4096], [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='ellipse', fillcolor='lightgray')
        
        # Layer normalization after MLP
        layer_cluster.node(f'lay{layer}_ln2', 
                         f'LayerNorm 2\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]\\nGPU: {gpu_id}', 
                         shape='rectangle', fillcolor='lightsteelblue')
        
        # Output transfer for this layer
        if layer < 15:
            layer_cluster.node(f'layer{layer}_output', 
                             f'Output Transfer\\nTo: GPU {layer+1}\\nSize: 524MB\\nGPU: {gpu_id}', 
                             shape='parallelogram', fillcolor='lightyellow')
        else:
            layer_cluster.node(f'layer{layer}_output', 
                             f'Model Output\\nTo: Final Output\\nSize: 524MB\\nGPU: {gpu_id}', 
                             shape='parallelogram', fillcolor='lightgreen')

# Final output node
dot.node('final_output', 'Final Model Output\\nInput: [128,10000,4096]\\nOutput: [128,10000,4096]', 
         shape='parallelogram', fillcolor='lightgreen')

# Create communication edges
for layer in range(16):
    # Input connections
    if layer == 0:
        dot.edge('input', f'layer{layer}_input')
    else:
        dot.edge(f'layer{layer-1}_output', f'layer{layer}_input')
    
    # Internal connections for each layer
    dot.edge(f'layer{layer}_input', f'lay{layer}_qkv_proj')
    dot.edge(f'lay{layer}_qkv_proj', f'lay{layer}_attn_score')
    dot.edge(f'lay{layer}_attn_score', f'lay{layer}_attn_concat')
    dot.edge(f'lay{layer}_attn_concat', f'lay{layer}_attn_out')
    dot.edge(f'lay{layer}_attn_out', f'lay{layer}_res1')
    dot.edge(f'layer{layer}_input', f'lay{layer}_res1')  # Residual connection
    dot.edge(f'lay{layer}_res1', f'lay{layer}_ln1')
    
    # MLP connections
    dot.edge(f'lay{layer}_ln1', f'lay{layer}_mlp_gate')
    dot.edge(f'lay{layer}_ln1', f'lay{layer}_mlp_up')
    dot.edge(f'lay{layer}_mlp_gate', f'lay{layer}_mlp_act')
    dot.edge(f'lay{layer}_mlp_up', f'lay{layer}_mlp_mul')
    dot.edge(f'lay{layer}_mlp_act', f'lay{layer}_mlp_mul')
    dot.edge(f'lay{layer}_mlp_mul', f'lay{layer}_mlp_down')
    dot.edge(f'lay{layer}_mlp_down', f'lay{layer}_res2')
    dot.edge(f'lay{layer}_ln1', f'lay{layer}_res2')  # Residual connection
    dot.edge(f'lay{layer}_res2', f'lay{layer}_ln2')
    dot.edge(f'lay{layer}_ln2', f'layer{layer}_output')

# Final output connection
dot.edge('layer15_output', 'final_output')

# Save the DOT file
dot.format = 'dot'
dot.render('../outputs/2025-11-24-10-26-01/proposed_dag')

# Save as SVG
dot.format = 'svg'
dot.render('../outputs/2025-11-24-10-26-01/proposed_dag')

print("Proposed DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-11-24-10-26-01/proposed_dag.dot")
print("- ../outputs/2025-11-24-10-26-01/proposed_dag.svg")