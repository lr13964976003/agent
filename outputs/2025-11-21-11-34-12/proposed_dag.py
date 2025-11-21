import graphviz

# Create proposed DAG for Two-Level Attention Partitioning
proposed = graphviz.Digraph('proposed_transformer_dag', format='svg')
proposed.attr(rankdir='TB', size='30,40')

# Define styles
communication = {'style': 'filled', 'fillcolor': 'lightyellow', 'shape': 'ellipse'}
computation = {'style': 'filled', 'fillcolor': 'lightblue', 'shape': 'box'}
routing = {'style': 'filled', 'fillcolor': 'lightgreen', 'shape': 'parallelogram'}

# Input node
proposed.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: Host', **routing)

# Input broadcast to all 16 devices
broadcast_node = 'input_broadcast'
proposed.node(broadcast_node, 'Input Broadcast\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15 (all)', **communication)
proposed.edge('input', broadcast_node)

# Layer representation (showing one layer with 16 partitions)
for layer in range(16):
    if layer > 0:  # Only show layer 0 to avoid repetition, indicate others
        layer_label = f'layer_{layer}_...'
        proposed.node(layer_label, f'Layer {layer} (Same as Layer 0)\\n16 partitions, 4 head groups × 4 dim slices\\nGPU: 0-15', **routing)
        if layer == 1:
            proposed.edge('layer_0_final_residual', layer_label)
        continue
    
    layer_prefix = f'layer_{layer}'
    
    # First LayerNorm (replicated)
    norm1 = f'{layer_prefix}_norm1'
    proposed.node(norm1, f'LayerNorm (Layer {layer})\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15 (all)', **computation)
    if layer == 0:
        proposed.edge(broadcast_node, norm1)
    
    # Query, Key, Value projections for all 16 partitions
    for head_group in range(4):
        for slice_group in range(4):
            device_id = head_group * 4 + slice_group
            partition_id = f'{head_group}_{slice_group}'
            
            # Query projection
            q_proj = f'{layer_prefix}_q_proj_{partition_id}'
            proposed.node(q_proj, f'Query Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, 8×32=256]\\nGPU: {device_id}', **computation)
            proposed.edge(norm1, q_proj)
            
            # Key projection
            k_proj = f'{layer_prefix}_k_proj_{partition_id}'
            proposed.node(k_proj, f'Key Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, 8×32=256]\\nGPU: {device_id}', **computation)
            proposed.edge(norm1, k_proj)
            
            # Value projection
            v_proj = f'{layer_prefix}_v_proj_{partition_id}'
            proposed.node(v_proj, f'Value Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, 8×32=256]\\nGPU: {device_id}', **computation)
            proposed.edge(norm1, v_proj)
            
            # Reshape for attention
            q_reshape = f'{layer_prefix}_q_reshape_{partition_id}'
            proposed.node(q_reshape, f'Reshape Query\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 8, 32]\\nGPU: {device_id}', **computation)
            proposed.edge(q_proj, q_reshape)
            
            k_reshape = f'{layer_prefix}_k_reshape_{partition_id}'
            proposed.node(k_reshape, f'Reshape Key\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 8, 32]\\nGPU: {device_id}', **computation)
            proposed.edge(k_proj, k_reshape)
            
            v_reshape = f'{layer_prefix}_v_reshape_{partition_id}'
            proposed.node(v_reshape, f'Reshape Value\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 8, 32]\\nGPU: {device_id}', **computation)
            proposed.edge(v_proj, v_reshape)
            
            # Attention computation
            attn = f'{layer_prefix}_attn_{partition_id}'
            proposed.node(attn, f'Scaled Dot-Product Attention\\nInput: Q[128,10000,8,32], K[128,10000,8,32], V[128,10000,8,32]\\nOutput: [batch_size=128, seq_len=10000, 8, 32]\\nGPU: {device_id}', **computation)
            proposed.edge(q_reshape, attn)
            proposed.edge(k_reshape, attn)
            proposed.edge(v_reshape, attn)
            
            # Reshape back
            attn_reshape = f'{layer_prefix}_attn_reshape_{partition_id}'
            proposed.node(attn_reshape, f'Reshape Attention Output\\nInput: [batch_size=128, seq_len=10000, 8, 32]\\nOutput: [batch_size=128, seq_len=10000, 256]\\nGPU: {device_id}', **computation)
            proposed.edge(attn, attn_reshape)
    
    # Intra-group concatenation (4 devices per head group)
    for head_group in range(4):
        concat_intra = f'{layer_prefix}_concat_intra_{head_group}'
        devices_in_group = [head_group * 4 + s for s in range(4)]
        proposed.node(concat_intra, f'Intra-Group Concatenation\\nInput: 4×[batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 1024]\\nGPU: {devices_in_group}', **communication)
        
        for slice_group in range(4):
            partition_id = f'{head_group}_{slice_group}'
            proposed.edge(f'{layer_prefix}_attn_reshape_{partition_id}', concat_intra)
    
    # Inter-group concatenation to form final attention output
    concat_inter = f'{layer_prefix}_concat_inter'
    proposed.node(concat_inter, f'Inter-Group Concatenation\\nInput: 4×[batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: 0-15 (hierarchical)', **communication)
    
    for head_group in range(4):
        concat_intra = f'{layer_prefix}_concat_intra_{head_group}'
        proposed.edge(concat_intra, concat_inter)
    
    # Output projection for each partition
    for head_group in range(4):
        for slice_group in range(4):
            device_id = head_group * 4 + slice_group
            partition_id = f'{head_group}_{slice_group}'
            
            out_proj = f'{layer_prefix}_out_proj_{partition_id}'
            proposed.node(out_proj, f'Output Projection\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: {device_id}', **computation)
            # This would come from attention output reshaping
    
    # Final concatenation and residual
    final_concat = f'{layer_prefix}_final_concat'
    proposed.node(final_concat, f'Final Attention Concatenation\\nInput: 16×[batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: 0-15 (all)', **communication)
    proposed.edge(concat_inter, final_concat)
    
    # First residual connection
    residual1 = f'{layer_prefix}_residual1'
    proposed.node(residual1, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, 4096], [batch_size=128, seq_len=10000, 4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15 (all)', **computation)
    proposed.edge(norm1, residual1)
    proposed.edge(concat_inter, residual1)
    
    # Second LayerNorm
    norm2 = f'{layer_prefix}_norm2'
    proposed.node(norm2, f'LayerNorm\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15 (all)', **computation)
    proposed.edge(residual1, norm2)
    
    # MLP projections (simplified - would need similar partitioning)
    # In practice, MLP would also be partitioned across devices
    mlp_gate = f'{layer_prefix}_mlp_gate'
    proposed.node(mlp_gate, f'MLP Gate\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, 16384]\\nGPU: 0-15 (distributed)', **computation)
    proposed.edge(norm2, mlp_gate)
    
    mlp_up = f'{layer_prefix}_mlp_up'
    proposed.node(mlp_up, f'MLP Up\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, 16384]\\nGPU: 0-15 (distributed)', **computation)
    proposed.edge(norm2, mlp_up)
    
    mlp_act = f'{layer_prefix}_mlp_act'
    proposed.node(mlp_act, f'MLP Activation\\nInput: [batch_size=128, seq_len=10000, 16384]\\nOutput: [batch_size=128, seq_len=10000, 16384]\\nGPU: 0-15 (all)', **computation)
    proposed.edge(mlp_gate, mlp_act)
    
    mlp_down = f'{layer_prefix}_mlp_down'
    proposed.node(mlp_down, f'MLP Down\\nInput: [batch_size=128, seq_len=10000, 16384]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: 0-15 (distributed)', **computation)
    proposed.edge(mlp_act, mlp_down)
    
    # Second residual connection
    residual2 = f'{layer_prefix}_final_residual'
    proposed.node(residual2, f'Residual Add\\nInput: [batch_size=128, seq_len=10000, 4096], [batch_size=128, seq_len=10000, 4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15 (all)', **computation)
    proposed.edge(residual1, residual2)
    proposed.edge(mlp_down, residual2)

# Indicate remaining layers (1-15)
for layer in range(1, 16):
    layer_ellipsis = f'layer_{layer}_ellipsis'
    proposed.node(layer_ellipsis, f'Layer {layer}\\n(Same partitioning as Layer 0)\\n16 partitions, hierarchical concat\\nGPU: 0-15', **routing)
    if layer == 1:
        proposed.edge('layer_0_final_residual', layer_ellipsis)
    elif layer > 1:
        prev_ellipsis = f'layer_{layer-1}_ellipsis'
        proposed.edge(prev_ellipsis, layer_ellipsis)

# Output
output_node = 'final_output'
proposed.node(output_node, 'Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: 0-15', **routing)
proposed.edge('layer_15_ellipsis', output_node)

# Save the files
proposed.render('../outputs/2025-11-21-11-34-12/proposed_dag', format='dot', cleanup=False)
proposed.render('../outputs/2025-11-21-11-34-12/proposed_dag', format='svg', cleanup=False)

print("Proposed DAG files generated successfully!")