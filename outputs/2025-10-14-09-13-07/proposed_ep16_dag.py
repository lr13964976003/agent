import graphviz

# Create DAG for proposed: EP=16 with one expert per GPU
dot = graphviz.Digraph('proposed_ep16_moe', comment='Proposed MoE EP=16 Deployment')
dot.attr(rankdir='TB', size='30,40')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/aggregation

# Global input
dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\\nGPU: all GPUs', 
         shape='ellipse', fillcolor='lightblue')

# Create layers - each layer has attention + MoE across 16 GPUs
for layer in range(4):
    with dot.subgraph(name=f'cluster_layer{layer}') as layer_graph:
        layer_graph.attr(label=f'Layer {layer} (EP=16)\\n16 GPUs, 1 expert per GPU', style='dashed')
        
        # Attention computation (replicated on all GPUs for local processing)
        for gpu_id in range(16):
            layer_graph.node(f'l{layer}_qkv_gpu{gpu_id}', 
                           f'QKV Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
            
            layer_graph.node(f'l{layer}_attn_gpu{gpu_id}',
                           f'Multi-Head Attention\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
            
            layer_graph.node(f'l{layer}_out_proj_gpu{gpu_id}',
                           f'Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
            
            layer_graph.node(f'l{layer}_attn_residual_gpu{gpu_id}',
                           f'Attention Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
        
        # Gate computation (replicated on all GPUs for local routing decisions)
        for gpu_id in range(16):
            layer_graph.node(f'l{layer}_gate_gpu{gpu_id}',
                           f'Gate Network\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
        
        # Expert computation (one expert per GPU)
        for gpu_id in range(16):
            layer_graph.node(f'l{layer}_expert_gpu{gpu_id}',
                           f'Expert {gpu_id}\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
        
        # Token routing nodes (handle cross-GPU communication)
        for gpu_id in range(16):
            # Token split based on gate decisions
            layer_graph.node(f'l{layer}_token_split_gpu{gpu_id}',
                           f'Token Split\\nInput: [1024,10000,8192]\\nOutput: [16,tokens_per_expert,8192]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
            
            # Token send operations (async)
            for dest_gpu in range(16):
                if gpu_id != dest_gpu:
                    layer_graph.node(f'l{layer}_send_gpu{gpu_id}_to_{dest_gpu}',
                                   f'Send Tokens\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {gpu_id}→{dest_gpu}',
                                   shape='parallelogram', fillcolor='orange')
            
            # Token receive operations (async)
            for src_gpu in range(16):
                if gpu_id != src_gpu:
                    layer_graph.node(f'l{layer}_recv_gpu{src_gpu}_to_{gpu_id}',
                                   f'Receive Tokens\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {src_gpu}→{gpu_id}',
                                   shape='parallelogram', fillcolor='orange')
            
            # Local token processing
            layer_graph.node(f'l{layer}_local_tokens_gpu{gpu_id}',
                           f'Local Token Buffer\\nInput: [local_tokens,8192]\\nOutput: [local_tokens,8192]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
            
            # Expert aggregation after computation
            layer_graph.node(f'l{layer}_expert_agg_gpu{gpu_id}',
                           f'Expert Aggregation\\nInput: [computed_tokens,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')
            
            # Final residual add
            layer_graph.node(f'l{layer}_final_residual_gpu{gpu_id}',
                           f'Layer {layer} Final Residual\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                           shape='parallelogram', fillcolor='lightyellow')

# Global output
dot.node('output', 'Output\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: all GPUs', 
         shape='ellipse', fillcolor='lightblue')

# Connections for each layer
for layer in range(4):
    # Input to layer
    if layer == 0:
        for gpu_id in range(16):
            dot.edge('input', f'l{layer}_qkv_gpu{gpu_id}')
    else:
        for gpu_id in range(16):
            dot.edge(f'l{layer-1}_final_residual_gpu{gpu_id}', f'l{layer}_qkv_gpu{gpu_id}')
    
    # Attention computation path (local to each GPU)
    for gpu_id in range(16):
        dot.edge(f'l{layer}_qkv_gpu{gpu_id}', f'l{layer}_attn_gpu{gpu_id}')
        dot.edge(f'l{layer}_attn_gpu{gpu_id}', f'l{layer}_out_proj_gpu{gpu_id}')
        dot.edge(f'l{layer}_out_proj_gpu{gpu_id}', f'l{layer}_attn_residual_gpu{gpu_id}')
        
        # Residual connection for attention
        if layer == 0:
            dot.edge('input', f'l{layer}_attn_residual_gpu{gpu_id}')
        else:
            dot.edge(f'l{layer-1}_final_residual_gpu{gpu_id}', f'l{layer}_attn_residual_gpu{gpu_id}')
        
        # Gate computation
        dot.edge(f'l{layer}_attn_residual_gpu{gpu_id}', f'l{layer}_gate_gpu{gpu_id}')
        
        # Token split based on gate decisions
        dot.edge(f'l{layer}_attn_residual_gpu{gpu_id}', f'l{layer}_token_split_gpu{gpu_id}')
        dot.edge(f'l{layer}_gate_gpu{gpu_id}', f'l{layer}_token_split_gpu{gpu_id}', style='dashed')
        
        # Token routing and communication
        # Send tokens to other GPUs
        for dest_gpu in range(16):
            if gpu_id != dest_gpu:
                dot.edge(f'l{layer}_token_split_gpu{gpu_id}', f'l{layer}_send_gpu{gpu_id}_to_{dest_gpu}')
                dot.edge(f'l{layer}_send_gpu{gpu_id}_to_{dest_gpu}', f'l{layer}_recv_gpu{gpu_id}_to_{dest_gpu}')
                dot.edge(f'l{layer}_recv_gpu{gpu_id}_to_{dest_gpu}', f'l{layer}_local_tokens_gpu{dest_gpu}')
        
        # Local tokens stay on same GPU
        dot.edge(f'l{layer}_token_split_gpu{gpu_id}', f'l{layer}_local_tokens_gpu{gpu_id}')
        
        # Expert computation (local expert processes its assigned tokens)
        dot.edge(f'l{layer}_local_tokens_gpu{gpu_id}', f'l{layer}_expert_gpu{gpu_id}')
        
        # Expert aggregation (collect results from all experts)
        for src_gpu in range(16):
            dot.edge(f'l{layer}_expert_gpu{src_gpu}', f'l{layer}_expert_agg_gpu{gpu_id}')
        
        # Final residual connection
        dot.edge(f'l{layer}_attn_residual_gpu{gpu_id}', f'l{layer}_final_residual_gpu{gpu_id}')
        dot.edge(f'l{layer}_expert_agg_gpu{gpu_id}', f'l{layer}_final_residual_gpu{gpu_id}')

# Connect final layer to output
for gpu_id in range(16):
    dot.edge(f'l3_final_residual_gpu{gpu_id}', 'output')

# Save files
dot.render('./outputs/2025-10-14-09-13-07/proposed_ep16_moe', format='dot')
dot.render('./outputs/2025-10-14-09-13-07/proposed_ep16_moe', format='svg')

print("Proposed EP=16 DAG generated successfully!")
print(f"DOT file: ./outputs/2025-10-14-09-13-07/proposed_ep16_moe.dot")
print(f"SVG file: ./outputs/2025-10-14-09-13-07/proposed_ep16_moe.svg")