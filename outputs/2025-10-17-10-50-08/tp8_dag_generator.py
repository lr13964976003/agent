import graphviz

# Create Tensor Parallelism 8 DAG
dot = graphviz.Digraph('TP8_Standalone', comment='Standalone Tensor Parallelism 8 Configuration')
dot.attr(rankdir='TB', size='25,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Global attributes
dot.attr(label='Standalone Tensor Parallelism 8\n8 GPUs total\nModel Parallel across all layers')
dot.attr(fontsize='20')

# Input Layer
dot.node('input', 'INPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 0-7', shape='ellipse', fillcolor='lightblue')

# 4 Layers with TP=8
for layer in range(4):
    # Tensor parallel attention across 8 GPUs
    for gpu_id in range(8):
        # QKV Projection with tensor parallelism
        dot.node(f'layer{layer}_qkv_gpu{gpu_id}', 
                f'L{layer}_QKV_Projection_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,512]\nHeads: {gpu_id*4}-{(gpu_id+1)*4-1}\nTP Slice: {gpu_id}\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Multi-Head Attention with tensor parallelism
        dot.node(f'layer{layer}_attn_gpu{gpu_id}', 
                f'L{layer}_Multi-Head_Attention_GPU{gpu_id}\nInput: [1024,2048,512]\nOutput: [1024,2048,512]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Attention output projection
        dot.node(f'layer{layer}_attn_proj_gpu{gpu_id}', 
                f'L{layer}_Attn_Projection_GPU{gpu_id}\nInput: [1024,2048,512]\nOutput: [1024,2048,512]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')

    # All-reduce for attention
    dot.node(f'layer{layer}_attn_allreduce', 
            f'L{layer}_Attention_All-Reduce\nInput: 8×[1024,2048,512]\nOutput: [1024,2048,4096]\nTP All-Reduce across GPUs 0-7',
            shape='diamond', fillcolor='lightcoral')
    
    # Distributed LayerNorm after attention
    for gpu_id in range(8):
        dot.node(f'layer{layer}_attn_norm_gpu{gpu_id}', 
                f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nReplicated across GPUs 0-7',
                shape='rectangle', fillcolor='lightgreen')

    # MoE with tensor parallelism
    for gpu_id in range(8):
        # Gate computation
        dot.node(f'layer{layer}_gate_gpu{gpu_id}', 
                f'L{layer}_Gate_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,16 experts]\nTP: expert routing\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Expert processing (all experts on each GPU)
        for expert in range(16):
            dot.node(f'layer{layer}_expert{gpu_id}_{expert}', 
                    f'L{layer}_Expert{expert}_GPU{gpu_id}\nInput: [tokens,4096]\nOutput: [tokens,4096]\nExpert hidden: 16384\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
        
        # MoE output aggregation
        dot.node(f'layer{layer}_moe_agg_gpu{gpu_id}', 
                f'L{layer}_MoE_Aggregate_GPU{gpu_id}\nInput: 16×[tokens,4096]\nOutput: [1024,2048,4096]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Final residual add & layer norm
        dot.node(f'layer{layer}_output_gpu{gpu_id}', 
                f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,4096]x2\nOutput: [1024,2048,4096]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')

    # All-reduce for MoE
    dot.node(f'layer{layer}_moe_allreduce', 
            f'L{layer}_MoE_All-Reduce\nInput: 8×[1024,2048,4096]\nOutput: [1024,2048,4096]\nTP All-Reduce across GPUs 0-7',
            shape='diamond', fillcolor='lightcoral')

# Output Layer
dot.node('output', 'OUTPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 0-7', shape='ellipse', fillcolor='lightblue')

# Connect all nodes
# Input to Layer 0
for gpu_id in range(8):
    dot.edge('input', f'layer0_qkv_gpu{gpu_id}')

# Connect within each layer
for layer in range(4):
    # Attention path
    for gpu_id in range(8):
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_proj_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_proj_gpu{gpu_id}', f'layer{layer}_attn_allreduce')
        dot.edge(f'layer{layer}_attn_allreduce', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        # MoE path
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_gate_gpu{gpu_id}')
        
        # Connect gate to experts
        for expert in range(16):
            dot.edge(f'layer{layer}_gate_gpu{gpu_id}', f'layer{layer}_expert{gpu_id}_{expert}', style='dashed')
        
        # Connect experts back to aggregation
        for expert in range(16):
            dot.edge(f'layer{layer}_expert{gpu_id}_{expert}', f'layer{layer}_moe_agg_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_moe_allreduce')
        dot.edge(f'layer{layer}_moe_allreduce', f'layer{layer}_output_gpu{gpu_id}')
    
    # Layer to layer connections
    if layer < 3:
        for gpu_id in range(8):
            dot.edge(f'layer{layer}_output_gpu{gpu_id}', f'layer{layer+1}_qkv_gpu{gpu_id}')

# Final connections to output
for gpu_id in range(8):
    dot.edge(f'layer3_output_gpu{gpu_id}', 'output')

# Save the DAG
dot.render('../outputs/2025-10-17-10-50-08/tp8_standalone_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-10-17-10-50-08/tp8_standalone_dag', format='svg', cleanup=False)

print("Standalone TP8 DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-10-17-10-50-08/tp8_standalone_dag.dot")
print("- ../outputs/2025-10-17-10-50-08/tp8_standalone_dag.svg")