import graphviz

# Create MA Separation DAG
dot = graphviz.Digraph('MA_Separation_MoE_Attention', comment='MA Separation: 12 Attention GPUs + 4 MoE GPUs')
dot.attr(rankdir='TB', size='30,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Global attributes
dot.attr(label='MA Separation: 4-Layer MoE Transformer\n12 Attention GPUs + 4 MoE GPUs\nSequence: 2048 tokens, Batch: 1024, Hidden: 4096')
dot.attr(fontsize='20')

# Input Layer
dot.node('input', 'INPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: all', shape='ellipse', fillcolor='lightblue')

# Layer 1 - Attention Phase (GPUs 0-11)
for layer in range(1, 5):
    # Attention Layer Processing
    for gpu_id in range(12):
        # QKV Projection
        dot.node(f'layer{layer}_qkv_gpu{gpu_id}', 
                f'L{layer}_QKV_Projection_GPU{gpu_id}\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 3*4096]\nHeads: {gpu_id*3}-{(gpu_id+1)*3-1}\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Multi-Head Attention
        dot.node(f'layer{layer}_attn_gpu{gpu_id}', 
                f'L{layer}_Multi-Head_Attention_GPU{gpu_id}\nInput: [1024, 2048, 3*4096]\nOutput: [1024, 2048, 4096]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Residual Add & LayerNorm
        dot.node(f'layer{layer}_attn_norm_gpu{gpu_id}', 
                f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,4096]x2\nOutput: [1024,2048,4096]\nGPU: {gpu_id}',
                shape='rectangle', fillcolor='lightgreen')
    
    # Attention Aggregation
    dot.node(f'layer{layer}_attn_aggregate', 
            f'L{layer}_Attention_Aggregate\nInput: 12×[1024,2048,4096]\nOutput: [1024,2048,4096]\nAll-Reduce across GPUs 0-11',
            shape='diamond', fillcolor='lightcoral')
    
    # Broadcast to MoE GPUs
    dot.node(f'layer{layer}_moe_broadcast', 
            f'L{layer}_Broadcast_to_MoE\nInput: [1024,2048,4096]\nOutput: 4×[1024,2048,4096]\nBroadcast to GPUs 12-15',
            shape='parallelogram', fillcolor='yellow')

    # MoE Processing (GPUs 12-15)
    for moe_gpu in range(12, 16):
        expert_id = moe_gpu - 12
        # Gate computation
        dot.node(f'layer{layer}_gate_gpu{moe_gpu}', 
                f'L{layer}_Gate_GPU{moe_gpu}\nInput: [1024,2048,4096]\nOutput: [1024,2048,16 experts]\nTop-2 routing\nGPU: {moe_gpu}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Expert processing (4 experts per GPU)
        for exp in range(4):
            expert_num = (moe_gpu-12)*4 + exp
            dot.node(f'layer{layer}_expert{moe_gpu}_{expert_num}', 
                    f'L{layer}_Expert{expert_num}_GPU{moe_gpu}\nInput: [tokens,4096]\nOutput: [tokens,4096]\nExpert hidden: 16384\nGPU: {moe_gpu}',
                    shape='rectangle', fillcolor='lightgreen')
        
        # MoE aggregation
        dot.node(f'layer{layer}_moe_aggregate_gpu{moe_gpu}', 
                f'L{layer}_MoE_Aggregate_GPU{moe_gpu}\nInput: 4×[tokens,4096]\nOutput: [1024,2048,4096]\nGPU: {moe_gpu}',
                shape='rectangle', fillcolor='lightgreen')
        
        # Final residual add & layer norm
        dot.node(f'layer{layer}_moe_norm_gpu{moe_gpu}', 
                f'L{layer}_Residual+LayerNorm_GPU{moe_gpu}\nInput: [1024,2048,4096]x2\nOutput: [1024,2048,4096]\nGPU: {moe_gpu}',
                shape='rectangle', fillcolor='lightgreen')

# Output Layer
dot.node('output', 'OUTPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 12-15', shape='ellipse', fillcolor='lightblue')

# Connect all nodes
# Input to Layer 1 attention
for gpu_id in range(12):
    dot.edge('input', f'layer1_qkv_gpu{gpu_id}')

# Connect within each layer
for layer in range(1, 5):
    # Attention phase connections
    for gpu_id in range(12):
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_norm_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_attn_aggregate')
    
    # Attention to MoE transition
    dot.edge(f'layer{layer}_attn_aggregate', f'layer{layer}_moe_broadcast')
    
    # MoE phase connections
    for moe_gpu in range(12, 16):
        dot.edge(f'layer{layer}_moe_broadcast', f'layer{layer}_gate_gpu{moe_gpu}')
        
        # Connect gate to experts (dashed lines for routing)
        for exp in range(4):
            expert_num = (moe_gpu-12)*4 + exp
            dot.edge(f'layer{layer}_gate_gpu{moe_gpu}', f'layer{layer}_expert{moe_gpu}_{expert_num}', style='dashed')
        
        # Connect experts to aggregation
        for exp in range(4):
            expert_num = (moe_gpu-12)*4 + exp
            dot.edge(f'layer{layer}_expert{moe_gpu}_{expert_num}', f'layer{layer}_moe_aggregate_gpu{moe_gpu}')
        
        dot.edge(f'layer{layer}_moe_aggregate_gpu{moe_gpu}', f'layer{layer}_moe_norm_gpu{moe_gpu}')
    
    # Layer to layer connections
    if layer < 4:
        for moe_gpu in range(12, 16):
            # Broadcast MoE outputs to next layer attention GPUs
            for next_gpu_id in range(12):
                dot.edge(f'layer{layer}_moe_norm_gpu{moe_gpu}', f'layer{layer+1}_qkv_gpu{next_gpu_id}')

# Final connections to output
for moe_gpu in range(12, 16):
    dot.edge(f'layer4_moe_norm_gpu{moe_gpu}', 'output')

# Save the DAG
dot.render('../outputs/2025-10-17-10-50-08/ma_separation_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-10-17-10-50-08/ma_separation_dag', format='svg', cleanup=False)

print("MA Separation DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-10-17-10-50-08/ma_separation_dag.dot")
print("- ../outputs/2025-10-17-10-50-08/ma_separation_dag.svg")