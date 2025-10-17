import graphviz

# Create Pipeline Parallelism 2 DAG
dot = graphviz.Digraph('PP2_Standalone', comment='Standalone Pipeline Parallelism 2 Configuration')
dot.attr(rankdir='TB', size='30,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Global attributes
dot.attr(label='Standalone Pipeline Parallelism 2\n16 GPUs total\n2 stages × 8 GPUs each')
dot.attr(fontsize='20')

# Input Layer
dot.node('input', 'INPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 0-7', shape='ellipse', fillcolor='lightblue')

# Stage 0: GPUs 0-7, Layers 0-1
for stage in range(2):
    stage_name = f"stage{stage}"
    gpu_start = stage * 8
    gpu_end = gpu_start + 8
    
    # Stage boundary
    dot.node(f'{stage_name}_boundary', f'Stage {stage} Boundary\nGPUs {gpu_start}-{gpu_end-1}\nLayers {stage*2}-{stage*2+1}', 
             shape='ellipse', fillcolor='lightblue', peripheries='2')
    
    for layer in range(stage*2, stage*2+2):
        # Standard layer processing (no tensor parallelism)
        for gpu_id in range(gpu_start, gpu_end):
            # QKV Projection (replicated)
            dot.node(f'layer{layer}_qkv_gpu{gpu_id}', 
                    f'L{layer}_QKV_Projection_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,12288]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Multi-Head Attention
            dot.node(f'layer{layer}_attn_gpu{gpu_id}', 
                    f'L{layer}_Multi-Head_Attention_GPU{gpu_id}\nInput: [1024,2048,12288]\nOutput: [1024,2048,4096]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Residual add & LayerNorm
            dot.node(f'layer{layer}_attn_norm_gpu{gpu_id}', 
                    f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,4096]x2\nOutput: [1024,2048,4096]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # MoE Gate
            dot.node(f'layer{layer}_gate_gpu{gpu_id}', 
                    f'L{layer}_Gate_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,16 experts]\nTop-2 routing\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Expert processing (all 16 experts on each GPU)
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

# Pipeline communication nodes
for stage in range(1):
    dot.node(f'pipeline_stage{stage}_{stage+1}', 
            f'Pipeline_Stage{stage}_to_{stage+1}\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nCross-stage communication\nGPU: 7 -> 8',
            shape='parallelogram', fillcolor='yellow')

# Output Layer
dot.node('output', 'OUTPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 8-15', shape='ellipse', fillcolor='lightblue')

# Connect all nodes
# Input to Stage 0
for gpu_id in range(8):
    dot.edge('input', f'layer0_qkv_gpu{gpu_id}')

# Connect within Stage 0
for layer in range(2):
    for gpu_id in range(8):
        # Attention path
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        # MoE path
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_gate_gpu{gpu_id}')
        
        # Connect gate to experts
        for expert in range(16):
            dot.edge(f'layer{layer}_gate_gpu{gpu_id}', f'layer{layer}_expert{gpu_id}_{expert}', style='dashed')
        
        # Connect experts back to aggregation
        for expert in range(16):
            dot.edge(f'layer{layer}_expert{gpu_id}_{expert}', f'layer{layer}_moe_agg_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_output_gpu{gpu_id}')

# Pipeline stage connections
for gpu_id in range(8):
    dot.edge(f'layer1_output_gpu{gpu_id}', f'pipeline_stage0_1')

for gpu_id in range(8, 16):
    dot.edge(f'pipeline_stage0_1', f'layer2_qkv_gpu{gpu_id}')

# Connect within Stage 1
for layer in range(2, 4):
    for gpu_id in range(8, 16):
        # Attention path
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        # MoE path
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_gate_gpu{gpu_id}')
        
        # Connect gate to experts
        for expert in range(16):
            dot.edge(f'layer{layer}_gate_gpu{gpu_id}', f'layer{layer}_expert{gpu_id}_{expert}', style='dashed')
        
        # Connect experts back to aggregation
        for expert in range(16):
            dot.edge(f'layer{layer}_expert{gpu_id}_{expert}', f'layer{layer}_moe_agg_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_output_gpu{gpu_id}')

# Final connections to output
for gpu_id in range(8, 16):
    dot.edge(f'layer3_output_gpu{gpu_id}', 'output')

# Save the DAG
dot.render('../outputs/2025-10-17-10-50-08/pp2_standalone_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-10-17-10-50-08/pp2_standalone_dag', format='svg', cleanup=False)

print("Standalone PP2 DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-10-17-10-50-08/pp2_standalone_dag.dot")
print("- ../outputs/2025-10-17-10-50-08/pp2_standalone_dag.svg")