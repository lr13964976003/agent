import graphviz

# Create Baseline DAG (TP=8, PP=2)
dot = graphviz.Digraph('TP8_PP2_Baseline', comment='Baseline: Tensor Parallelism 8 + Pipeline Parallelism 2')
dot.attr(rankdir='TB', size='30,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication

# Global attributes
dot.attr(label='Baseline: TP=8, PP=2 Hybrid\n16 GPUs total\nTensor Parallelism 8 per stage, Pipeline 2 stages')
dot.attr(fontsize='20')

# Input Layer
dot.node('input', 'INPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 0-7', shape='ellipse', fillcolor='lightblue')

# Stage 0: GPUs 0-7, Layers 0-1
for stage in [0, 1]:
    stage_name = f"stage{stage}"
    gpu_start = stage * 8
    gpu_end = gpu_start + 8
    
    # Stage header
    dot.node(f'{stage_name}_header', f'Stage {stage} (GPUs {gpu_start}-{gpu_end-1})\nLayers {stage*2}-{stage*2+1}', 
             shape='ellipse', fillcolor='lightblue', peripheries='2')
    
    for layer in range(stage*2, stage*2+2):
        # Tensor parallel attention across 8 GPUs
        for gpu_offset in range(8):
            gpu_id = gpu_start + gpu_offset
            
            # QKV Projection with tensor parallelism
            dot.node(f'layer{layer}_qkv_gpu{gpu_id}', 
                    f'L{layer}_QKV_Projection_GPU{gpu_id}\nInput: [1024,2048,4096]\nOutput: [1024,2048,3*512]\nHeads: {gpu_offset*4}-{(gpu_offset+1)*4-1}\nTP Slice: {gpu_offset}\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Multi-Head Attention with tensor parallelism
            dot.node(f'layer{layer}_attn_gpu{gpu_id}', 
                    f'L{layer}_Multi-Head_Attention_GPU{gpu_id}\nInput: [1024,2048,512]x8\nOutput: [1024,2048,512]\nTP Reduction\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Attention output projection
            dot.node(f'layer{layer}_attn_proj_gpu{gpu_id}', 
                    f'L{layer}_Attn_Projection_GPU{gpu_id}\nInput: [1024,2048,512]\nOutput: [1024,2048,512]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Residual add & LayerNorm
            dot.node(f'layer{layer}_attn_norm_gpu{gpu_id}', 
                    f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,512]x2\nOutput: [1024,2048,512]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # MoE with tensor parallelism
            # Gate computation
            dot.node(f'layer{layer}_gate_gpu{gpu_id}', 
                    f'L{layer}_Gate_GPU{gpu_id}\nInput: [1024,2048,512]\nOutput: [1024,2048,16 experts]\nTP: expert routing\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Expert processing
            for expert in range(16):
                dot.node(f'layer{layer}_expert{gpu_id}_{expert}', 
                        f'L{layer}_Expert{expert}_GPU{gpu_id}\nInput: [tokens,512]\nOutput: [tokens,512]\nTP slice\nGPU: {gpu_id}',
                        shape='rectangle', fillcolor='lightgreen')
            
            # MoE output aggregation
            dot.node(f'layer{layer}_moe_agg_gpu{gpu_id}', 
                    f'L{layer}_MoE_Aggregate_GPU{gpu_id}\nInput: 16×[tokens,512]\nOutput: [1024,2048,512]\nTP Reduction\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')
            
            # Final residual add & layer norm
            dot.node(f'layer{layer}_output_gpu{gpu_id}', 
                    f'L{layer}_Residual+LayerNorm_GPU{gpu_id}\nInput: [1024,2048,512]x2\nOutput: [1024,2048,512]\nGPU: {gpu_id}',
                    shape='rectangle', fillcolor='lightgreen')

        # Tensor parallelism communication nodes
        dot.node(f'layer{layer}_allreduce_attn', 
                f'L{layer}_All-Reduce_Attention\nInput: 8×[1024,2048,512]\nOutput: [1024,2048,4096]\nTP All-Reduce',
                shape='diamond', fillcolor='lightcoral')
        
        dot.node(f'layer{layer}_allreduce_moe', 
                f'L{layer}_All-Reduce_MoE\nInput: 8×[1024,2048,512]\nOutput: [1024,2048,4096]\nTP All-Reduce',
                shape='diamond', fillcolor='lightcoral')

# Pipeline communication nodes
for stage in range(1):
    dot.node(f'pipeline_stage{stage}_{stage+1}', 
            f'Pipeline_Stage{stage}_to_{stage+1}\nInput: [1024,2048,4096]\nOutput: [1024,2048,4096]\nCross-stage communication',
            shape='parallelogram', fillcolor='yellow')

# Output Layer
dot.node('output', 'OUTPUT\nInput: [batch_size=1024, seq_len=2048, hidden=4096]\nGPU: 8-15', shape='ellipse', fillcolor='lightblue')

# Connect all nodes
# Input to Stage 0
for gpu_id in range(8):
    dot.edge('input', f'layer0_qkv_gpu{gpu_id}')

# Connect within each layer for Stage 0
for layer in range(2):
    for gpu_id in range(8):
        # Attention path
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_proj_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_proj_gpu{gpu_id}', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        # MoE path
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_gate_gpu{gpu_id}')
        
        # Connect gate to experts
        for expert in range(16):
            dot.edge(f'layer{layer}_gate_gpu{gpu_id}', f'layer{layer}_expert{gpu_id}_{expert}', style='dashed')
        
        # Connect experts back to aggregation
        for expert in range(16):
            dot.edge(f'layer{layer}_expert{gpu_id}_{expert}', f'layer{layer}_moe_agg_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_output_gpu{gpu_id}')
    
    # Tensor parallelism connections
    for gpu_id in range(8):
        dot.edge(f'layer{layer}_attn_proj_gpu{gpu_id}', f'layer{layer}_allreduce_attn')
        dot.edge(f'layer{layer}_allreduce_attn', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_allreduce_moe')
        dot.edge(f'layer{layer}_allreduce_moe', f'layer{layer}_output_gpu{gpu_id}')

# Pipeline stage connections
for stage in range(1):
    for gpu_id in range(8):
        dot.edge(f'layer{stage*2+1}_output_gpu{gpu_id}', f'pipeline_stage{stage}_{stage+1}')
    
    for next_gpu_id in range(8, 16):
        dot.edge(f'pipeline_stage{stage}_{stage+1}', f'layer{(stage+1)*2}_qkv_gpu{next_gpu_id}')

# Stage 1 connections
for layer in range(2, 4):
    for gpu_id in range(8, 16):
        # Attention path
        dot.edge(f'layer{layer}_qkv_gpu{gpu_id}', f'layer{layer}_attn_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_gpu{gpu_id}', f'layer{layer}_attn_proj_gpu{gpu_id}')
        dot.edge(f'layer{layer}_attn_proj_gpu{gpu_id}', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        # MoE path
        dot.edge(f'layer{layer}_attn_norm_gpu{gpu_id}', f'layer{layer}_gate_gpu{gpu_id}')
        
        # Connect gate to experts
        for expert in range(16):
            dot.edge(f'layer{layer}_gate_gpu{gpu_id}', f'layer{layer}_expert{gpu_id}_{expert}', style='dashed')
        
        # Connect experts back to aggregation
        for expert in range(16):
            dot.edge(f'layer{layer}_expert{gpu_id}_{expert}', f'layer{layer}_moe_agg_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_output_gpu{gpu_id}')
    
    # Tensor parallelism connections
    for gpu_id in range(8, 16):
        dot.edge(f'layer{layer}_attn_proj_gpu{gpu_id}', f'layer{layer}_allreduce_attn')
        dot.edge(f'layer{layer}_allreduce_attn', f'layer{layer}_attn_norm_gpu{gpu_id}')
        
        dot.edge(f'layer{layer}_moe_agg_gpu{gpu_id}', f'layer{layer}_allreduce_moe')
        dot.edge(f'layer{layer}_allreduce_moe', f'layer{layer}_output_gpu{gpu_id}')

# Final connections to output
for gpu_id in range(8, 16):
    dot.edge(f'layer3_output_gpu{gpu_id}', 'output')

# Save the DAG
dot.render('../outputs/2025-10-17-10-50-08/baseline_tp8_pp2_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-10-17-10-50-08/baseline_tp8_pp2_dag', format='svg', cleanup=False)

print("Baseline TP8+PP2 DAG generated successfully!")
print("Files saved:")
print("- ../outputs/2025-10-17-10-50-08/baseline_tp8_pp2_dag.dot")
print("- ../outputs/2025-10-17-10-50-08/baseline_tp8_pp2_dag.svg")