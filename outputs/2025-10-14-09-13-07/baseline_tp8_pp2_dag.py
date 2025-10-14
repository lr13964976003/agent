import graphviz

# Create DAG for baseline: TP=8, PP=2 with colocated experts
dot = graphviz.Digraph('baseline_tp8_pp2_moe', comment='Baseline MoE TP=8 PP=2 Deployment')
dot.attr(rankdir='TB', size='20,30')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/output
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/aggregation

# Global input
dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\\nGPU: all GPUs', 
         shape='ellipse', fillcolor='lightblue')

# Pipeline Stage 0 (Layers 0-1) - GPUs 0-7
with dot.subgraph(name='cluster_stage0') as stage0:
    stage0.attr(label='Pipeline Stage 0 (Layers 0-1)\\nGPUs 0-7', style='dashed')
    
    # Layer 0
    with stage0.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0', style='dotted')
        
        # Attention across 8 GPUs (TP=8)
        for tp_rank in range(8):
            # QKV projection
            stage0.node(f'l0_qkv_tp{tp_rank}', 
                       f'QKV Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            
            # Attention computation
            stage0.node(f'l0_attn_tp{tp_rank}',
                       f'Multi-Head Attention\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            
            # Output projection
            stage0.node(f'l0_out_proj_tp{tp_rank}',
                       f'Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            
            # Residual add
            stage0.node(f'l0_residual_tp{tp_rank}',
                       f'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
        
        # Gate computation (replicated on each GPU)
        for tp_rank in range(8):
            stage0.node(f'l0_gate_tp{tp_rank}',
                       f'Gate Network\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
        
        # Expert routing (colocated 8 experts per GPU)
        for tp_rank in range(8):
            for expert_id in range(8):  # 8 experts per GPU
                stage0.node(f'l0_exp{expert_id}_gpu{tp_rank}',
                           f'Expert {expert_id}\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {tp_rank}',
                           shape='rectangle', fillcolor='lightgreen')
            
            # Expert aggregation
            stage0.node(f'l0_exp_agg_tp{tp_rank}',
                       f'Expert Aggregation\\nInput: [16,tokens,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
            
            # Final residual add for layer 0
            stage0.node(f'l0_final_res_tp{tp_rank}',
                       f'Layer 0 Final Residual\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')

    # Layer 1 (same structure as layer 0)
    with stage0.subgraph(name='cluster_layer1') as layer1:
        layer1.attr(label='Layer 1', style='dotted')
        
        for tp_rank in range(8):
            stage0.node(f'l1_qkv_tp{tp_rank}', 
                       f'QKV Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            stage0.node(f'l1_attn_tp{tp_rank}',
                       f'Multi-Head Attention\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            stage0.node(f'l1_out_proj_tp{tp_rank}',
                       f'Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='rectangle', fillcolor='lightgreen')
            stage0.node(f'l1_residual_tp{tp_rank}',
                       f'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage0.node(f'l1_gate_tp{tp_rank}',
                       f'Gate Network\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
            
            for expert_id in range(8):
                stage0.node(f'l1_exp{expert_id}_gpu{tp_rank}',
                           f'Expert {expert_id}\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {tp_rank}',
                           shape='rectangle', fillcolor='lightgreen')
            
            stage0.node(f'l1_exp_agg_tp{tp_rank}',
                       f'Expert Aggregation\\nInput: [16,tokens,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage0.node(f'l1_final_res_tp{tp_rank}',
                       f'Layer 1 Final Residual\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {tp_rank}',
                       shape='parallelogram', fillcolor='lightyellow')

# Pipeline Stage 1 (Layers 2-3) - GPUs 8-15
with dot.subgraph(name='cluster_stage1') as stage1:
    stage1.attr(label='Pipeline Stage 1 (Layers 2-3)\\nGPUs 8-15', style='dashed')
    
    # Layer 2
    with stage1.subgraph(name='cluster_layer2') as layer2:
        layer2.attr(label='Layer 2', style='dotted')
        
        for tp_rank in range(8):
            gpu_id = tp_rank + 8
            stage1.node(f'l2_qkv_tp{tp_rank}', 
                       f'QKV Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l2_attn_tp{tp_rank}',
                       f'Multi-Head Attention\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l2_out_proj_tp{tp_rank}',
                       f'Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l2_residual_tp{tp_rank}',
                       f'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage1.node(f'l2_gate_tp{tp_rank}',
                       f'Gate Network\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            
            for expert_id in range(8):
                stage1.node(f'l2_exp{expert_id}_gpu{gpu_id}',
                           f'Expert {expert_id}\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
            
            stage1.node(f'l2_exp_agg_tp{tp_rank}',
                       f'Expert Aggregation\\nInput: [16,tokens,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage1.node(f'l2_final_res_tp{tp_rank}',
                       f'Layer 2 Final Residual\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')

    # Layer 3
    with stage1.subgraph(name='cluster_layer3') as layer3:
        layer3.attr(label='Layer 3', style='dotted')
        
        for tp_rank in range(8):
            gpu_id = tp_rank + 8
            stage1.node(f'l3_qkv_tp{tp_rank}', 
                       f'QKV Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16,512]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l3_attn_tp{tp_rank}',
                       f'Multi-Head Attention\\nInput: [1024,10000,16,512]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l3_out_proj_tp{tp_rank}',
                       f'Output Projection\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='rectangle', fillcolor='lightgreen')
            stage1.node(f'l3_residual_tp{tp_rank}',
                       f'Residual Add\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage1.node(f'l3_gate_tp{tp_rank}',
                       f'Gate Network\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,16]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            
            for expert_id in range(8):
                stage1.node(f'l3_exp{expert_id}_gpu{gpu_id}',
                           f'Expert {expert_id}\\nInput: [tokens,8192]\\nOutput: [tokens,8192]\\nGPU: {gpu_id}',
                           shape='rectangle', fillcolor='lightgreen')
            
            stage1.node(f'l3_exp_agg_tp{tp_rank}',
                       f'Expert Aggregation\\nInput: [16,tokens,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')
            stage1.node(f'l3_final_res_tp{tp_rank}',
                       f'Layer 3 Final Residual\\nInput: [1024,10000,8192], [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: {gpu_id}',
                       shape='parallelogram', fillcolor='lightyellow')

# Global output
dot.node('output', 'Output\\nInput: [1024,10000,8192]\\nOutput: [1024,10000,8192]\\nGPU: all GPUs', 
         shape='ellipse', fillcolor='lightblue')

# Connections for Layer 0
for tp_rank in range(8):
    # Attention path
    dot.edge('input', f'l0_qkv_tp{tp_rank}')
    dot.edge(f'l0_qkv_tp{tp_rank}', f'l0_attn_tp{tp_rank}')
    dot.edge(f'l0_attn_tp{tp_rank}', f'l0_out_proj_tp{tp_rank}')
    dot.edge(f'l0_out_proj_tp{tp_rank}', f'l0_residual_tp{tp_rank}')
    dot.edge('input', f'l0_residual_tp{tp_rank}')  # Residual connection
    
    # Expert path
    dot.edge(f'l0_residual_tp{tp_rank}', f'l0_gate_tp{tp_rank}')
    
    # Expert routing (dashed for selection)
    for expert_id in range(8):
        dot.edge(f'l0_gate_tp{tp_rank}', f'l0_exp{expert_id}_gpu{tp_rank}', style='dashed')
        dot.edge(f'l0_residual_tp{tp_rank}', f'l0_exp{expert_id}_gpu{tp_rank}')  # Token routing
        dot.edge(f'l0_exp{expert_id}_gpu{tp_rank}', f'l0_exp_agg_tp{tp_rank}')
    
    dot.edge(f'l0_exp_agg_tp{tp_rank}', f'l0_final_res_tp{tp_rank}')
    dot.edge(f'l0_residual_tp{tp_rank}', f'l0_final_res_tp{tp_rank}')  # Residual

# Connections between layers within stage 0
for tp_rank in range(8):
    dot.edge(f'l0_final_res_tp{tp_rank}', f'l1_qkv_tp{tp_rank}')
    dot.edge(f'l1_qkv_tp{tp_rank}', f'l1_attn_tp{tp_rank}')
    dot.edge(f'l1_attn_tp{tp_rank}', f'l1_out_proj_tp{tp_rank}')
    dot.edge(f'l1_out_proj_tp{tp_rank}', f'l1_residual_tp{tp_rank}')
    dot.edge(f'l0_final_res_tp{tp_rank}', f'l1_residual_tp{tp_rank}')  # Residual
    
    # Expert path layer 1
    dot.edge(f'l1_residual_tp{tp_rank}', f'l1_gate_tp{tp_rank}')
    for expert_id in range(8):
        dot.edge(f'l1_gate_tp{tp_rank}', f'l1_exp{expert_id}_gpu{tp_rank}', style='dashed')
        dot.edge(f'l1_residual_tp{tp_rank}', f'l1_exp{expert_id}_gpu{tp_rank}')
        dot.edge(f'l1_exp{expert_id}_gpu{tp_rank}', f'l1_exp_agg_tp{tp_rank}')
    
    dot.edge(f'l1_exp_agg_tp{tp_rank}', f'l1_final_res_tp{tp_rank}')
    dot.edge(f'l1_residual_tp{tp_rank}', f'l1_final_res_tp{tp_rank}')  # Residual

# Pipeline communication between stages
for tp_rank in range(8):
    gpu_src = tp_rank
    gpu_dst = tp_rank + 8
    dot.edge(f'l1_final_res_tp{tp_rank}', f'l2_qkv_tp{tp_rank}', 
             label=f'Pipeline P2P\\nGPU {gpu_src} → GPU {gpu_dst}')

# Connections for Layer 2
for tp_rank in range(8):
    dot.edge(f'l2_qkv_tp{tp_rank}', f'l2_attn_tp{tp_rank}')
    dot.edge(f'l2_attn_tp{tp_rank}', f'l2_out_proj_tp{tp_rank}')
    dot.edge(f'l2_out_proj_tp{tp_rank}', f'l2_residual_tp{tp_rank}')
    dot.edge(f'l2_qkv_tp{tp_rank}', f'l2_residual_tp{tp_rank}')  # Residual from pipeline input
    
    # Expert path layer 2
    dot.edge(f'l2_residual_tp{tp_rank}', f'l2_gate_tp{tp_rank}')
    for expert_id in range(8):
        dot.edge(f'l2_gate_tp{tp_rank}', f'l2_exp{expert_id}_gpu{tp_rank+8}', style='dashed')
        dot.edge(f'l2_residual_tp{tp_rank}', f'l2_exp{expert_id}_gpu{tp_rank+8}')
        dot.edge(f'l2_exp{expert_id}_gpu{tp_rank+8}', f'l2_exp_agg_tp{tp_rank}')
    
    dot.edge(f'l2_exp_agg_tp{tp_rank}', f'l2_final_res_tp{tp_rank}')
    dot.edge(f'l2_residual_tp{tp_rank}', f'l2_final_res_tp{tp_rank}')  # Residual

# Connections for Layer 3
for tp_rank in range(8):
    dot.edge(f'l2_final_res_tp{tp_rank}', f'l3_qkv_tp{tp_rank}')
    dot.edge(f'l3_qkv_tp{tp_rank}', f'l3_attn_tp{tp_rank}')
    dot.edge(f'l3_attn_tp{tp_rank}', f'l3_out_proj_tp{tp_rank}')
    dot.edge(f'l3_out_proj_tp{tp_rank}', f'l3_residual_tp{tp_rank}')
    dot.edge(f'l2_final_res_tp{tp_rank}', f'l3_residual_tp{tp_rank}')  # Residual
    
    # Expert path layer 3
    dot.edge(f'l3_residual_tp{tp_rank}', f'l3_gate_tp{tp_rank}')
    for expert_id in range(8):
        dot.edge(f'l3_gate_tp{tp_rank}', f'l3_exp{expert_id}_gpu{tp_rank+8}', style='dashed')
        dot.edge(f'l3_residual_tp{tp_rank}', f'l3_exp{expert_id}_gpu{tp_rank+8}')
        dot.edge(f'l3_exp{expert_id}_gpu{tp_rank+8}', f'l3_exp_agg_tp{tp_rank}')
    
    dot.edge(f'l3_exp_agg_tp{tp_rank}', f'l3_final_res_tp{tp_rank}')
    dot.edge(f'l3_residual_tp{tp_rank}', f'l3_final_res_tp{tp_rank}')  # Residual
    dot.edge(f'l3_final_res_tp{tp_rank}', 'output')

# Save files
dot.render('./outputs/2025-10-14-09-13-07/baseline_tp8_pp2_moe', format='dot')
dot.render('./outputs/2025-10-14-09-13-07/baseline_tp8_pp2_moe', format='svg')

print("Baseline TP=8 PP=2 DAG generated successfully!")
print(f"DOT file: ./outputs/2025-10-14-09-13-07/baseline_tp8_pp2_moe.dot")
print(f"SVG file: ./outputs/2025-10-14-09-13-07/baseline_tp8_pp2_moe.svg")