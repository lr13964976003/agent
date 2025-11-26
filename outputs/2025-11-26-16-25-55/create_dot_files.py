import graphviz

def create_dot_files():
    """Create DOT files for all DAGs"""
    
    # Model parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # 1. Main Deployment DAG
    dot1 = graphviz.Digraph('MOE_Main_Deployment', format='dot')
    
    # Input node
    dot1.node('input', f'Input Layer\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Dense layers (first 3)
    for layer in range(3):
        dot1.node(f'dense_{layer}', f'Dense Layer {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7',
                shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # MoE layers (58 total)
    for layer in range(3, 61):
        # MHA component
        dot1.node(f'mha_{layer}', f'MHA Layer {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: {(layer%4)*64}',
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Gate computation
        dot1.node(f'gate_{layer}', f'Expert Gate {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {(layer%4)*64}',
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Expert placement (16 experts, one per GPU)
        for expert in range(16):
            expert_gpu = (layer * 16 + expert) % 928
            dot1.node(f'expert_{layer}_{expert}', f'Expert {expert}\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                    shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Expert aggregation
        dot1.node(f'aggregate_{layer}', f'Expert Aggregate {layer}\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {(layer%4)*64}',
                shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Output node
    dot1.node('output', f'Output Layer\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot1.edge('input', 'dense_0')
    dot1.edge('dense_0', 'dense_1')
    dot1.edge('dense_1', 'dense_2')
    dot1.edge('dense_2', 'mha_3')
    
    for layer in range(3, 61):
        dot1.edge(f'mha_{layer}', f'gate_{layer}')
        dot1.edge(f'gate_{layer}', f'aggregate_{layer}')
        
        # Connect to all experts
        for expert in range(16):
            dot1.edge(f'gate_{layer}', f'expert_{layer}_{expert}', style='dashed')
            dot1.edge(f'expert_{layer}_{expert}', f'aggregate_{layer}', style='dashed')
        
        if layer < 60:
            dot1.edge(f'aggregate_{layer}', f'mha_{layer+1}')
        else:
            dot1.edge(f'aggregate_{layer}', 'output')
    
    # Save DOT file
    with open('../outputs/2025-11-26-16-25-55/moe_main_deployment.dot', 'w') as f:
        f.write(dot1.source)
    
    # 2. Expert Parallelism DAG
    dot2 = graphviz.Digraph('Expert_Parallelism_DAG', format='dot')
    
    # Input to MoE layer
    dot2.node('mha_input', f'MHA Input\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Gate
    dot2.node('gate', f'Expert Gate\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Token split
    dot2.node('token_split', f'Token Split\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # 16 experts (one per GPU)
    for expert in range(16):
        expert_gpu = expert
        
        # Expert MLP components
        dot2.node(f'expert_{expert}_linear1', f'Expert {expert} Linear 1\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot2.node(f'expert_{expert}_gelu', f'Expert {expert} GELU\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot2.node(f'expert_{expert}_linear2', f'Expert {expert} Linear 2\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregation
    dot2.node('expert_aggregate', f'Expert Aggregate\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Output
    dot2.node('mha_output', f'MHA Output\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot2.edge('mha_input', 'gate')
    dot2.edge('gate', 'token_split')
    
    for expert in range(16):
        dot2.edge('token_split', f'expert_{expert}_linear1', style='dashed', label=f'send to GPU {expert}')
        dot2.edge(f'expert_{expert}_linear1', f'expert_{expert}_gelu')
        dot2.edge(f'expert_{expert}_gelu', f'expert_{expert}_linear2')
        dot2.edge(f'expert_{expert}_linear2', 'expert_aggregate', style='dashed', label=f'recv from GPU {expert}')
    
    dot2.edge('expert_aggregate', 'mha_output')
    
    # Save DOT file
    with open('../outputs/2025-11-26-16-25-55/expert_parallelism_dag.dot', 'w') as f:
        f.write(dot2.source)
    
    # 3. Combined Parallelism DAG
    dot3 = graphviz.Digraph('Combined_Parallel_DAG', format='dot')
    
    # Input
    dot3.node('input', f'Global Batch\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Data Parallel split
    dot3.node('dp_split', f'Data Parallel Split\\n8-way DP\\nGPU: 0-927',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Pipeline stages
    for stage in range(16):
        stage_gpu_start = stage * 58
        stage_gpu_end = (stage + 1) * 58 - 1
        
        if stage == 0:
            dot3.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{stage*4+3}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightyellow')
        elif stage == 15:
            dot3.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{60}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightgreen')
        else:
            dot3.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{stage*4+3}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Tensor parallelism within stages
    for stage in range(16):
        dot3.node(f'tp_stage_{stage}', f'Tensor Parallel\\nTP=2 within experts\\nGPU: {stage*58}-{stage*58+1}',
                shape='parallelogram', style='dashed', fillcolor='orange')
    
    # Expert parallelism
    for stage in range(16):
        for expert in range(16):
            expert_gpu = stage * 58 + expert
            dot3.node(f'expert_{stage}_{expert}', f'Expert {expert}\\n[batch_size={batch_size//8}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                    shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Output
    dot3.node('output', f'Output\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot3.edge('input', 'dp_split')
    dot3.edge('dp_split', 'pp_stage_0')
    
    for stage in range(16):
        dot3.edge(f'pp_stage_{stage}', f'tp_stage_{stage}')
        
        for expert in range(16):
            dot3.edge(f'tp_stage_{stage}', f'expert_{stage}_{expert}')
        
        if stage < 15:
            dot3.edge(f'expert_{stage}_15', f'pp_stage_{stage+1}')
        else:
            dot3.edge(f'expert_{stage}_15', 'output')
    
    # Save DOT file
    with open('../outputs/2025-11-26-16-25-55/combined_parallel_dag.dot', 'w') as f:
        f.write(dot3.source)
    
    # 4. Tensor Shapes DAG
    dot4 = graphviz.Digraph('Tensor_Shapes_DAG', format='dot')
    
    # Input
    dot4.node('input', f'Input Tensor\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Layer normalization
    dot4.node('layernorm1', f'LayerNorm\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # QKV projection
    dot4.node('qkv_proj', f'QKV Projection\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k*3}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Attention computation
    dot4.node('attention', f'MHA\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}] -> [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Attention output
    dot4.node('attn_out', f'Attention Output\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Residual connection
    dot4.node('residual1', f'Residual Add\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Layer norm 2
    dot4.node('layernorm2', f'LayerNorm 2\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Gate
    dot4.node('gate', f'Expert Gate\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Token split
    dot4.node('token_split', f'Token Split\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Expert MLPs
    for expert in range(16):
        expert_gpu = expert
        dot4.node(f'expert_{expert}_linear1', f'Expert {expert} Linear 1\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot4.node(f'expert_{expert}_gelu', f'Expert {expert} GELU\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot4.node(f'expert_{expert}_linear2', f'Expert {expert} Linear 2\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregate
    dot4.node('expert_aggregate', f'Expert Aggregate\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Final residual
    dot4.node('residual2', f'Residual Add 2\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Output
    dot4.node('output', f'Layer Output\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot4.edge('input', 'layernorm1')
    dot4.edge('layernorm1', 'qkv_proj')
    dot4.edge('qkv_proj', 'attention')
    dot4.edge('attention', 'attn_out')
    dot4.edge('attn_out', 'residual1')
    dot4.edge('residual1', 'layernorm2')
    dot4.edge('layernorm2', 'gate')
    dot4.edge('gate', 'token_split')
    
    for expert in range(16):
        dot4.edge('token_split', f'expert_{expert}_linear1', style='dashed', label=f'send to GPU {expert}')
        dot4.edge(f'expert_{expert}_linear1', f'expert_{expert}_gelu')
        dot4.edge(f'expert_{expert}_gelu', f'expert_{expert}_linear2')
        dot4.edge(f'expert_{expert}_linear2', 'expert_aggregate', style='dashed', label=f'recv from GPU {expert}')
    
    dot4.edge('expert_aggregate', 'residual2')
    dot4.edge('residual2', 'output')
    
    # Save DOT file
    with open('../outputs/2025-11-26-16-25-55/tensor_shapes_dag.dot', 'w') as f:
        f.write(dot4.source)
    
    print("DOT files created successfully!")

if __name__ == "__main__":
    create_dot_files()