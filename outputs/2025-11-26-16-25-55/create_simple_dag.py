import graphviz

def create_main_deployment_dag():
    """Create the main deployment DAG showing combined strategies"""
    dot = graphviz.Digraph('MOE_Main_Deployment', format='svg',
                          graph_attr={'rankdir': 'TB', 'compound': 'true'})
    
    # Model parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # Input node
    dot.node('input', f'Input Layer\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Dense layers (first 3)
    for layer in range(3):
        dot.node(f'dense_{layer}', f'Dense Layer {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7',
                shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # MoE layers (58 total)
    for layer in range(3, 61):
        # MHA component
        dot.node(f'mha_{layer}', f'MHA Layer {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: {(layer%4)*64}',
                shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Gate computation
        dot.node(f'gate_{layer}', f'Expert Gate {layer}\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {(layer%4)*64}',
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # Expert placement (16 experts, one per GPU)
        for expert in range(16):
            expert_gpu = (layer * 16 + expert) % 928
            dot.node(f'expert_{layer}_{expert}', f'Expert {expert}\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                    shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Expert aggregation
        dot.node(f'aggregate_{layer}', f'Expert Aggregate {layer}\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {(layer%4)*64}',
                shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Output node
    dot.node('output', f'Output Layer\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'dense_0')
    dot.edge('dense_0', 'dense_1')
    dot.edge('dense_1', 'dense_2')
    dot.edge('dense_2', 'mha_3')
    
    for layer in range(3, 61):
        dot.edge(f'mha_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'aggregate_{layer}')
        
        # Connect to all experts
        for expert in range(16):
            dot.edge(f'gate_{layer}', f'expert_{layer}_{expert}', style='dashed')
            dot.edge(f'expert_{layer}_{expert}', f'aggregate_{layer}', style='dashed')
        
        if layer < 60:
            dot.edge(f'aggregate_{layer}', f'mha_{layer+1}')
        else:
            dot.edge(f'aggregate_{layer}', 'output')
    
    # Save the DAG
    dot.render('../outputs/2025-11-26-16-25-55/moe_main_deployment', cleanup=False)
    
    return dot.source

def create_expert_parallelism_dag():
    """Create DAG showing expert parallelism"""
    dot = graphviz.Digraph('Expert_Parallelism_DAG', format='svg',
                          graph_attr={'rankdir': 'LR'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # Input to MoE layer
    dot.node('mha_input', f'MHA Input\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Gate
    dot.node('gate', f'Expert Gate\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Token split
    dot.node('token_split', f'Token Split\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # 16 experts (one per GPU)
    for expert in range(16):
        expert_gpu = expert
        
        # Expert MLP components
        dot.node(f'expert_{expert}_linear1', f'Expert {expert} Linear 1\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'expert_{expert}_gelu', f'Expert {expert} GELU\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'expert_{expert}_linear2', f'Expert {expert} Linear 2\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregation
    dot.node('expert_aggregate', f'Expert Aggregate\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Output
    dot.node('mha_output', f'MHA Output\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('mha_input', 'gate')
    dot.edge('gate', 'token_split')
    
    for expert in range(16):
        dot.edge('token_split', f'expert_{expert}_linear1', style='dashed', label=f'send to GPU {expert}')
        dot.edge(f'expert_{expert}_linear1', f'expert_{expert}_gelu')
        dot.edge(f'expert_{expert}_gelu', f'expert_{expert}_linear2')
        dot.edge(f'expert_{expert}_linear2', 'expert_aggregate', style='dashed', label=f'recv from GPU {expert}')
    
    dot.edge('expert_aggregate', 'mha_output')
    
    # Save the DAG
    dot.render('../outputs/2025-11-26-16-25-55/expert_parallelism_dag', cleanup=False)
    
    return dot.source

def create_combined_parallel_dag():
    """Create DAG showing combined strategies"""
    dot = graphviz.Digraph('Combined_Parallel_DAG', format='svg',
                          graph_attr={'rankdir': 'TB'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # Input
    dot.node('input', f'Global Batch\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Data Parallel split
    dot.node('dp_split', f'Data Parallel Split\\n8-way DP\\nGPU: 0-927',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Pipeline stages
    for stage in range(16):
        stage_gpu_start = stage * 58
        stage_gpu_end = (stage + 1) * 58 - 1
        
        if stage == 0:
            dot.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{stage*4+3}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightyellow')
        elif stage == 15:
            dot.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{60}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightgreen')
        else:
            dot.node(f'pp_stage_{stage}', f'Pipeline Stage {stage}\\nLayers {stage*4}-{stage*4+3}\\nGPU: {stage_gpu_start}-{stage_gpu_end}',
                    shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Tensor parallelism within stages
    for stage in range(16):
        dot.node(f'tp_stage_{stage}', f'Tensor Parallel\\nTP=2 within experts\\nGPU: {stage*58}-{stage*58+1}',
                shape='parallelogram', style='dashed', fillcolor='orange')
    
    # Expert parallelism
    for stage in range(16):
        for expert in range(16):
            expert_gpu = stage * 58 + expert
            dot.node(f'expert_{stage}_{expert}', f'Expert {expert}\\n[batch_size={batch_size//8}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                    shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Output
    dot.node('output', f'Output\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'dp_split')
    dot.edge('dp_split', 'pp_stage_0')
    
    for stage in range(16):
        dot.edge(f'pp_stage_{stage}', f'tp_stage_{stage}')
        
        for expert in range(16):
            dot.edge(f'tp_stage_{stage}', f'expert_{stage}_{expert}')
        
        if stage < 15:
            dot.edge(f'expert_{stage}_15', f'pp_stage_{stage+1}')
        else:
            dot.edge(f'expert_{stage}_15', 'output')
    
    # Save the DAG
    dot.render('../outputs/2025-11-26-16-25-55/combined_parallel_dag', cleanup=False)
    
    return dot.source

def create_tensor_shapes_dag():
    """Create DAG showing detailed tensor shapes and transformations"""
    dot = graphviz.Digraph('Tensor_Shapes_DAG', format='svg',
                          graph_attr={'rankdir': 'TB'})
    
    # Parameters
    batch_size = 32
    seq_len = 2048
    token_dim = 7168
    heads = 128
    d_k = 128
    mlp_hidden = 2048
    
    # Input
    dot.node('input', f'Input Tensor\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Layer normalization
    dot.node('layernorm1', f'LayerNorm\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # QKV projection
    dot.node('qkv_proj', f'QKV Projection\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k*3}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Attention computation
    dot.node('attention', f'MHA\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}] -> [batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightcoral')
    
    # Attention output
    dot.node('attn_out', f'Attention Output\\n[batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightyellow')
    
    # Residual connection
    dot.node('residual1', f'Residual Add\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Layer norm 2
    dot.node('layernorm2', f'LayerNorm 2\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Gate
    dot.node('gate', f'Expert Gate\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, top_k=2]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='orange')
    
    # Token split
    dot.node('token_split', f'Token Split\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Expert MLPs
    for expert in range(16):
        expert_gpu = expert
        dot.node(f'expert_{expert}_linear1', f'Expert {expert} Linear 1\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        dot.node(f'expert_{expert}_gelu', f'Expert {expert} GELU\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'expert_{expert}_linear2', f'Expert {expert} Linear 2\\n[batch_size={batch_size}, tokens_per_expert, hidden={mlp_hidden}] -> [batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}]\\nGPU: {expert_gpu}',
                shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Expert aggregate
    dot.node('expert_aggregate', f'Expert Aggregate\\n[batch_size={batch_size}, tokens_per_expert, token_dim={token_dim}] -> [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='dashed', fillcolor='yellow')
    
    # Final residual
    dot.node('residual2', f'Residual Add 2\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='parallelogram', style='filled', fillcolor='purple')
    
    # Output
    dot.node('output', f'Layer Output\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect nodes
    dot.edge('input', 'layernorm1')
    dot.edge('layernorm1', 'qkv_proj')
    dot.edge('qkv_proj', 'attention')
    dot.edge('attention', 'attn_out')
    dot.edge('attn_out', 'residual1')
    dot.edge('residual1', 'layernorm2')
    dot.edge('layernorm2', 'gate')
    dot.edge('gate', 'token_split')
    
    for expert in range(16):
        dot.edge('token_split', f'expert_{expert}_linear1', style='dashed', label=f'send to GPU {expert}')
        dot.edge(f'expert_{expert}_linear1', f'expert_{expert}_gelu')
        dot.edge(f'expert_{expert}_gelu', f'expert_{expert}_linear2')
        dot.edge(f'expert_{expert}_linear2', 'expert_aggregate', style='dashed', label=f'recv from GPU {expert}')
    
    dot.edge('expert_aggregate', 'residual2')
    dot.edge('residual2', 'output')
    
    # Save the DAG
    dot.render('../outputs/2025-11-26-16-25-55/tensor_shapes_dag', cleanup=False)
    
    return dot.source

if __name__ == "__main__":
    print("Creating main deployment DAG...")
    main_dag = create_main_deployment_dag()
    
    print("Creating expert parallelism DAG...")
    expert_dag = create_expert_parallelism_dag()
    
    print("Creating combined parallelism DAG...")
    combined_dag = create_combined_parallel_dag()
    
    print("Creating tensor shapes DAG...")
    shapes_dag = create_tensor_shapes_dag()
    
    print("All DAGs created successfully!")
    print("Files saved in ../outputs/2025-11-26-16-25-55/")