import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_deployment', format='svg')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Layer 0 - Attention and MoE
    dot.attr('node', shape='rectangle', fillcolor='lightcoral')
    
    # Multi-Head Attention Layer 0 - Tensor Parallel across 8 GPUs
    for gpu_id in range(8):
        dot.node(f'mha_qkv_l0_g{gpu_id}', 
                f'MHA QKV Linear Layer 0\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\\nType: Column-parallel',
                fillcolor='lightcoral')
        
        dot.node(f'mha_attn_l0_g{gpu_id}',
                f'MHA Attention Layer 0\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]',
                fillcolor='lightcoral')
        
        dot.node(f'mha_out_l0_g{gpu_id}',
                f'MHA Output Linear Layer 0\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, dim=1024]\\nType: Row-parallel',
                fillcolor='lightcoral')
    
    # Residual connections and all-reduce
    dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
    for gpu_id in range(8):
        dot.node(f'allreduce_l0_g{gpu_id}',
                f'All-Reduce Sum\\nLayer 0\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=1024]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                fillcolor='lightyellow')
        
        dot.node(f'residual_l0_g{gpu_id}',
                f'Residual Add\\nLayer 0\\nGPU {gpu_id}\\nInput1: [batch_size=1024, seq_len=10000, dim=8192]\\nInput2: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                fillcolor='lightyellow')
    
    # MoE Layer 0 - Experts distributed across GPUs
    dot.attr('node', shape='rectangle', fillcolor='lightgreen')
    for gpu_id in range(16):
        expert_id = gpu_id % 16
        if gpu_id < 8:
            # First 8 GPUs have experts 0-7 for all layers
            expert_range = f"{expert_id}"
        else:
            # Last 8 GPUs have experts 8-15 for all layers
            expert_range = f"{expert_id}"
        
        dot.node(f'gate_l0_g{gpu_id}',
                f'Gating Network\\nLayer 0\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, num_experts=16]',
                fillcolor='lightgreen')
        
        dot.node(f'expert_l0_e{expert_id}_g{gpu_id}',
                f'Expert {expert_id}\\nLayer 0\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=variable, seq_len=variable, dim=8192]\\nType: MLP [8192→32768→8192]',
                fillcolor='lightgreen')
    
    # Expert aggregation
    dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
    for gpu_id in range(16):
        dot.node(f'aggregate_l0_g{gpu_id}',
                f'Expert Aggregation\\nLayer 0\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                fillcolor='lightyellow')
    
    # Repeat for layers 1, 2, 3 with similar structure
    for layer in range(1, 4):
        # MHA for layer
        dot.attr('node', shape='rectangle', fillcolor='lightcoral')
        for gpu_id in range(8):
            dot.node(f'mha_qkv_l{layer}_g{gpu_id}',
                    f'MHA QKV Linear Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]',
                    fillcolor='lightcoral')
            
            dot.node(f'mha_attn_l{layer}_g{gpu_id}',
                    f'MHA Attention Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]',
                    fillcolor='lightcoral')
            
            dot.node(f'mha_out_l{layer}_g{gpu_id}',
                    f'MHA Output Linear Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=2, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, dim=1024]',
                    fillcolor='lightcoral')
        
        # Residual connections
        dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
        for gpu_id in range(8):
            dot.node(f'allreduce_l{layer}_g{gpu_id}',
                    f'All-Reduce Sum\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=1024]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
            
            dot.node(f'residual_l{layer}_g{gpu_id}',
                    f'Residual Add\\nLayer {layer}\\nGPU {gpu_id}\\nInput1: [batch_size=1024, seq_len=10000, dim=8192]\\nInput2: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
        
        # MoE gates and experts
        dot.attr('node', shape='rectangle', fillcolor='lightgreen')
        for gpu_id in range(16):
            expert_id = gpu_id % 16
            dot.node(f'gate_l{layer}_g{gpu_id}',
                    f'Gating Network\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, num_experts=16]',
                    fillcolor='lightgreen')
            
            dot.node(f'expert_l{layer}_e{expert_id}_g{gpu_id}',
                    f'Expert {expert_id}\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=variable, seq_len=variable, dim=8192]\\nType: MLP [8192→32768→8192]',
                    fillcolor='lightgreen')
        
        # Expert aggregation
        dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
        for gpu_id in range(16):
            dot.node(f'aggregate_l{layer}_g{gpu_id}',
                    f'Expert Aggregation\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create edges for Layer 0
    for gpu_id in range(8):
        dot.edge('input', f'mha_qkv_l0_g{gpu_id}')
        dot.edge(f'mha_qkv_l0_g{gpu_id}', f'mha_attn_l0_g{gpu_id}')
        dot.edge(f'mha_attn_l0_g{gpu_id}', f'mha_out_l0_g{gpu_id}')
        dot.edge(f'mha_out_l0_g{gpu_id}', f'allreduce_l0_g{gpu_id}')
        dot.edge('input', f'residual_l0_g{gpu_id}')
        dot.edge(f'allreduce_l0_g{gpu_id}', f'residual_l0_g{gpu_id}')
    
    # Connect to gates and experts
    for gpu_id in range(16):
        dot.edge(f'residual_l0_g{gpu_id % 8}', f'gate_l0_g{gpu_id}')
        dot.edge(f'gate_l0_g{gpu_id}', f'expert_l0_e{gpu_id % 16}_g{gpu_id}', style='dashed')
        dot.edge(f'expert_l0_e{gpu_id % 16}_g{gpu_id}', f'aggregate_l0_g{gpu_id}')
        dot.edge(f'residual_l0_g{gpu_id % 8}', f'aggregate_l0_g{gpu_id}')
    
    # Connect layers
    for layer in range(1, 4):
        prev_layer = layer - 1
        for gpu_id in range(8):
            for prev_gpu in range(16):
                if prev_gpu % 8 == gpu_id:
                    dot.edge(f'aggregate_l{prev_layer}_g{prev_gpu}', f'mha_qkv_l{layer}_g{gpu_id}')
        
        for gpu_id in range(8):
            dot.edge(f'mha_qkv_l{layer}_g{gpu_id}', f'mha_attn_l{layer}_g{gpu_id}')
            dot.edge(f'mha_attn_l{layer}_g{gpu_id}', f'mha_out_l{layer}_g{gpu_id}')
            dot.edge(f'mha_out_l{layer}_g{gpu_id}', f'allreduce_l{layer}_g{gpu_id}')
            for prev_gpu in range(16):
                if prev_gpu % 8 == gpu_id:
                    dot.edge(f'aggregate_l{prev_layer}_g{prev_gpu}', f'residual_l{layer}_g{gpu_id}')
            dot.edge(f'allreduce_l{layer}_g{gpu_id}', f'residual_l{layer}_g{gpu_id}')
        
        for gpu_id in range(16):
            dot.edge(f'residual_l{layer}_g{gpu_id % 8}', f'gate_l{layer}_g{gpu_id}')
            dot.edge(f'gate_l{layer}_g{gpu_id}', f'expert_l{layer}_e{gpu_id % 16}_g{gpu_id}', style='dashed')
            dot.edge(f'expert_l{layer}_e{gpu_id % 16}_g{gpu_id}', f'aggregate_l{layer}_g{gpu_id}')
            dot.edge(f'residual_l{layer}_g{gpu_id % 8}', f'aggregate_l{layer}_g{gpu_id}')
    
    # Connect to output
    for gpu_id in range(16):
        dot.edge(f'aggregate_l3_g{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('./outputs/2025-10-13-20-35-53/baseline_moe_deployment', format='svg', cleanup=False)
    with open('./outputs/2025-10-13-20-35-53/baseline_moe_deployment.dot', 'w') as f:
        f.write(dag.source)