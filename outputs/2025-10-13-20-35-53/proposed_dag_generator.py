import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_deployment', format='svg')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node - distributed to all GPUs
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Process each layer
    for layer in range(4):
        layer_color = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][layer % 4]
        
        # Multi-Head Attention - no tensor parallelism in proposed method
        dot.attr('node', shape='rectangle', fillcolor='lightcoral')
        for gpu_id in range(16):
            dot.node(f'mha_qkv_l{layer}_g{gpu_id}',
                    f'MHA QKV Linear Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]',
                    fillcolor='lightcoral')
            
            dot.node(f'mha_attn_l{layer}_g{gpu_id}',
                    f'MHA Attention Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]',
                    fillcolor='lightcoral')
            
            dot.node(f'mha_out_l{layer}_g{gpu_id}',
                    f'MHA Output Linear Layer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightcoral')
        
        # Residual connections
        dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
        for gpu_id in range(16):
            dot.node(f'residual_mha_l{layer}_g{gpu_id}',
                    f'Residual Add\\nMHA Layer {layer}\\nGPU {gpu_id}\\nInput1: [batch_size=1024, seq_len=10000, dim=8192]\\nInput2: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
        
        # Gating networks - one per GPU for routing
        dot.attr('node', shape='parallelogram', fillcolor='lightgreen')
        for gpu_id in range(16):
            dot.node(f'gate_l{layer}_g{gpu_id}',
                    f'Gating Network\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, num_experts=16]\\nType: Top-K routing',
                    fillcolor='lightgreen')
        
        # Token sharding and routing
        dot.attr('node', shape='parallelogram', fillcolor='lightblue')
        for gpu_id in range(16):
            dot.node(f'shard_l{layer}_g{gpu_id}',
                    f'Token Sharding\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=variable, seq_len=variable, dim=8192]\\nType: Async routing',
                    fillcolor='lightblue')
        
        # Expert computation - one expert per GPU
        dot.attr('node', shape='rectangle', fillcolor='lightgreen')
        for gpu_id in range(16):
            dot.node(f'expert_l{layer}_e{gpu_id}_g{gpu_id}',
                    f'Expert {gpu_id}\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=variable, seq_len=variable, dim=8192]\\nType: MLP [8192→32768→8192]',
                    fillcolor='lightgreen')
        
        # Expert aggregation and all-gather
        dot.attr('node', shape='parallelogram', fillcolor='lightyellow')
        for gpu_id in range(16):
            dot.node(f'gather_l{layer}_g{gpu_id}',
                    f'Expert Output Gather\\nLayer {layer}\\nGPU {gpu_id}\\nInput: [batch_size=variable, seq_len=variable, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]\\nType: All-gather',
                    fillcolor='lightyellow')
            
            dot.node(f'weighted_sum_l{layer}_g{gpu_id}',
                    f'Weighted Sum\\nLayer {layer}\\nGPU {gpu_id}\\nInput1: [batch_size=1024, seq_len=10000, dim=8192]\\nInput2: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
        
        # Final residual connection
        for gpu_id in range(16):
            dot.node(f'residual_final_l{layer}_g{gpu_id}',
                    f'Final Residual Add\\nLayer {layer}\\nGPU {gpu_id}\\nInput1: [batch_size=1024, seq_len=10000, dim=8192]\\nInput2: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]',
                    fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nOutput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create edges for each layer
    for layer in range(4):
        # Connect input to first layer
        if layer == 0:
            for gpu_id in range(16):
                dot.edge('input', f'mha_qkv_l{layer}_g{gpu_id}')
        else:
            for gpu_id in range(16):
                dot.edge(f'residual_final_l{layer-1}_g{gpu_id}', f'mha_qkv_l{layer}_g{gpu_id}')
        
        # MHA computation
        for gpu_id in range(16):
            dot.edge(f'mha_qkv_l{layer}_g{gpu_id}', f'mha_attn_l{layer}_g{gpu_id}')
            dot.edge(f'mha_attn_l{layer}_g{gpu_id}', f'mha_out_l{layer}_g{gpu_id}')
        
        # Residual connection for MHA
        for gpu_id in range(16):
            if layer == 0:
                dot.edge('input', f'residual_mha_l{layer}_g{gpu_id}')
            else:
                dot.edge(f'residual_final_l{layer-1}_g{gpu_id}', f'residual_mha_l{layer}_g{gpu_id}')
            dot.edge(f'mha_out_l{layer}_g{gpu_id}', f'residual_mha_l{layer}_g{gpu_id}')
        
        # Gating and routing
        for gpu_id in range(16):
            dot.edge(f'residual_mha_l{layer}_g{gpu_id}', f'gate_l{layer}_g{gpu_id}')
            dot.edge(f'residual_mha_l{layer}_g{gpu_id}', f'shard_l{layer}_g{gpu_id}')
            dot.edge(f'gate_l{layer}_g{gpu_id}', f'shard_l{layer}_g{gpu_id}', style='dashed')
        
        # Cross-GPU routing
        for src_gpu in range(16):
            for dst_gpu in range(16):
                if src_gpu != dst_gpu:
                    dot.edge(f'shard_l{layer}_g{src_gpu}', f'expert_l{layer}_e{dst_gpu}_g{dst_gpu}',
                            label=f'async routing', style='dotted')
        
        # Local routing
        for gpu_id in range(16):
            dot.edge(f'shard_l{layer}_g{gpu_id}', f'expert_l{layer}_e{gpu_id}_g{gpu_id}')
        
        # Expert output gathering
        for src_gpu in range(16):
            for dst_gpu in range(16):
                dot.edge(f'expert_l{layer}_e{src_gpu}_g{src_gpu}', f'gather_l{layer}_g{dst_gpu}',
                        label=f'async gather', style='dotted')
        
        # Weighted sum and final residual
        for gpu_id in range(16):
            dot.edge(f'gather_l{layer}_g{gpu_id}', f'weighted_sum_l{layer}_g{gpu_id}')
            dot.edge(f'gate_l{layer}_g{gpu_id}', f'weighted_sum_l{layer}_g{gpu_id}', style='dashed')
            dot.edge(f'residual_mha_l{layer}_g{gpu_id}', f'residual_final_l{layer}_g{gpu_id}')
            dot.edge(f'weighted_sum_l{layer}_g{gpu_id}', f'residual_final_l{layer}_g{gpu_id}')
    
    # Connect final layer to output
    for gpu_id in range(16):
        dot.edge(f'residual_final_l3_g{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('./outputs/2025-10-13-20-35-53/proposed_moe_deployment', format='svg', cleanup=False)
    with open('./outputs/2025-10-13-20-35-53/proposed_moe_deployment.dot', 'w') as f:
        f.write(dag.source)