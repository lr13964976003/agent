import graphviz

def create_deepseek_r1_helix_dag():
    dot = graphviz.Digraph('DeepSeek_R1_Helix_Parallelism')
    dot.attr(rankdir='TB', size='30,20')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Global attributes
    B = 1  # batch_size (can be parameterized)
    S = 1000000  # sequence length
    H = 16384  # hidden dimension
    Q = 128  # query heads
    K = 1  # kv heads (MLA)
    Hsz = 128  # head size
    F = 65536  # ffn hidden
    
    # Define GPU clusters
    gpus = [f'gpu_{i}' for i in range(8)]
    
    # Input Layer
    dot.node('input', f'Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: CPU', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Split across GPUs
    dot.node('split_all_gpus', f'Split Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # QKV Projection (replicated across all 8 GPUs)
    for gpu in gpus:
        dot.node(f'qkv_proj_{gpu}', f'QKV Projection\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nDevice: {gpu}',
                fillcolor='lightcoral')
    
    # KV Cache (sequence sharded)
    shard_size = S // 8
    for i, gpu in enumerate(gpus):
        start_token = i * shard_size
        end_token = (i + 1) * shard_size - 1
        dot.node(f'kv_cache_{gpu}', f'KV Cache\\nInput: [batch_size={B}, seq_len={shard_size}, heads={K}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={shard_size}, heads={K}, d_k={Hsz}]\\nDevice: {gpu}\\nTokens: {start_token}-{end_token}',
                fillcolor='lightpink')
    
    # FlashAttention (local computation per GPU)
    for gpu in gpus:
        dot.node(f'flash_attn_{gpu}', f'FlashAttention\\nInput: [batch_size={B}, seq_len={shard_size}, heads={Q}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={shard_size}, heads={Q}, d_k={Hsz}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # All-to-All Communication (query head exchange)
    dot.node('all2all', f'All-to-All\\nExchange Query Heads\\nInput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nVolume: {B}×{Q}×{Hsz}×0.5 bytes\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Attention Output Projection (tensor parallel across 8 GPUs)
    for i, gpu in enumerate(gpus):
        start_row = i * (H // 8)
        end_row = (i + 1) * (H // 8) - 1
        dot.node(f'attn_out_{gpu}', f'Attention Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nDevice: {gpu}\\nRows: {start_row}-{end_row}',
                fillcolor='lightcoral')
    
    # All-Reduce for attention output
    dot.node('attn_all_reduce', f'All-Reduce\\nAttention Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Residual Add for attention
    dot.node('residual_attn', f'Residual Add\\nAttention\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # RMSNorm after attention
    dot.node('rmsnorm1', f'RMSNorm\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             fillcolor='lightgray')
    
    # Expert Routing
    for gpu in gpus:
        dot.node(f'expert_gate_{gpu}', f'Expert Gate\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: {gpu}\\nExperts: {i*32}-{(i+1)*32-1}',
                shape='diamond', fillcolor='orange', style='dashed')
    
    # Expert FC1 (column parallel)
    for gpu in gpus:
        dot.node(f'expert_fc1_{gpu}', f'Expert FC1\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # Expert activation
    for gpu in gpus:
        dot.node(f'expert_act_{gpu}', f'SiLU Activation\\nInput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # Expert FC2 (row parallel)
    for gpu in gpus:
        dot.node(f'expert_fc2_{gpu}', f'Expert FC2\\nInput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # All-Reduce for FFN
    dot.node('ffn_all_reduce', f'All-Reduce\\nFFN Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Residual Add for FFN
    dot.node('residual_ffn', f'Residual Add\\nFFN\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # RMSNorm after FFN
    dot.node('rmsnorm2', f'RMSNorm\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             fillcolor='lightgray')
    
    # Output Layer
    dot.node('output', f'Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, vocab_size=32000]\\nDevice: CPU',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'split_all_gpus')
    
    # QKV projection connections
    for gpu in gpus:
        dot.edge('split_all_gpus', f'qkv_proj_{gpu}')
        dot.edge(f'qkv_proj_{gpu}', f'kv_cache_{gpu}')
        dot.edge(f'kv_cache_{gpu}', f'flash_attn_{gpu}')
    
    # All-to-all communication
    for gpu in gpus:
        dot.edge(f'flash_attn_{gpu}', 'all2all')
    
    # Attention output projection
    for gpu in gpus:
        dot.edge('all2all', f'attn_out_{gpu}')
        dot.edge(f'attn_out_{gpu}', 'attn_all_reduce')
    
    dot.edge('attn_all_reduce', 'residual_attn')
    dot.edge('split_all_gpus', 'residual_attn')  # residual connection
    dot.edge('residual_attn', 'rmsnorm1')
    
    # FFN connections
    for gpu in gpus:
        dot.edge('rmsnorm1', f'expert_gate_{gpu}')
        dot.edge(f'expert_gate_{gpu}', f'expert_fc1_{gpu}')
        dot.edge(f'expert_fc1_{gpu}', f'expert_act_{gpu}')
        dot.edge(f'expert_act_{gpu}', f'expert_fc2_{gpu}')
        dot.edge(f'expert_fc2_{gpu}', 'ffn_all_reduce')
    
    dot.edge('ffn_all_reduce', 'residual_ffn')
    dot.edge('rmsnorm1', 'residual_ffn')  # residual connection
    dot.edge('residual_ffn', 'rmsnorm2')
    dot.edge('rmsnorm2', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_deepseek_r1_helix_dag()
    dag.render('../outputs/2025-10-29-10-09-37/deepseek_r1_helix', format='dot', cleanup=False)
    dag.render('../outputs/2025-10-29-10-09-37/deepseek_r1_helix', format='svg', cleanup=False)