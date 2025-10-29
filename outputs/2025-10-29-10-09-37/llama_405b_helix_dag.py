import graphviz

def create_llama_405b_helix_dag():
    dot = graphviz.Digraph('Llama_405B_Helix_Parallelism')
    dot.attr(rankdir='TB', size='40,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Global attributes
    B = 1  # batch_size
    S = 1000000  # sequence length
    H = 16384  # hidden dimension
    Q = 128  # query heads
    K = 8  # kv heads (GQA)
    Hsz = 128  # head size
    F = 65536  # ffn hidden
    
    # Define GPU clusters - 16 GPUs total
    gpus = [f'gpu_{i}' for i in range(16)]
    
    # Define KVP groups (4 groups of 4 GPUs each)
    kvp_groups = {
        'group_0': gpus[0:4],
        'group_1': gpus[4:8],
        'group_2': gpus[8:12],
        'group_3': gpus[12:16]
    }
    
    # Input Layer
    dot.node('input', f'Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: CPU',
             shape='ellipse', fillcolor='lightgreen')
    
    # Split across all GPUs
    dot.node('split_all_gpus', f'Split Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # QKV Projection (head parallel within groups)
    for group_name, group_gpus in kvp_groups.items():
        for gpu in group_gpus:
            if group_name == 'group_0':
                query_range = "0-31"
                kv_range = "0-1"
            elif group_name == 'group_1':
                query_range = "32-63"
                kv_range = "2-3"
            elif group_name == 'group_2':
                query_range = "64-95"
                kv_range = "4-5"
            else:  # group_3
                query_range = "96-127"
                kv_range = "6-7"
                
            dot.node(f'qkv_proj_{gpu}', f'QKV Projection\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, heads=32, d_k={Hsz}]\\nDevice: {gpu}\\nQueries: {query_range}, KV: {kv_range}',
                    fillcolor='lightcoral')
    
    # KV Cache (sequence sharded by group)
    sequence_per_group = S // 4
    for group_idx, (group_name, group_gpus) in enumerate(kvp_groups.items()):
        start_seq = group_idx * sequence_per_group
        end_seq = (group_idx + 1) * sequence_per_group - 1
        
        for gpu in group_gpus:
            dot.node(f'kv_cache_{gpu}', f'KV Cache\\nInput: [batch_size={B}, seq_len={sequence_per_group//4}, heads={K//4}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={sequence_per_group//4}, heads={K//4}, d_k={Hsz}]\\nDevice: {gpu}\\nSequence: {start_seq}-{end_seq}',
                    fillcolor='lightpink')
    
    # FlashAttention (local computation per GPU)
    for gpu in gpus:
        dot.node(f'flash_attn_{gpu}', f'FlashAttention\\nInput: [batch_size={B}, seq_len={250000//4}, heads=32, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={250000//4}, heads=32, d_k={Hsz}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # Group-wise All-to-All Communication
    for group_name in kvp_groups:
        dot.node(f'all2all_{group_name}', f'All-to-All\\nGroup {group_name[-1]}\\nInput: [batch_size={B}, seq_len={250000}, heads=128, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={250000}, heads=128, d_k={Hsz}]\\nVolume: {B}×{H}×0.5 bytes\\nDevice: {group_name}',
                 shape='parallelogram', fillcolor='yellow')
    
    # Global All-to-All for cross-group communication
    dot.node('global_all2all', f'Global All-to-All\\nCross-Group Exchange\\nInput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nVolume: {B}×{H}×0.5 bytes\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Attention Output Projection (tensor parallel across 16 GPUs)
    for i, gpu in enumerate(gpus):
        start_row = i * (H // 16)
        end_row = (i + 1) * (H // 16) - 1
        dot.node(f'attn_out_{gpu}', f'Attention Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H//16}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H//16}]\\nDevice: {gpu}\\nRows: {start_row}-{end_row}',
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
    
    # Dense FFN (since Llama-405B is dense)
    # FC1 Column Parallel
    for i, gpu in enumerate(gpus):
        start_col = i * (F // 16)
        end_col = (i + 1) * (F // 16) - 1
        dot.node(f'ffn_fc1_{gpu}', f'FFN FC1\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//16}]\\nDevice: {gpu}\\nColumns: {start_col}-{end_col}',
                fillcolor='lightblue')
    
    # Activation
    for gpu in gpus:
        dot.node(f'ffn_act_{gpu}', f'SwiGLU Activation\\nInput: [batch_size={B}, seq_len={S}, ffn={F//16}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//16}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # FC2 Row Parallel
    for i, gpu in enumerate(gpus):
        start_row = i * (H // 16)
        end_row = (i + 1) * (H // 16) - 1
        dot.node(f'ffn_fc2_{gpu}', f'FFN FC2\\nInput: [batch_size={B}, seq_len={S}, ffn={F//16}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H//16}]\\nDevice: {gpu}\\nRows: {start_row}-{end_row}',
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
    dot.node('output', f'Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, vocab_size=128000]\\nDevice: CPU',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'split_all_gpus')
    
    # QKV projection connections
    for gpu in gpus:
        dot.edge('split_all_gpus', f'qkv_proj_{gpu}')
        dot.edge(f'qkv_proj_{gpu}', f'kv_cache_{gpu}')
        dot.edge(f'kv_cache_{gpu}', f'flash_attn_{gpu}')
    
    # Group communication
    for group_name, group_gpus in kvp_groups.items():
        for gpu in group_gpus:
            dot.edge(f'flash_attn_{gpu}', f'all2all_{group_name}')
        for gpu in group_gpus:
            dot.edge(f'all2all_{group_name}', 'global_all2all')
    
    # Attention output projection
    for gpu in gpus:
        dot.edge('global_all2all', f'attn_out_{gpu}')
        dot.edge(f'attn_out_{gpu}', 'attn_all_reduce')
    
    dot.edge('attn_all_reduce', 'residual_attn')
    dot.edge('split_all_gpus', 'residual_attn')  # residual connection
    dot.edge('residual_attn', 'rmsnorm1')
    
    # FFN connections
    for gpu in gpus:
        dot.edge('rmsnorm1', f'ffn_fc1_{gpu}')
        dot.edge(f'ffn_fc1_{gpu}', f'ffn_act_{gpu}')
        dot.edge(f'ffn_act_{gpu}', f'ffn_fc2_{gpu}')
        dot.edge(f'ffn_fc2_{gpu}', 'ffn_all_reduce')
    
    dot.edge('ffn_all_reduce', 'residual_ffn')
    dot.edge('rmsnorm1', 'residual_ffn')  # residual connection
    dot.edge('residual_ffn', 'rmsnorm2')
    dot.edge('rmsnorm2', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_llama_405b_helix_dag()
    dag.render('../outputs/2025-10-29-10-09-37/llama_405b_helix', format='dot', cleanup=False)
    dag.render('../outputs/2025-10-29-10-09-37/llama_405b_helix', format='svg', cleanup=False)