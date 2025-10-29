import graphviz

def create_llama_405b_baseline_tp():
    dot = graphviz.Digraph('Llama_405B_Baseline_TP')
    dot.attr(rankdir='TB', size='30,20')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Global attributes
    B = 1  # batch_size
    S = 1000000  # sequence length
    H = 16384  # hidden dimension
    Q = 128  # query heads
    K = 8  # kv heads
    Hsz = 128  # head size
    F = 65536  # ffn hidden
    
    # In baseline TP for Llama-405B, we can use up to 8 GPUs (constrained by KV heads=8)
    gpus = [f'gpu_{i}' for i in range(8)]
    
    # Input Layer
    dot.node('input', f'Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: CPU',
             shape='ellipse', fillcolor='lightgreen')
    
    # Split across GPUs
    dot.node('split_all_gpus', f'Split Input\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # QKV Projection (tensor parallel across 8 GPUs)
    for i, gpu in enumerate(gpus):
        start_col = i * (H // 8)
        end_col = (i + 1) * (H // 8) - 1
        query_start = i * (Q // 8)
        query_end = (i + 1) * (Q // 8) - 1
        kv_start = i * (K // 8)
        kv_end = (i + 1) * (K // 8) - 1
        
        dot.node(f'qkv_proj_{gpu}', f'QKV Projection\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q//8}, d_k={Hsz}]\\nDevice: {gpu}\\nColumns: {start_col}-{end_col}\\nQueries: {query_start}-{query_end}, KV: {kv_start}-{kv_end}',
                fillcolor='lightcoral')
    
    # KV Cache (duplicated across all GPUs - limitation)
    for gpu in gpus:
        dot.node(f'kv_cache_{gpu}', f'KV Cache\\nInput: [batch_size={B}, seq_len={S}, heads={K}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={S}, heads={K}, d_k={Hsz}]\\nDevice: {gpu}\\nDUPLICATED ACROSS GPUS',
                fillcolor='lightpink')
    
    # FlashAttention (local computation per GPU)
    for gpu in gpus:
        dot.node(f'flash_attn_{gpu}', f'FlashAttention\\nInput: [batch_size={B}, seq_len={S}, heads={Q//8}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q//8}, d_k={Hsz}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # All-Reduce for attention
    dot.node('attn_all_reduce', f'All-Reduce\\nAttention Output\\nInput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nOutput: [batch_size={B}, seq_len={S}, heads={Q}, d_k={Hsz}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Attention Output Projection (tensor parallel across 8 GPUs)
    for i, gpu in enumerate(gpus):
        start_row = i * (H // 8)
        end_row = (i + 1) * (H // 8) - 1
        dot.node(f'attn_out_{gpu}', f'Attention Output\\nInput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nDevice: {gpu}\\nRows: {start_row}-{end_row}',
                fillcolor='lightcoral')
    
    # All-Reduce for attention output
    dot.node('attn_out_all_reduce', f'All-Reduce\\nFinal Attention\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='parallelogram', fillcolor='yellow')
    
    # Residual Add for attention
    dot.node('residual_attn', f'Residual Add\\nAttention\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # RMSNorm after attention
    dot.node('rmsnorm1', f'RMSNorm\\nInput: [batch_size={B}, seq_len={S}, hidden={H}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H}]\\nDevice: all GPUs',
             fillcolor='lightgray')
    
    # FFN FC1 (column parallel across 8 GPUs)
    for i, gpu in enumerate(gpus):
        start_col = i * (F // 8)
        end_col = (i + 1) * (F // 8) - 1
        dot.node(f'ffn_fc1_{gpu}', f'FFN FC1\\nInput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nDevice: {gpu}\\nColumns: {start_col}-{end_col}',
                fillcolor='lightblue')
    
    # Activation
    for gpu in gpus:
        dot.node(f'ffn_act_{gpu}', f'SwiGLU Activation\\nInput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nOutput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nDevice: {gpu}',
                fillcolor='lightblue')
    
    # FFN FC2 (row parallel across 8 GPUs)
    for i, gpu in enumerate(gpus):
        start_row = i * (H // 8)
        end_row = (i + 1) * (H // 8) - 1
        dot.node(f'ffn_fc2_{gpu}', f'FFN FC2\\nInput: [batch_size={B}, seq_len={S}, ffn={F//8}]\\nOutput: [batch_size={B}, seq_len={S}, hidden={H//8}]\\nDevice: {gpu}\\nRows: {start_row}-{end_row}',
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
    
    # Attention all-reduce and output
    for gpu in gpus:
        dot.edge(f'flash_attn_{gpu}', 'attn_all_reduce')
        dot.edge('attn_all_reduce', f'attn_out_{gpu}')
        dot.edge(f'attn_out_{gpu}', 'attn_out_all_reduce')
    
    dot.edge('attn_out_all_reduce', 'residual_attn')
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
    dag = create_llama_405b_baseline_tp()
    dag.render('../outputs/2025-10-29-10-09-37/llama_405b_baseline_tp', format='dot', cleanup=False)
    dag.render('../outputs/2025-10-29-10-09-37/llama_405b_baseline_tp', format='svg', cleanup=False)