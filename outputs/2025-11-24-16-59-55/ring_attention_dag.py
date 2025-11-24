import graphviz

# Create Ring Attention + Sequence Parallelism DAG
dot = graphviz.Digraph('ring_attention_sp', comment='Ring Attention + Sequence Parallelism')

# Set graph attributes
dot.attr(rankdir='TB', ranksep='2.0', nodesep='1.0')
dot.attr('node', shape='rectangle', style='filled')

# Input sequence partitioning
input_node = 'input_split'
dot.node(input_node, 'Input Sequence Split\\n(Sequence Parallel)', 
         shape='ellipse', 
         fillcolor='lightblue',
         xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096] per device\\n16 devices → 6250 tokens each')

# Ring topology communication
for device_id in range(16):
    device_cluster = f'device_{device_id}'
    
    with dot.subgraph(name=f'cluster_device_{device_id}') as device:
        device.attr(label=f'Device {device_id}\\nSequence Slice: tokens {device_id*6250}-{(device_id+1)*6250-1}', 
                   style='dashed', color='blue')
        
        # Layer processing for all 16 layers on each device
        for layer in range(16):
            layer_prefix = f'd{device_id}_l{layer}'
            
            # QKV Projection - Full model on each device
            qkv_node = f'{layer_prefix}_qkv'
            device.node(qkv_node, f'D{device_id} L{layer}\\nQKV Projection\\n(Local)', 
                       fillcolor='lightgreen',
                       xlabel=f'Input: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, heads=32, qkv_dim=128]\\nDevice: {device_id}')
            
            # Ring Attention Stages
            for stage in range(16):
                stage_node = f'{layer_prefix}_ring_stage_{stage}'
                
                # Determine which K,V block this stage processes
                kv_source = (device_id - stage) % 16
                
                device.node(stage_node, f'D{device_id} L{layer}\\nRing Stage {stage}\\nKV from D{kv_source}', 
                           fillcolor='yellow' if stage == 0 else 'lightyellow',
                           xlabel=f'Q: [batch_size=128, seq_len=6250, heads=32, qkv_dim=128]\\nK,V: [batch_size=128, seq_len=6250, heads=32, qkv_dim=128]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096]\\nDevice: {device_id}')
            
            # Attention Output - Local computation
            attn_out = f'{layer_prefix}_attn_out'
            device.node(attn_out, f'D{device_id} L{layer}\\nAttention Output\\n(Local)', 
                       fillcolor='lightgreen',
                       xlabel=f'Input: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096]\\nDevice: {device_id}')
            
            # Residual Connection
            residual = f'{layer_prefix}_residual'
            device.node(residual, f'D{device_id} L{layer}\\nResidual Add', 
                       shape='parallelogram', 
                       fillcolor='lightgray',
                       xlabel=f'Input1: [batch_size=128, seq_len=6250, d_model=4096]\\nInput2: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096]\\nDevice: {device_id}')
            
            # MLP Block
            mlp_gate = f'{layer_prefix}_mlp_gate'
            device.node(mlp_gate, f'D{device_id} L{layer}\\nMLP Gate\\n(Local)', 
                       fillcolor='lightcoral',
                       xlabel=f'Input: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, mlp_hidden=16384]\\nDevice: {device_id}')
            
            mlp_up = f'{layer_prefix}_mlp_up'
            device.node(mlp_up, f'D{device_id} L{layer}\\nMLP Up\\n(Local)', 
                       fillcolor='lightcoral',
                       xlabel=f'Input: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, mlp_hidden=16384]\\nDevice: {device_id}')
            
            mlp_down = f'{layer_prefix}_mlp_down'
            device.node(mlp_down, f'D{device_id} L{layer}\\nMLP Down\\n(Local)', 
                       fillcolor='lightcoral',
                       xlabel=f'Input: [batch_size=128, seq_len=6250, mlp_hidden=16384]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096]\\nDevice: {device_id}')
            
            mlp_residual = f'{layer_prefix}_mlp_residual'
            device.node(mlp_residual, f'D{device_id} L{layer}\\nMLP Residual Add', 
                       shape='parallelogram', 
                       fillcolor='lightgray',
                       xlabel=f'Input1: [batch_size=128, seq_len=6250, d_model=4096]\\nInput2: [batch_size=128, seq_len=6250, d_model=4096]\\nOutput: [batch_size=128, seq_len=6250, d_model=4096]\\nDevice: {device_id}')

# Ring communication nodes
for device_id in range(16):
    for layer in range(16):
        layer_prefix = f'd{device_id}_l{layer}'
        
        # KV communication nodes for ring topology
        for stage in range(16):
            if stage > 0:
                # Send K,V to next device
                send_node = f'{layer_prefix}_send_{stage}'
                recv_node = f'{layer_prefix}_recv_{stage}'
                
                dot.node(send_node, f'D{device_id} L{layer}\\nSend K,V\\nStage {stage}', 
                        shape='ellipse', 
                        fillcolor='orange',
                        xlabel=f'K,V: [batch_size=128, seq_len=6250, heads=32, qkv_dim=128]\\nTo: Device {(device_id+1)%16}')
                
                dot.node(recv_node, f'D{device_id} L{layer}\\nRecv K,V\\nStage {stage}', 
                        shape='ellipse', 
                        fillcolor='lightblue',
                        xlabel=f'K,V: [batch_size=128, seq_len=6250, heads=32, qkv_dim=128]\\nFrom: Device {(device_id-1)%16}')

# Output aggregation
output_agg = 'output_aggregate'
dot.node(output_agg, 'Output Aggregation\\n(Sequence Parallel)', 
         shape='ellipse', 
         fillcolor='lightblue',
         xlabel='Input: [batch_size=128, seq_len=6250, d_model=4096] per device\\nOutput: [batch_size=128, seq_len=100000, d_model=4096]\\n16 devices → concatenate')

output_final = 'output_final'
dot.node(output_final, 'Final Output', 
         shape='ellipse', 
         fillcolor='lightgreen',
         xlabel='Input: [batch_size=128, seq_len=100000, d_model=4096]\\nOutput: [batch_size=128, seq_len=100000, vocab_size]')

# Connections for ring attention DAG
for device_id in range(16):
    # Input to QKV
    dot.edge(input_node, f'd{device_id}_l0_qkv')
    
    for layer in range(16):
        layer_prefix = f'd{device_id}_l{layer}'
        
        # Layer connections
        dot.edge(f'{layer_prefix}_qkv', f'{layer_prefix}_ring_stage_0')
        
        # Ring stages
        for stage in range(16):
            if stage < 15:
                dot.edge(f'{layer_prefix}_ring_stage_{stage}', f'{layer_prefix}_ring_stage_{stage+1}')
            else:
                dot.edge(f'{layer_prefix}_ring_stage_{stage}', f'{layer_prefix}_attn_out')
        
        # Communication between stages
        for stage in range(1, 16):
            # Receive K,V from previous device
            recv_from = (device_id - 1) % 16
            send_to = (device_id + 1) % 16
            
            # Communication edges (dashed for KV transfer)
            kv_recv_node = f'd{device_id}_l{layer}_recv_{stage}'
            kv_send_node = f'd{device_id}_l{layer}_send_{stage}'
            
            # Connect receive to attention stage
            if stage > 0:
                dot.edge(f'd{(device_id-1)%16}_l{layer}_send_{stage}', kv_recv_node, style='dashed')
                dot.edge(kv_recv_node, f'{layer_prefix}_ring_stage_{stage}')
            
            # Connect attention computation to send
            if stage < 15:
                dot.edge(f'{layer_prefix}_ring_stage_{stage}', f'd{device_id}_l{layer}_send_{stage+1}')
        
        # Continue layer processing
        dot.edge(f'{layer_prefix}_attn_out', f'{layer_prefix}_residual')
        dot.edge(f'{layer_prefix}_residual', f'{layer_prefix}_mlp_gate')
        dot.edge(f'{layer_prefix}_mlp_gate', f'{layer_prefix}_mlp_up')
        dot.edge(f'{layer_prefix}_mlp_up', f'{layer_prefix}_mlp_down')
        dot.edge(f'{layer_prefix}_mlp_down', f'{layer_prefix}_mlp_residual')
        
        # Next layer or output
        if layer < 15:
            dot.edge(f'{layer_prefix}_mlp_residual', f'd{device_id}_l{layer+1}_qkv')
        else:
            dot.edge(f'{layer_prefix}_mlp_residual', output_agg)

# Final aggregation
dot.edge(output_agg, output_final)

# Save the DAG
dot.render('../outputs/2025-11-24-16-59-55/ring_attention_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-11-24-16-59-55/ring_attention_dag', format='svg', cleanup=False)

print("Ring Attention DAG generated successfully")
print("Files saved:")
print("- ../outputs/2025-11-24-16-59-55/ring_attention_dag.dot")
print("- ../outputs/2025-11-24-16-59-55/ring_attention_dag.svg")