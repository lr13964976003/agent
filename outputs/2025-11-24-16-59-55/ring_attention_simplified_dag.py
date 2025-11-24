import graphviz

# Create simplified Ring Attention + Sequence Parallelism DAG
dot = graphviz.Digraph('ring_attention_sp_simplified', comment='Ring Attention + Sequence Parallelism (Simplified)')

# Set graph attributes
dot.attr(rankdir='LR', ranksep='1.0', nodesep='1.0')
dot.attr('node', shape='rectangle', style='filled')

# Input sequence split across 16 devices
input_node = 'input_split'
dot.node(input_node, 'Input Sequence\\nSplit (100k → 6.25k)', 
         shape='ellipse', 
         fillcolor='lightblue',
         xlabel='Input: [batch=128, seq=100k, d_model=4096]\\nOutput: [batch=128, seq=6.25k, d_model=4096] per device')

# Show 3 representative devices (0, 1, 2) to demonstrate the pattern
for device_id in range(3):
    device_cluster = f'device_{device_id}'
    
    with dot.subgraph(name=f'cluster_device_{device_id}') as device:
        device.attr(label=f'Device {device_id}\\nTokens: {device_id*6250}-{(device_id+1)*6250-1}', 
                   style='dashed', color='blue')
        
        # Layer processing (show 3 representative layers)
        for layer in [0, 8, 15]:
            layer_prefix = f'd{device_id}_l{layer}'
            
            # QKV Projection
            qkv_node = f'{layer_prefix}_qkv'
            device.node(qkv_node, f'D{device_id}\\nL{layer}\\nQKV Proj', 
                       fillcolor='lightgreen',
                       xlabel='Input: [128, 6250, 4096]\\nOutput: [128, 6250, 32*128]')
            
            # Ring Attention Stages (show 3 stages)
            for stage in [0, 8, 15]:
                stage_node = f'{layer_prefix}_ring_stage_{stage}'
                kv_source = (device_id - stage) % 16
                device.node(stage_node, f'D{device_id}\\nL{layer}\\nStage {stage}\\nKV from D{kv_source}', 
                           fillcolor='yellow',
                           xlabel='Q: [128, 6250, 32, 128]\\nK,V: [128, 6250, 32, 128]\\nOutput: [128, 6250, 4096]')
            
            # MLP Block
            mlp_node = f'{layer_prefix}_mlp'
            device.node(mlp_node, f'D{device_id}\\nL{layer}\\nMLP Block', 
                       fillcolor='lightcoral',
                       xlabel='Input: [128, 6250, 4096]\\nOutput: [128, 6250, 4096]')

# Ring communication pattern
comm_node = 'ring_comm'
dot.node(comm_node, 'Ring Communication\\nPattern', 
         shape='ellipse', 
         fillcolor='orange',
         xlabel='P2P Send/Recv\\nKV blocks\\nEach → next device')

# Output aggregation
output_node = 'output_agg'
dot.node(output_node, 'Output Aggregation', 
         shape='ellipse', 
         fillcolor='lightgreen',
         xlabel='Concatenate 16×[128, 6250, 4096]\\n→ [128, 100k, 4096]')

# Connections showing the pattern
for device_id in range(3):
    # Input to first layer
    dot.edge(input_node, f'd{device_id}_l0_qkv')
    
    # Layer 0 processing
    dot.edge(f'd{device_id}_l0_qkv', f'd{device_id}_l0_ring_stage_0')
    dot.edge(f'd{device_id}_l0_ring_stage_0', f'd{device_id}_l0_mlp')
    
    # Ring communication (show 3 devices communicating)
    if device_id < 2:
        dot.edge(f'd{device_id}_l0_ring_stage_0', f'd{(device_id+1)%16}_l0_ring_stage_1', 
                style='dashed', label='KV transfer')
    
    # Connect to next layer
    dot.edge(f'd{device_id}_l0_mlp', f'd{device_id}_l8_qkv')
    dot.edge(f'd{device_id}_l8_qkv', f'd{device_id}_l8_ring_stage_8')
    dot.edge(f'd{device_id}_l8_ring_stage_8', f'd{device_id}_l8_mlp')
    dot.edge(f'd{device_id}_l8_mlp', f'd{device_id}_l15_qkv')
    dot.edge(f'd{device_id}_l15_qkv', f'd{device_id}_l15_ring_stage_15')
    dot.edge(f'd{device_id}_l15_ring_stage_15', f'd{device_id}_l15_mlp')
    
    # Connect to output
    dot.edge(f'd{device_id}_l15_mlp', output_node)

# Add device labels for clarity
dot.edge('input_split', 'cluster_device_0', style='invis')
dot.edge('cluster_device_0', 'cluster_device_1', style='invis')
dot.edge('cluster_device_1', 'cluster_device_2', style='invis')
dot.edge('cluster_device_2', 'output_agg', style='invis')

# Save the simplified DAG
dot.render('../outputs/2025-11-24-16-59-55/ring_attention_simplified_dag', format='dot', cleanup=False)
dot.render('../outputs/2025-11-24-16-59-55/ring_attention_simplified_dag', format='svg', cleanup=False)

print("Simplified Ring Attention DAG generated successfully")
print("Files saved:")
print("- ../outputs/2025-11-24-16-59-55/ring_attention_simplified_dag.dot")
print("- ../outputs/2025-11-24-16-59-55/ring_attention_simplified_dag.svg")