#!/usr/bin/env python3
"""
HPipe GPT3-2B Deployment DAG Generator
Comprehensive DAG showing token-level pipeline parallelism across 6 heterogeneous devices
"""

import graphviz
from typing import Dict, List, Tuple

def create_gpt3_2b_hpipe_dag():
    """Create complete GPT3-2B HPipe DAG with token-level pipeline parallelism"""
    
    # Initialize DAG with strict hierarchical layout
    dot = graphviz.Digraph('GPT3_2B_HPipe', 
                           comment='GPT3-2B HPipe Token-Level Pipeline Deployment',
                           format='svg',
                           graph_attr={
                               'rankdir': 'TB',
                               'compound': 'true',
                               'ranksep': '1.5',
                               'nodesep': '0.2',
                               'concentrate': 'true'
                           })
    
    # Define all devices as subgraph clusters
    devices = {
        'P100_1': {'id': 'P100_1', 'host': 'host_1', 'layers': [1, 2, 3, 4], 'color': 'lightblue'},
        'P100_2': {'id': 'P100_2', 'host': 'host_1', 'layers': [5, 6, 7, 8], 'color': 'lightblue'},
        'P100_3': {'id': 'P100_3', 'host': 'host_1', 'layers': [9, 10, 11, 12], 'color': 'lightblue'},
        'P100_4': {'id': 'P100_4', 'host': 'host_1', 'layers': [13, 14, 15, 16], 'color': 'lightblue'},
        'RTX3090_1': {'id': 'RTX3090_1', 'host': 'host_2', 'layers': [17, 18, 19, 20], 'color': 'lightgreen'},
        'RTX3090_2': {'id': 'RTX3090_2', 'host': 'host_2', 'layers': [21, 22, 23, 24], 'color': 'lightgreen'}
    }
    
    # Model dimensions
    model_dims = {
        'batch_size': 12,
        'seq_len': 2048,
        'hidden_size': 2048,
        'num_heads': 16,
        'head_dim': 128,  # 2048/16
        'vocab_size': 50257,
        'ffn_hidden_size': 8192
    }
    
    # Token slice configuration for pipeline
    token_slices = [
        ('slice_1', 256), ('slice_2', 252), ('slice_3', 248), ('slice_4', 244),
        ('slice_5', 240), ('slice_6', 236), ('slice_7', 232), ('slice_8', 228),
        ('slice_9', 224), ('slice_10', 220), ('slice_11', 216), ('slice_12', 212),
        ('slice_13', 208), ('slice_14', 204), ('slice_15', 200), ('slice_16', 196),
        ('slice_17', 192)
    ]
    
    # Create device clusters
    for device_name, device_info in devices.items():
        with dot.subgraph(name=f'cluster_{device_name}') as c:
            c.attr(label=f'{device_name}\\n{device_info["host"]}', 
                   style='rounded,filled', 
                   fillcolor=device_info['color'],
                   color='black')
            
            # Create nodes for each layer on this device
            for layer_idx in device_info['layers']:
                layer_id = f'layer_{layer_idx}'
                
                # Multi-Head Attention components
                qkv_proj = f'{device_name}_{layer_id}_qkv_proj'
                attn_calc = f'{device_name}_{layer_id}_attn_calc'
                attn_out = f'{device_name}_{layer_id}_attn_out'
                attn_res = f'{device_name}_{layer_id}_attn_res'
                
                # FFN components
                ffn_up = f'{device_name}_{layer_id}_ffn_up'
                ffn_gate = f'{device_name}_{layer_id}_ffn_gate'
                ffn_down = f'{device_name}_{layer_id}_ffn_down'
                ffn_res = f'{device_name}_{layer_id}_ffn_res'
                
                # QKV Projection
                c.node(qkv_proj, 
                       f'QKV Projection\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, heads=16, head_dim=128]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Calculation
                c.node(attn_calc, 
                       f'Multi-Head Attention\\nInput: [batch=12, seq=2048, heads=16, head_dim=128]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Output Projection
                c.node(attn_out, 
                       f'Attention Output\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Residual
                c.node(attn_res, 
                       f'Attention+Residual\\nInput1: [batch=12, seq=2048, hidden=2048]\\nInput2: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # FFN components
                c.node(ffn_up, 
                       f'FFN Up\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, ffn=8192]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_gate, 
                       f'FFN Gate\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, ffn=8192]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_down, 
                       f'FFN Down\\nInput: [batch=12, seq=2048, ffn=8192]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_res, 
                       f'FFN+Residual\\nInput1: [batch=12, seq=2048, hidden=2048]\\nInput2: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # Layer normalization nodes
                norm1 = f'{device_name}_{layer_id}_norm1'
                norm2 = f'{device_name}_{layer_id}_norm2'
                
                c.node(norm1, f'LayerNorm\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='ellipse', style='filled', fillcolor='lightgray')
                
                c.node(norm2, f'LayerNorm\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: {device_name}',
                       shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Token slice nodes for pipeline
    for slice_name, slice_len in token_slices:
        slice_node = f'{slice_name}_input'
        dot.node(slice_node, 
                 f'{slice_name}\\nTokens: {slice_len}\\nShape: [batch=12, seq={slice_len}, hidden=2048]',
                 shape='octagon', style='filled', fillcolor='lightpink')
    
    # Input embedding
    dot.node('input_embedding', 
             f'Input Embedding\\nInput: [batch=12, seq=2048, vocab=50257]\\nOutput: [batch=12, seq=2048, hidden=2048]\\nDevice: All GPUs',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Output projection
    dot.node('output_projection', 
             f'Output Projection\\nInput: [batch=12, seq=2048, hidden=2048]\\nOutput: [batch=12, seq=2048, vocab=50257]\\nDevice: RTX3090_2',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Final output
    dot.node('final_output', 
             f'Final Output\\nShape: [batch=12, seq=2048, vocab=50257]',
             shape='doubleoctagon', style='filled', fillcolor='lightblue')
    
    # Communication nodes - inter-device transfers
    device_list = ['P100_1', 'P100_2', 'P100_3', 'P100_4', 'RTX3090_1', 'RTX3090_2']
    for i, (from_device, to_device) in enumerate(zip(device_list, device_list[1:])):
        comm_node = f'comm_{from_device}_to_{to_device}'
        host_from = devices[from_device]['host']
        host_to = devices[to_device]['host']
        bandwidth = '15.75Gbps' if host_from == host_to else '1Gbps'
        
        dot.node(comm_node, 
                 f'Communication\\n{from_device} → {to_device}\\n[batch=12, seq=2048, hidden=2048]\\nBandwidth: {bandwidth}',
                 shape='ellipse', style='dashed,filled', fillcolor='lightyellow')
    
    # KV Cache nodes for token reuse
    kv_cache_nodes = {}
    for device_name in devices.keys():
        for slice_name, _ in token_slices:
            kv_node = f'kv_cache_{device_name}_{slice_name}'
            dot.node(kv_node, 
                     f'KV Cache\\n{device_name}\\n{slice_name}\\n[batch=12, heads=16, seq=?, head_dim=128]',
                     shape='cylinder', style='filled', fillcolor='lightgray')
            kv_cache_nodes[(device_name, slice_name)] = kv_node
    
    # Input connections
    dot.edge('input_embedding', 'P100_1_layer_1_qkv_proj', 
             label='[batch=12, seq=2048, hidden=2048]')
    
    # Layer connections within each device
    for device_name, device_info in devices.items():
        for layer_idx in device_info['layers']:
            layer_id = f'layer_{layer_idx}'
            
            # Attention path
            dot.edge(f'{device_name}_{layer_id}_qkv_proj', f'{device_name}_{layer_id}_attn_calc')
            dot.edge(f'{device_name}_{layer_id}_attn_calc', f'{device_name}_{layer_id}_attn_out')
            dot.edge(f'{device_name}_{layer_id}_attn_out', f'{device_name}_{layer_id}_attn_res')
            
            # FFN path
            dot.edge(f'{device_name}_{layer_id}_ffn_up', f'{device_name}_{layer_id}_ffn_gate')
            dot.edge(f'{device_name}_{layer_id}_ffn_gate', f'{device_name}_{layer_id}_ffn_down')
            dot.edge(f'{device_name}_{layer_id}_ffn_down', f'{device_name}_{layer_id}_ffn_res')
            
            # Layer norm connections
            dot.edge(f'{device_name}_{layer_id}_norm1', f'{device_name}_{layer_id}_qkv_proj')
            dot.edge(f'{device_name}_{layer_id}_norm2', f'{device_name}_{layer_id}_ffn_up')
            
            # Residual connections
            dot.edge(f'{device_name}_{layer_id}_attn_res', f'{device_name}_{layer_id}_norm2')
            
    # Inter-device connections
    device_transfers = [
        ('P100_1', 'P100_2', 4),      # P100_1 (layers 1-4) -> P100_2 (layers 5-8)
        ('P100_2', 'P100_3', 8),      # P100_2 (layers 5-8) -> P100_3 (layers 9-12)
        ('P100_3', 'P100_4', 12),     # P100_3 (layers 9-12) -> P100_4 (layers 13-16)
        ('P100_4', 'RTX3090_1', 16),  # P100_4 (layers 13-16) -> RTX3090_1 (layers 17-20)
        ('RTX3090_1', 'RTX3090_2', 20) # RTX3090_1 (layers 17-20) -> RTX3090_2 (layers 21-24)
    ]
    
    for from_device, to_device, last_layer in device_transfers:
        comm_node = f'comm_{from_device}_to_{to_device}'
        next_layer = last_layer + 1
        dot.edge(f'{from_device}_layer_{last_layer}_ffn_res', comm_node)
        dot.edge(comm_node, f'{to_device}_layer_{next_layer}_norm1')
    
    # Final output connections
    dot.edge('RTX3090_2_layer_24_ffn_res', 'output_projection')
    dot.edge('output_projection', 'final_output')
    
    # Add constraint edges for hierarchical layout
    for device_name in devices.keys():
        with dot.subgraph() as s:
            s.attr(rank='same')
            layers = devices[device_name]['layers']
            for layer_idx in layers:
                layer_id = f'layer_{layer_idx}'
                s.node(f'{device_name}_{layer_id}_norm1')
                s.node(f'{device_name}_{layer_id}_norm2')
    
    return dot

if __name__ == "__main__":
    dag = create_gpt3_2b_hpipe_dag()
    dag.render('../outputs/2025-10-30-09-17-39/gpt3_2b_hpipe', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/gpt3_2b_hpipe.dot')
    print("GPT3-2B HPipe DAG generated successfully")