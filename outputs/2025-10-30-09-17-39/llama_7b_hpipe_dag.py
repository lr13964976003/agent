#!/usr/bin/env python3
"""
HPipe LLaMA-7B Deployment DAG Generator
Comprehensive DAG showing token-level pipeline parallelism across 6 heterogeneous devices
"""

import graphviz
from typing import Dict, List, Tuple

def create_llama_7b_hpipe_dag():
    """Create complete LLaMA-7B HPipe DAG with token-level pipeline parallelism"""
    
    # Initialize DAG with strict hierarchical layout
    dot = graphviz.Digraph('LLaMA_7B_HPipe', 
                           comment='LLaMA-7B HPipe Token-Level Pipeline Deployment',
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
        'P100_2': {'id': 'P100_2', 'host': 'host_1', 'layers': [5, 6, 7, 8, 9], 'color': 'lightblue'},
        'P100_3': {'id': 'P100_3', 'host': 'host_1', 'layers': [10, 11, 12, 13, 14], 'color': 'lightblue'},
        'P100_4': {'id': 'P100_4', 'host': 'host_1', 'layers': [15, 16, 17, 18], 'color': 'lightblue'},
        'RTX3090_1': {'id': 'RTX3090_1', 'host': 'host_2', 'layers': [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 'color': 'lightgreen'},
    }
    
    # Model dimensions
    model_dims = {
        'batch_size': 6,
        'seq_len': 2048,
        'hidden_size': 4096,
        'num_heads': 32,
        'head_dim': 128,  # 4096/32
        'vocab_size': 32000,
        'ffn_hidden_size': 11008
    }
    
    # Token slice configuration for pipeline
    token_slices = [
        ('slice_1', 256), ('slice_2', 248), ('slice_3', 240), ('slice_4', 232),
        ('slice_5', 224), ('slice_6', 216), ('slice_7', 208), ('slice_8', 200),
        ('slice_9', 192), ('slice_10', 184), ('slice_11', 176), ('slice_12', 168),
        ('slice_13', 160), ('slice_14', 152), ('slice_15', 144), ('slice_16', 128)
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
                       f'QKV Projection\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, heads=32, head_dim=128]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Calculation
                c.node(attn_calc, 
                       f'Multi-Head Attention\\nInput: [batch=6, seq=2048, heads=32, head_dim=128]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Output Projection
                c.node(attn_out, 
                       f'Attention Output\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Attention Residual
                c.node(attn_res, 
                       f'Attention+Residual\\nInput1: [batch=6, seq=2048, hidden=4096]\\nInput2: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # FFN components
                c.node(ffn_up, 
                       f'FFN Up\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, ffn=11008]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_gate, 
                       f'FFN Gate\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, ffn=11008]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_down, 
                       f'FFN Down\\nInput: [batch=6, seq=2048, ffn=11008]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='rectangle', style='filled', fillcolor='lightcyan')
                
                c.node(ffn_res, 
                       f'FFN+Residual\\nInput1: [batch=6, seq=2048, hidden=4096]\\nInput2: [batch=6, seq=2048, hidden=4098]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # Layer normalization nodes (within each layer)
                norm1 = f'{device_name}_{layer_id}_norm1'
                norm2 = f'{device_name}_{layer_id}_norm2'
                
                c.node(norm1, f'LayerNorm\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='ellipse', style='filled', fillcolor='lightgray')
                
                c.node(norm2, f'LayerNorm\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: {device_name}',
                       shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Token slice nodes for pipeline
    for slice_name, slice_len in token_slices:
        slice_node = f'{slice_name}_input'
        dot.node(slice_node, 
                 f'{slice_name}\\nTokens: {slice_len}\\nShape: [batch=6, seq={slice_len}, hidden=4096]',
                 shape='octagon', style='filled', fillcolor='lightpink')
    
    # Input embedding
    dot.node('input_embedding', 
             f'Input Embedding\\nInput: [batch=6, seq=2048, vocab=32000]\\nOutput: [batch=6, seq=2048, hidden=4096]\\nDevice: All GPUs',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Output projection
    dot.node('output_projection', 
             f'Output Projection\\nInput: [batch=6, seq=2048, hidden=4096]\\nOutput: [batch=6, seq=2048, vocab=32000]\\nDevice: RTX3090_1',
             shape='rectangle', style='filled', fillcolor='lightgreen')
    
    # Final output
    dot.node('final_output', 
             f'Final Output\\nShape: [batch=6, seq=2048, vocab=32000]',
             shape='doubleoctagon', style='filled', fillcolor='lightblue')
    
    # Communication nodes - inter-device transfers
    device_list = ['P100_1', 'P100_2', 'P100_3', 'P100_4', 'RTX3090_1']
    for i, (from_device, to_device) in enumerate(zip(device_list, device_list[1:])):
        comm_node = f'comm_{from_device}_to_{to_device}'
        dot.node(comm_node, 
                 f'Communication\\n{from_device} → {to_device}\\n[batch=6, seq=2048, hidden=4096]\\nBandwidth: 1Gbps',
                 shape='ellipse', style='dashed,filled', fillcolor='lightyellow')
    
    # KV Cache nodes for token reuse
    kv_cache_nodes = {}
    for device_name in devices.keys():
        for slice_name, _ in token_slices:
            kv_node = f'kv_cache_{device_name}_{slice_name}'
            dot.node(kv_node, 
                     f'KV Cache\\n{device_name}\\n{slice_name}\\n[batch=6, heads=32, seq=?, head_dim=128]',
                     shape='cylinder', style='filled', fillcolor='lightgray')
            kv_cache_nodes[(device_name, slice_name)] = kv_node
    
    # Input connections
    dot.edge('input_embedding', 'P100_1_layer_1_qkv_proj', 
             label='[batch=6, seq=2048, hidden=4096]')
    
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
            dot.edge(f'{device_name}_{layer_id}_ffn_res', f'{device_name}_layer_{layer_idx+1}_norm1' 
                     if layer_idx < max(device_info['layers']) else 'next_device')
    
    # Inter-device connections with communication nodes
    device_layers = [
        ('P100_1', range(1, 5)),  # layers 1-4
        ('P100_2', range(5, 10)),  # layers 5-9
        ('P100_3', range(10, 15)),  # layers 10-14
        ('P100_4', range(15, 19)),  # layers 15-18
        ('RTX3090_1', range(19, 33))  # layers 19-32
    ]
    
    for i, ((from_device, _), (to_device, _)) in enumerate(zip(device_layers, device_layers[1:])):
        last_layer_from = max(device_info['layers']) if from_device == device_layers[i][0] else None
        first_layer_to = min([info['layers'][0] for info in devices.values() if info['id'] == to_device])
        
        dot.edge(f'{from_device}_layer_{last_layer_from}_ffn_res', f'comm_{from_device}_to_{to_device}')
        dot.edge(f'comm_{from_device}_to_{to_device}', f'{to_device}_layer_{first_layer_to}_norm1')
    
    # Final output connections
    dot.edge('RTX3090_1_layer_32_ffn_res', 'output_projection')
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
    dag = create_llama_7b_hpipe_dag()
    dag.render('../outputs/2025-10-30-09-17-39/llama_7b_hpipe', format='svg', cleanup=False)
    dag.save('../outputs/2025-10-30-09-17-39/llama_7b_hpipe.dot')
    print("LLaMA-7B HPipe DAG generated successfully")