#!/usr/bin/env python3

import graphviz
import json
import os

def create_baseline_dag():
    """
    Create baseline DAG with tensor parallelism (8-way) and pipeline parallelism (2-way)
    16 GPUs total: 8 GPUs per pipeline stage
    """
    dot = graphviz.Digraph('baseline_transformer', 
                          comment='Dense Transformer - Baseline (TP=8, PP=2)')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Color schemes for different pipeline stages
    stage0_color = 'lightblue'
    stage1_color = 'lightgreen'
    
    # Input node
    dot.node('input', 'Input\n[batch_size=128, seq_len=100000, d_model=4096]', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Pipeline Stage 0 - Devices 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0\n(Devices 0-7)', style='filled', 
                   fillcolor=stage0_color, color='black', rank='same')
        
        # Layer 0
        stage0.node('l0_input', 'Layer 0 Input\n[128, 100000, 4096]', 
                   shape='ellipse', style='filled', fillcolor='white')
        
        # Pre-Attention LayerNorm
        stage0.node('l0_layernorm1', 'Pre-Attention LN\n[128, 100000, 4096]', 
                   shape='rectangle', style='filled', fillcolor='white')
        
        # QKV Projections - Split across 8 devices
        for i in range(8):
            stage0.node(f'l0_q_proj_{i}', f'Q Proj {i}\n[128, 100000, 512]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            stage0.node(f'l0_k_proj_{i}', f'K Proj {i}\n[128, 100000, 512]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            stage0.node(f'l0_v_proj_{i}', f'V Proj {i}\n[128, 100000, 512]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Multi-Head Attention - Split across all devices
        stage0.node('l0_attention', 'Multi-Head Attention\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        
        # Post-Attention All-Reduce
        stage0.node('l0_attn_allreduce', 'All-Reduce\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='ellipse', style='dashed', fillcolor='white')
        
        # Residual Add
        stage0.node('l0_residual1', 'Residual Add\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='parallelogram', style='filled', fillcolor='white')
        
        # LayerNorm 2
        stage0.node('l0_layernorm2', 'Pre-MLP LN\n[128, 100000, 4096]', 
                   shape='rectangle', style='filled', fillcolor='white')
        
        # MLP - Split across 8 devices
        for i in range(8):
            stage0.node(f'l0_mlp_gate_{i}', f'MLP Gate {i}\n[128, 100000, 4096]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            stage0.node(f'l0_mlp_up_{i}', f'MLP Up {i}\n[128, 100000, 4096]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
            stage0.node(f'l0_mlp_down_{i}', f'MLP Down {i}\n[128, 100000, 4096]\nDevice {i}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # MLP Reduce
        stage0.node('l0_mlp_reduce', 'MLP Reduce\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='ellipse', style='dashed', fillcolor='white')
        
        # Final Residual
        stage0.node('l0_output', 'Layer 0 Output\n[128, 100000, 4096]', 
                   shape='parallelogram', style='filled', fillcolor='white')
        
        # Layer 1 (similar structure)
        stage0.node('l1_input', 'Layer 1 Input\n[128, 100000, 4096]', 
                   shape='ellipse', style='filled', fillcolor='white')
        stage0.node('l1_layernorm1', 'Pre-Attention LN\n[128, 100000, 4096]', 
                   shape='rectangle', style='filled', fillcolor='white')
        stage0.node('l1_attention', 'Multi-Head Attention\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        stage0.node('l1_residual1', 'Residual Add\n[128, 100000, 4096]\nAll Devices 0-7', 
                   shape='parallelogram', style='filled', fillcolor='white')
        stage0.node('l1_output', 'Layer 1 Output\n[128, 100000, 4096]', 
                   shape='parallelogram', style='filled', fillcolor='white')
    
    # Pipeline Stage 1 - Devices 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1\n(Devices 8-15)', style='filled', 
                   fillcolor=stage1_color, color='black', rank='same')
        
        # Layer 2
        stage1.node('l2_input', 'Layer 2 Input\n[128, 100000, 4096]', 
                   shape='ellipse', style='filled', fillcolor='white')
        stage1.node('l2_attention', 'Multi-Head Attention\n[128, 100000, 4096]\nAll Devices 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        stage1.node('l2_output', 'Layer 2 Output\n[128, 100000, 4096]', 
                   shape='parallelogram', style='filled', fillcolor='white')
        
        # Layer 3
        stage1.node('l3_input', 'Layer 3 Input\n[128, 100000, 4096]', 
                   shape='ellipse', style='filled', fillcolor='white')
        stage1.node('l3_attention', 'Multi-Head Attention\n[128, 100000, 4096]\nAll Devices 8-15', 
                   shape='rectangle', style='filled', fillcolor='lightblue')
        stage1.node('l3_output', 'Layer 3 Output\n[128, 100000, 4096]', 
                   shape='parallelogram', style='filled', fillcolor='white')
    
    # Output
    dot.node('output', 'Output\n[batch_size=128, seq_len=100000, d_model=4096]', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connections
    dot.edge('input', 'l0_input')
    dot.edge('l0_input', 'l0_layernorm1')
    
    # Connect QKV projections
    for i in range(8):
        dot.edge('l0_layernorm1', f'l0_q_proj_{i}')
        dot.edge('l0_layernorm1', f'l0_k_proj_{i}')
        dot.edge('l0_layernorm1', f'l0_v_proj_{i}')
    
    # Connect to attention
    for i in range(8):
        dot.edge(f'l0_q_proj_{i}', 'l0_attention')
        dot.edge(f'l0_k_proj_{i}', 'l0_attention')
        dot.edge(f'l0_v_proj_{i}', 'l0_attention')
    
    dot.edge('l0_attention', 'l0_attn_allreduce')
    dot.edge('l0_input', 'l0_residual1')  # Residual connection
    dot.edge('l0_attn_allreduce', 'l0_residual1')
    dot.edge('l0_residual1', 'l0_layernorm2')
    
    # MLP connections
    for i in range(8):
        dot.edge('l0_layernorm2', f'l0_mlp_gate_{i}')
        dot.edge('l0_layernorm2', f'l0_mlp_up_{i}')
        dot.edge(f'l0_mlp_gate_{i}', f'l0_mlp_down_{i}')
        dot.edge(f'l0_mlp_up_{i}', f'l0_mlp_down_{i}')
        dot.edge(f'l0_mlp_down_{i}', 'l0_mlp_reduce')
    
    dot.edge('l0_residual1', 'l0_output')  # Residual connection
    dot.edge('l0_mlp_reduce', 'l0_output')
    dot.edge('l0_output', 'l1_input')
    
    # Layer 1 connections (simplified)
    dot.edge('l1_input', 'l1_layernorm1')
    dot.edge('l1_layernorm1', 'l1_attention')
    dot.edge('l1_input', 'l1_residual1')
    dot.edge('l1_attention', 'l1_residual1')
    dot.edge('l1_residual1', 'l1_output')
    
    # Pipeline communication
    dot.edge('l1_output', 'l2_input', style='dashed', label='Pipeline Send')
    dot.edge('l2_input', 'l2_attention')
    dot.edge('l2_attention', 'l2_output')
    dot.edge('l2_output', 'l3_input')
    dot.edge('l3_input', 'l3_attention')
    dot.edge('l3_attention', 'l3_output')
    dot.edge('l3_output', 'output')
    
    return dot

def create_proposed_dag():
    """
    Create proposed DAG with Ring Attention + Sequence Parallelism
    16 GPUs, sequence split 6250 tokens per device
    """
    dot = graphviz.Digraph('proposed_transformer', 
                          comment='Dense Transformer - Ring Attention + Sequence Parallel')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='1.0')
    
    # Define node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 
              'lightpink', 'lightcyan', 'lightgray', 'lightorange',
              'lightsteelblue', 'lightseagreen', 'lightsalmon', 'lightseagreen',
              'lightsteelblue', 'lightskyblue', 'lightgoldenrod', 'lightcoral']
    
    # Input node
    dot.node('input', 'Input\n[batch_size=128, seq_len=100000, d_model=4096]', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Sequence Split
    dot.node('sequence_split', 'Sequence Split\nInto 16 parts\n[128, 6250, 4096] each', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Create nodes for each device
    for device_id in range(16):
        with dot.subgraph(name=f'cluster_device_{device_id}') as dev:
            dev.attr(label=f'Device {device_id} (6250 tokens)', style='filled', 
                    fillcolor=colors[device_id], color='black')
            
            # Local sequence processing
            dev.node(f'd{device_id}_input', f'Local Input\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='ellipse', style='filled', fillcolor='white')
            
            # Layer 0
            dev.node(f'd{device_id}_ln1', f'LayerNorm\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='white')
            
            # QKV Full projections (no split)
            dev.node(f'd{device_id}_q', f'Q Projection\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            dev.node(f'd{device_id}_k', f'K Projection\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            dev.node(f'd{device_id}_v', f'V Projection\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Head split for attention
            dev.node(f'd{device_id}_q_heads', f'Split to Heads\n[128, 6250, 32, 128]\nDevice {device_id}', 
                    shape='ellipse', style='filled', fillcolor='white')
            dev.node(f'd{device_id}_k_heads', f'Split to Heads\n[128, 6250, 32, 128]\nDevice {device_id}', 
                    shape='ellipse', style='filled', fillcolor='white')
            dev.node(f'd{device_id}_v_heads', f'Split to Heads\n[128, 6250, 32, 128]\nDevice {device_id}', 
                    shape='ellipse', style='filled', fillcolor='white')
            
            # Ring attention computation
            dev.node(f'd{device_id}_ring_attn', f'Ring Attention\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightblue')
            
            # Ring communication nodes
            dev.node(f'd{device_id}_send_kv', f'Send KV Block\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='ellipse', style='dashed', fillcolor='white')
            dev.node(f'd{device_id}_recv_kv', f'Receive KV Block\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='ellipse', style='dashed', fillcolor='white')
            
            # Post-attention
            dev.node(f'd{device_id}_out_proj', f'Output Projection\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightcoral')
            dev.node(f'd{device_id}_res1', f'Residual Add\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='parallelogram', style='filled', fillcolor='white')
            
            # MLP
            dev.node(f'd{device_id}_ln2', f'LayerNorm\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='white')
            dev.node(f'd{device_id}_mlp_gate', f'MLP Gate\n[128, 6250, 16384]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dev.node(f'd{device_id}_mlp_up', f'MLP Up\n[128, 6250, 16384]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dev.node(f'd{device_id}_mlp_down', f'MLP Down\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgreen')
            dev.node(f'd{device_id}_res2', f'Residual Add\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='parallelogram', style='filled', fillcolor='white')
            
            # Layer 1-3 (abbreviated)
            dev.node(f'd{device_id}_layer1', f'Layer 1\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgray')
            dev.node(f'd{device_id}_layer2', f'Layer 2\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgray')
            dev.node(f'd{device_id}_layer3', f'Layer 3\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='rectangle', style='filled', fillcolor='lightgray')
            
            dev.node(f'd{device_id}_output', f'Local Output\n[128, 6250, 4096]\nDevice {device_id}', 
                    shape='parallelogram', style='filled', fillcolor='white')
    
    # Sequence aggregation
    dot.node('sequence_agg', 'Sequence Aggregation\n[128, 100000, 4096]', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Output
    dot.node('output', 'Output\n[batch_size=128, seq_len=100000, d_model=4096]', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connections for each device
    dot.edge('input', 'sequence_split')
    
    for device_id in range(16):
        dot.edge('sequence_split', f'd{device_id}_input')
        dot.edge(f'd{device_id}_input', f'd{device_id}_ln1')
        dot.edge(f'd{device_id}_ln1', f'd{device_id}_q')
        dot.edge(f'd{device_id}_ln1', f'd{device_id}_k')
        dot.edge(f'd{device_id}_ln1', f'd{device_id}_v')
        
        dot.edge(f'd{device_id}_q', f'd{device_id}_q_heads')
        dot.edge(f'd{device_id}_k', f'd{device_id}_k_heads')
        dot.edge(f'd{device_id}_v', f'd{device_id}_v_heads')
        
        # Ring attention flow
        dot.edge(f'd{device_id}_q_heads', f'd{device_id}_ring_attn')
        dot.edge(f'd{device_id}_k_heads', f'd{device_id}_ring_attn')
        dot.edge(f'd{device_id}_v_heads', f'd{device_id}_ring_attn')
        
        # Ring communication
        dot.edge(f'd{device_id}_ring_attn', f'd{device_id}_send_kv')
        
        # Ring topology connections
        next_device = (device_id + 1) % 16
        prev_device = (device_id - 1) % 16
        dot.edge(f'd{device_id}_send_kv', f'd{next_device}_recv_kv', 
                style='dashed', constraint='false')
        dot.edge(f'd{prev_device}_send_kv', f'd{device_id}_recv_kv', 
                style='dashed', constraint='false')
        
        # Continue flow
        dot.edge(f'd{device_id}_ring_attn', f'd{device_id}_out_proj')
        dot.edge(f'd{device_id}_input', f'd{device_id}_res1')  # Residual
        dot.edge(f'd{device_id}_out_proj', f'd{device_id}_res1')
        
        dot.edge(f'd{device_id}_res1', f'd{device_id}_ln2')
        dot.edge(f'd{device_id}_ln2', f'd{device_id}_mlp_gate')
        dot.edge(f'd{device_id}_ln2', f'd{device_id}_mlp_up')
        dot.edge(f'd{device_id}_mlp_gate', f'd{device_id}_mlp_down')
        dot.edge(f'd{device_id}_mlp_up', f'd{device_id}_mlp_down')
        dot.edge(f'd{device_id}_mlp_down', f'd{device_id}_res2')
        dot.edge(f'd{device_id}_res1', f'd{device_id}_res2')  # Residual
        
        # Layer connections (simplified)
        dot.edge(f'd{device_id}_res2', f'd{device_id}_layer1')
        dot.edge(f'd{device_id}_layer1', f'd{device_id}_layer2')
        dot.edge(f'd{device_id}_layer2', f'd{device_id}_layer3')
        dot.edge(f'd{device_id}_layer3', f'd{device_id}_output')
        
        dot.edge(f'd{device_id}_output', 'sequence_agg')
    
    dot.edge('sequence_agg', 'output')
    
    return dot

def main():
    # Create output directory
    os.makedirs('../outputs/2025-11-24-09-19-47', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    
    # Save DOT file
    with open('../outputs/2025-11-24-09-19-47/baseline_transformer.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    # Render SVG
    baseline_dag.render('../outputs/2025-11-24-09-19-47/baseline_transformer', 
                       format='svg', cleanup=False)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    
    # Save DOT file
    with open('../outputs/2025-11-24-09-19-47/proposed_transformer.dot', 'w') as f:
        f.write(proposed_dag.source)
    
    # Render SVG
    proposed_dag.render('../outputs/2025-11-24-09-19-47/proposed_transformer', 
                       format='svg', cleanup=False)
    
    # Create summary JSON
    summary = {
        "generated_dags": [
            {
                "name": "baseline_transformer",
                "description": "Dense Transformer with Tensor Parallelism (8-way) and Pipeline Parallelism (2-way)",
                "devices": 16,
                "strategy": "TP=8, PP=2",
                "files": [
                    "../outputs/2025-11-24-09-19-47/baseline_transformer.dot",
                    "../outputs/2025-11-24-09-19-47/baseline_transformer.svg"
                ]
            },
            {
                "name": "proposed_transformer", 
                "description": "Dense Transformer with Ring Attention + Sequence Parallelism",
                "devices": 16,
                "strategy": "Sequence Parallel=16, Ring Attention",
                "files": [
                    "../outputs/2025-11-24-09-19-47/proposed_transformer.dot",
                    "../outputs/2025-11-24-09-19-47/proposed_transformer.svg"
                ]
            }
        ]
    }
    
    with open('../outputs/2025-11-24-09-19-47/dag_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("DAGs generated successfully:")
    print("- baseline_transformer.dot / .svg")
    print("- proposed_transformer.dot / .svg")
    print("- dag_summary.json")

if __name__ == "__main__":
    main()