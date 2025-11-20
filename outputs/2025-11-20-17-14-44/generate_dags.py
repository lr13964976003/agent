#!/usr/bin/env python3

import os
from pathlib import Path
import graphviz

def create_baseline_dag():
    """
    Create DAG for baseline model with tensor parallelism (TP=8) + pipeline parallelism (PP=2)
    4-layer dense transformer with 16 devices total
    Pipeline: stage 0 (devices 0-7) has layers 0,1; stage 1 (devices 8-15) has layers 2,3
    Tensor parallelism splits each layer across 8 devices
    """
    
    dot = graphviz.Digraph('baseline_dag', filename='baseline_dag')
    dot.attr(rankdir='TB', compound='true', concentrate='true')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: all GPUs', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create subgraphs for pipeline stages
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0\\nDevices: 0-7\\nLayers: 0,1', style='rounded', fillcolor='lightgray', bgcolor='lightgray')
        
        # Layer 0
        with c0.subgraph(name='cluster_layer_0') as l0:
            l0.attr(label='Layer 0', style='rounded')
            
            # Split input across 8 devices for tensor parallelism
            for tp_rank in range(8):
                device_id = tp_rank
                # Input split for layer 0
                l0.node(f'l0_split_q_{tp_rank}', f'Q Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                l0.node(f'l0_split_k_{tp_rank}', f'K Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                l0.node(f'l0_split_v_{tp_rank}', f'V Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                # Linear projections
                l0.node(f'l0_q_proj_{tp_rank}', f'Q Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                l0.node(f'l0_k_proj_{tp_rank}', f'K Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                l0.node(f'l0_v_proj_{tp_rank}', f'V Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Multi-head attention
                l0.node(f'l0_mha_{tp_rank}', f'MHA\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # MLP
                l0.node(f'l0_mlp_fc_{tp_rank}', f'MLP FC\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightpink')
                l0.node(f'l0_mlp_proj_{tp_rank}', f'MLP Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightpink')
                
                # Residual connections
                l0.node(f'l0_residual_{tp_rank}', f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                # Concatenation
                l0.node(f'l0_concat_{tp_rank}', f'TP Concat\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        # Layer 1 (similar structure)
        with c0.subgraph(name='cluster_layer_1') as l1:
            l1.attr(label='Layer 1', style='rounded')
            
            for tp_rank in range(8):
                device_id = tp_rank
                l1.node(f'l1_split_q_{tp_rank}', f'Q Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                l1.node(f'l1_split_k_{tp_rank}', f'K Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                l1.node(f'l1_split_v_{tp_rank}', f'V Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
                
                l1.node(f'l1_q_proj_{tp_rank}', f'Q Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                l1.node(f'l1_k_proj_{tp_rank}', f'K Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                l1.node(f'l1_v_proj_{tp_rank}', f'V Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
                
                l1.node(f'l1_mha_{tp_rank}', f'MHA\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightgreen')
                
                l1.node(f'l1_mlp_fc_{tp_rank}', f'MLP FC\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightpink')
                l1.node(f'l1_mlp_proj_{tp_rank}', f'MLP Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightpink')
                
                l1.node(f'l1_residual_{tp_rank}', f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='rectangle', style='filled', fillcolor='lightsteelblue')
                
                l1.node(f'l1_concat_{tp_rank}', f'TP Concat\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                       shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Pipeline stage 1
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1\\nDevices: 8-15\\nLayers: 2,3', style='rounded', fillcolor='lightgray', bgcolor='lightgray')
        
        # Layer 2 and 3 (similar structure)
        for layer in [2, 3]:
            with c1.subgraph(name=f'cluster_layer_{layer}') as l:
                l.attr(label=f'Layer {layer}', style='rounded')
                
                for tp_rank in range(8):
                    device_id = 8 + tp_rank
                    l.node(f'l{layer}_split_q_{tp_rank}', f'Q Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='parallelogram', style='filled', fillcolor='lightyellow')
                    l.node(f'l{layer}_split_k_{tp_rank}', f'K Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='parallelogram', style='filled', fillcolor='lightyellow')
                    l.node(f'l{layer}_split_v_{tp_rank}', f'V Split\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='parallelogram', style='filled', fillcolor='lightyellow')
                    
                    l.node(f'l{layer}_q_proj_{tp_rank}', f'Q Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightcoral')
                    l.node(f'l{layer}_k_proj_{tp_rank}', f'K Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightcoral')
                    l.node(f'l{layer}_v_proj_{tp_rank}', f'V Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightcoral')
                    
                    l.node(f'l{layer}_mha_{tp_rank}', f'MHA\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightgreen')
                    
                    l.node(f'l{layer}_mlp_fc_{tp_rank}', f'MLP FC\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightpink')
                    l.node(f'l{layer}_mlp_proj_{tp_rank}', f'MLP Projection\\nInput: [batch_size=128, seq_len=10000, hidden_size=2048]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightpink')
                    
                    l.node(f'l{layer}_residual_{tp_rank}', f'Residual Add\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='rectangle', style='filled', fillcolor='lightsteelblue')
                    
                    l.node(f'l{layer}_concat_{tp_rank}', f'TP Concat\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=512]\\nGPU: {device_id}', 
                           shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Final aggregation and output
    dot.node('final_concat', 'Final Concat\\nInput: [batch_size=128, seq_len=10000, hidden_size=512]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: all GPUs', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: all GPUs', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connections (simplified for brevity - each device has full connectivity)
    for tp_rank in range(8):
        # Layer 0 connections
        dot.edge('input', f'l0_split_q_{tp_rank}')
        dot.edge('input', f'l0_split_k_{tp_rank}')
        dot.edge('input', f'l0_split_v_{tp_rank}')
        dot.edge(f'l0_split_q_{tp_rank}', f'l0_q_proj_{tp_rank}')
        dot.edge(f'l0_split_k_{tp_rank}', f'l0_k_proj_{tp_rank}')
        dot.edge(f'l0_split_v_{tp_rank}', f'l0_v_proj_{tp_rank}')
        dot.edge(f'l0_q_proj_{tp_rank}', f'l0_mha_{tp_rank}')
        dot.edge(f'l0_k_proj_{tp_rank}', f'l0_mha_{tp_rank}')
        dot.edge(f'l0_v_proj_{tp_rank}', f'l0_mha_{tp_rank}')
        dot.edge(f'l0_mha_{tp_rank}', f'l0_residual_{tp_rank}')
        dot.edge(f'l0_residual_{tp_rank}', f'l0_mlp_fc_{tp_rank}')
        dot.edge(f'l0_mlp_fc_{tp_rank}', f'l0_mlp_proj_{tp_rank}')
        dot.edge(f'l0_mlp_proj_{tp_rank}', f'l0_residual_{tp_rank}')
        dot.edge(f'l0_residual_{tp_rank}', f'l0_concat_{tp_rank}')
        
        # Layer 1 connections
        dot.edge(f'l0_concat_{tp_rank}', f'l1_split_q_{tp_rank}')
        dot.edge(f'l0_concat_{tp_rank}', f'l1_split_k_{tp_rank}')
        dot.edge(f'l0_concat_{tp_rank}', f'l1_split_v_{tp_rank}')
        # ... similar connections for layer 1 ...
        
        # Pipeline communication
        dot.edge(f'l1_concat_{tp_rank}', f'l2_split_q_{tp_rank}', style='dashed', label='PP Send')
        
    return dot

def create_helix_dag():
    """
    Create DAG for Helix two-level attention partitioning model
    4x4 grid = 16 devices total
    Each device handles: 8 heads × 32 dimensions = 1024 dimensions
    """
    
    dot = graphviz.Digraph('helix_dag', filename='helix_dag')
    dot.attr(rankdir='TB', compound='true', concentrate='true')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: all GPUs', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create subgraph for each layer (4 layers total)
    for layer_idx in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer_idx}') as layer:
            layer.attr(label=f'Layer {layer_idx}\\nHelix 4x4 Partitioning', style='rounded', fillcolor='lightgray', bgcolor='lightgray')
            
            # Create 16 device subgraphs for this layer
            for head_group in range(4):
                for dim_slice in range(4):
                    device_id = head_group * 4 + dim_slice
                    head_start = head_group * 8
                    head_end = head_start + 7
                    dim_start = dim_slice * 32
                    dim_end = dim_start + 31
                    
                    # Create cluster for this device
                    with layer.subgraph(name=f'cluster_device_{device_id}_layer_{layer_idx}') as dev:
                        dev.attr(label=f'Device {device_id}\\nHeads {head_start}-{head_end}\\nDims {dim_start}-{dim_end}', 
                                style='rounded', fillcolor='lightcyan', bgcolor='lightcyan')
                        
                        # Input broadcast
                        dev.node(f'l{layer_idx}_broadcast_{device_id}', 
                                f'Input Broadcast\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: {device_id}', 
                                shape='parallelogram', style='filled', fillcolor='lightyellow')
                        
                        # QKV projections for this partition
                        dev.node(f'l{layer_idx}_q_proj_{device_id}', 
                                f'Q Projection\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 256]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightcoral')
                        dev.node(f'l{layer_idx}_k_proj_{device_id}', 
                                f'K Projection\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 256]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightcoral')
                        dev.node(f'l{layer_idx}_v_proj_{device_id}', 
                                f'V Projection\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 256]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightcoral')
                        
                        # Attention computation for partition
                        dev.node(f'l{layer_idx}_attention_{device_id}', 
                                f'Local Attention\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 256]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightgreen')
                        
                        # Output projection for partition
                        dev.node(f'l{layer_idx}_o_proj_{device_id}', 
                                f'O Projection\\nInput: [batch_size=128, seq_len=10000, 256]\\nOutput: [batch_size=128, seq_len=10000, 1024]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightcoral')
                        
                        # MLP for partition
                        dev.node(f'l{layer_idx}_mlp_fc_{device_id}', 
                                f'MLP FC\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightpink')
                        dev.node(f'l{layer_idx}_mlp_proj_{device_id}', 
                                f'MLP Projection\\nInput: [batch_size=128, seq_len=10000, 4096]\\nOutput: [batch_size=128, seq_len=10000, 1024]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightpink')
                        
                        # Residual connections
                        dev.node(f'l{layer_idx}_residual1_{device_id}', 
                                f'Residual Add\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 1024]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightsteelblue')
                        dev.node(f'l{layer_idx}_residual2_{device_id}', 
                                f'Residual Add\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 1024]\\nGPU: {device_id}', 
                                shape='rectangle', style='filled', fillcolor='lightsteelblue')
    
    # Concatenation and communication nodes
    for layer_idx in range(4):
        # Dimension concatenation within each head group (4 devices each)
        for head_group in range(4):
            dot.node(f'l{layer_idx}_dim_concat_{head_group}', 
                    f'Dimension Concat\\nHead Group {head_group}\\nInput: [batch_size=128, seq_len=10000, 32]\\nOutput: [batch_size=128, seq_len=10000, 128]\\nGPU: {head_group*4}-{head_group*4+3}', 
                    shape='parallelogram', style='filled', fillcolor='orange')
        
        # Head concatenation across all groups
        dot.node(f'l{layer_idx}_head_concat', 
                f'Head Concat\\nAll Groups\\nInput: [batch_size=128, seq_len=10000, 1024]\\nOutput: [batch_size=128, seq_len=10000, 4096]\\nGPU: all GPUs', 
                shape='parallelogram', style='filled', fillcolor='gold')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nOutput: [batch_size=128, seq_len=10000, hidden_size=4096]\\nGPU: all GPUs', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Connect the DAG
    # Layer 0
    for device_id in range(16):
        dot.edge('input', f'l0_broadcast_{device_id}')
        dot.edge(f'l0_broadcast_{device_id}', f'l0_q_proj_{device_id}')
        dot.edge(f'l0_broadcast_{device_id}', f'l0_k_proj_{device_id}')
        dot.edge(f'l0_broadcast_{device_id}', f'l0_v_proj_{device_id}')
        dot.edge(f'l0_q_proj_{device_id}', f'l0_attention_{device_id}')
        dot.edge(f'l0_k_proj_{device_id}', f'l0_attention_{device_id}')
        dot.edge(f'l0_v_proj_{device_id}', f'l0_attention_{device_id}')
        dot.edge(f'l0_attention_{device_id}', f'l0_o_proj_{device_id}')
        dot.edge(f'l0_o_proj_{device_id}', f'l0_residual1_{device_id}')
        dot.edge(f'l0_residual1_{device_id}', f'l0_mlp_fc_{device_id}')
        dot.edge(f'l0_mlp_fc_{device_id}', f'l0_mlp_proj_{device_id}')
        dot.edge(f'l0_mlp_proj_{device_id}', f'l0_residual2_{device_id}')
        
        # Connect to dimension concatenation
        head_group = device_id // 4
        dot.edge(f'l0_residual2_{device_id}', f'l0_dim_concat_{head_group}')
    
    # Connect dimension concatenations to head concatenation
    for layer_idx in range(4):
        for head_group in range(4):
            dot.edge(f'l{layer_idx}_dim_concat_{head_group}', f'l{layer_idx}_head_concat')
        
        if layer_idx < 3:
            # Connect to next layer
            dot.edge(f'l{layer_idx}_head_concat', f'l{layer_idx+1}_broadcast_{device_id}')
        else:
            # Final output
            dot.edge(f'l{layer_idx}_head_concat', 'output')
    
    return dot

def main():
    # Create output directory
    output_dir = Path('../outputs/2025-11-20-17-14-44')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    baseline_dag.format = 'svg'
    baseline_dag.render(str(output_dir / 'baseline_dag'), cleanup=True)
    
    # Generate Helix DAG
    print("Generating Helix DAG...")
    helix_dag = create_helix_dag()
    helix_dag.format = 'svg'
    helix_dag.render(str(output_dir / 'helix_dag'), cleanup=True)
    
    # Save DOT files
    with open(output_dir / 'baseline_dag.dot', 'w') as f:
        f.write(baseline_dag.source)
    
    with open(output_dir / 'helix_dag.dot', 'w') as f:
        f.write(helix_dag.source)
    
    print(f"DAGs generated in: {output_dir}")
    print("Files created:")
    print("- baseline_dag.svg")
    print("- baseline_dag.dot")
    print("- helix_dag.svg")
    print("- helix_dag.dot")

if __name__ == '__main__':
    main()