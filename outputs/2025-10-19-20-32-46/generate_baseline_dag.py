#!/usr/bin/env python3
"""
Baseline DAG Generator for Tensor + Pipeline Parallelism
Model: 16-layer dense network with TP=8, PP=2 on 16 GPUs
"""

import os
from graphviz import Digraph

def create_baseline_dag():
    """Create comprehensive DAG for baseline tensor+pipeline parallelism"""
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    hidden_size = 8192
    num_heads = 16
    head_dim = 512
    mlp_hidden_size = 32768
    
    # Create new directed graph
    dag = Digraph('baseline_tensor_pipeline_parallel')
    dag.attr(rankdir='TB', size='100,200')
    dag.attr('node', fontname='Arial', fontsize='10')
    
    # Global styles
    dag.attr('node', shape='rectangle', style='filled')
    dag.attr('edge', fontname='Arial', fontsize='8')
    
    # Create subgraphs for each pipeline stage
    with dag.subgraph(name='cluster_pipeline_stage_0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (Layers 1-8)\nDevices 0-7 (TP=8)', 
                   style='filled', color='lightblue', fillcolor='lightblue1')
        
        # Input to stage 0
        stage0.node('input_stage0', 
                   f'Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Host',
                   shape='ellipse', fillcolor='lightgreen')

        # Process each layer in stage 0
        for layer_num in range(1, 9):
            layer_prefix = f'layer{layer_num}'
            
            # Layer input reception
            stage0.node(f'{layer_prefix}_input_stage0',
                       f'Layer {layer_num} Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       shape='parallelogram', fillcolor='lightyellow')
            
            # Multi-Head Attention - Query Linear (Column Parallel)
            stage0.node(f'{layer_prefix}_mha_query_stage0',
                       f'Layer {layer_num} MHA Query Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_q={head_dim}]\nGPU: Each shard 0-7',
                       fillcolor='pink')
            
            # Multi-Head Attention - Key Linear (Column Parallel)
            stage0.node(f'{layer_prefix}_mha_key_stage0',
                       f'Layer {layer_num} MHA Key Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\nGPU: Each shard 0-7',
                       fillcolor='pink')
            
            # Multi-Head Attention - Value Linear (Column Parallel)
            stage0.node(f'{layer_prefix}_mha_value_stage0',
                       f'Layer {layer_num} MHA Value Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_v={head_dim}]\nGPU: Each shard 0-7',
                       fillcolor='pink')
            
            # Multi-Head Attention - QKT calculation
            stage0.node(f'{layer_prefix}_mha_qkt_stage0',
                       f'Layer {layer_num} MHA QKT Calculation\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: Each shard 0-7',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Attention Weights
            stage0.node(f'{layer_prefix}_mha_weights_stage0',
                       f'Layer {layer_num} MHA Attention Weights\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: Each shard 0-7',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Apply Attention to V
            stage0.node(f'{layer_prefix}_mha_apply_stage0',
                       f'Layer {layer_num} MHA Apply Attention\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_v={head_dim}]\nGPU: Each shard 0-7',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Output Linear (Row Parallel)
            stage0.node(f'{layer_prefix}_mha_out_stage0',
                       f'Layer {layer_num} MHA Output Linear\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_v={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Each shard 0-7',
                       fillcolor='pink')
            
            # Multi-Head Attention - All-reduce
            stage0.node(f'{layer_prefix}_mha_allreduce_stage0',
                       f'Layer {layer_num} MHA All-reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       shape='ellipse', fillcolor='lightgray')
            
            # Residual Add 1
            stage0.node(f'{layer_prefix}_residual1_stage0',
                       f'Layer {layer_num} Residual Add 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 1
            stage0.node(f'{layer_prefix}_layernorm1_stage0',
                       f'Layer {layer_num} Layer Norm 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       fillcolor='lightgreen')
            
            # MLP - Gate Linear (Column Parallel)
            stage0.node(f'{layer_prefix}_mlp_gate_stage0',
                       f'Layer {layer_num} MLP Gate Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 0-7',
                       fillcolor='lightyellow')
            
            # MLP - Up Linear (Column Parallel)
            stage0.node(f'{layer_prefix}_mlp_up_stage0',
                       f'Layer {layer_num} MLP Up Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 0-7',
                       fillcolor='lightyellow')
            
            # MLP - GELU Activation
            stage0.node(f'{layer_prefix}_mlp_gelu_stage0',
                       f'Layer {layer_num} MLP GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 0-7',
                       fillcolor='orange')
            
            # MLP - Down Linear (Row Parallel)
            stage0.node(f'{layer_prefix}_mlp_down_stage0',
                       f'Layer {layer_num} MLP Down Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Each shard 0-7',
                       fillcolor='lightyellow')
            
            # MLP - All-reduce
            stage0.node(f'{layer_prefix}_mlp_allreduce_stage0',
                       f'Layer {layer_num} MLP All-reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       shape='ellipse', fillcolor='lightgray')
            
            # Residual Add 2
            stage0.node(f'{layer_prefix}_residual2_stage0',
                       f'Layer {layer_num} Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 2
            stage0.node(f'{layer_prefix}_layernorm2_stage0',
                       f'Layer {layer_num} Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 0-7',
                       fillcolor='lightgreen')

    with dag.subgraph(name='cluster_pipeline_stage_1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (Layers 9-16)\nDevices 8-15 (TP=8)', 
                   style='filled', color='lightcoral', fillcolor='lightcoral1')
        
        # Continue with layers 9-16 in stage 1
        for layer_num in range(9, 17):
            layer_prefix = f'layer{layer_num}'
            
            # Layer input reception
            stage1.node(f'{layer_prefix}_input_stage1',
                       f'Layer {layer_num} Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       shape='parallelogram', fillcolor='lightyellow')
            
            # Multi-Head Attention - Query Linear (Column Parallel)
            stage1.node(f'{layer_prefix}_mha_query_stage1',
                       f'Layer {layer_num} MHA Query Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_q={head_dim}]\nGPU: Each shard 8-15',
                       fillcolor='pink')
            
            # Multi-Head Attention - Key Linear (Column Parallel)
            stage1.node(f'{layer_prefix}_mha_key_stage1',
                       f'Layer {layer_num} MHA Key Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\nGPU: Each shard 8-15',
                       fillcolor='pink')
            
            # Multi-Head Attention - Value Linear (Column Parallel)
            stage1.node(f'{layer_prefix}_mha_value_stage1',
                       f'Layer {layer_num} MHA Value Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_v={head_dim}]\nGPU: Each shard 8-15',
                       fillcolor='pink')
            
            # Multi-Head Attention - QKT calculation
            stage1.node(f'{layer_prefix}_mha_qkt_stage1',
                       f'Layer {layer_num} MHA QKT Calculation\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: Each shard 8-15',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Attention Weights
            stage1.node(f'{layer_prefix}_mha_weights_stage1',
                       f'Layer {layer_num} MHA Attention Weights\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nGPU: Each shard 8-15',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Apply Attention to V
            stage1.node(f'{layer_prefix}_mha_apply_stage1',
                       f'Layer {layer_num} MHA Apply Attention\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, seq_len={seq_len}]\nOutput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_v={head_dim}]\nGPU: Each shard 8-15',
                       fillcolor='lightcoral')
            
            # Multi-Head Attention - Output Linear (Row Parallel)
            stage1.node(f'{layer_prefix}_mha_out_stage1',
                       f'Layer {layer_num} MHA Output Linear\nInput: [batch_size={batch_size}, heads={num_heads//8}, seq_len={seq_len}, d_v={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Each shard 8-15',
                       fillcolor='pink')
            
            # Multi-Head Attention - All-reduce
            stage1.node(f'{layer_prefix}_mha_allreduce_stage1',
                       f'Layer {layer_num} MHA All-reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       shape='ellipse', fillcolor='lightgray')
            
            # Residual Add 1
            stage1.node(f'{layer_prefix}_residual1_stage1',
                       f'Layer {layer_num} Residual Add 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 1
            stage1.node(f'{layer_prefix}_layernorm1_stage1',
                       f'Layer {layer_num} Layer Norm 1\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       fillcolor='lightgreen')
            
            # MLP - Gate Linear (Column Parallel)
            stage1.node(f'{layer_prefix}_mlp_gate_stage1',
                       f'Layer {layer_num} MLP Gate Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 8-15',
                       fillcolor='lightyellow')
            
            # MLP - Up Linear (Column Parallel)
            stage1.node(f'{layer_prefix}_mlp_up_stage1',
                       f'Layer {layer_num} MLP Up Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 8-15',
                       fillcolor='lightyellow')
            
            # MLP - GELU Activation
            stage1.node(f'{layer_prefix}_mlp_gelu_stage1',
                       f'Layer {layer_num} MLP GELU\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nGPU: Each shard 8-15',
                       fillcolor='orange')
            
            # MLP - Down Linear (Row Parallel)
            stage1.node(f'{layer_prefix}_mlp_down_stage1',
                       f'Layer {layer_num} MLP Down Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, mlp_hidden={mlp_hidden_size//8}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: Each shard 8-15',
                       fillcolor='lightyellow')
            
            # MLP - All-reduce
            stage1.node(f'{layer_prefix}_mlp_allreduce_stage1',
                       f'Layer {layer_num} MLP All-reduce\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       shape='ellipse', fillcolor='lightgray')
            
            # Residual Add 2
            stage1.node(f'{layer_prefix}_residual2_stage1',
                       f'Layer {layer_num} Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}], [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       shape='diamond', fillcolor='lightblue')
            
            # Layer Norm 2
            stage1.node(f'{layer_prefix}_layernorm2_stage1',
                       f'Layer {layer_num} Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
                       fillcolor='lightgreen')

    # Pipeline communication between stages
    dag.node('pipeline_communication',
            f'Pipeline Communication\nTransfer: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nFrom: Devices 0-7\nTo: Devices 8-15',
            shape='ellipse', fillcolor='gray')

    # Output
    dag.node('output',
            f'Final Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\nGPU: All devices 8-15',
            shape='ellipse', fillcolor='red')

    # Create edges for stage 0
    dag.edge('input_stage0', 'layer1_input_stage0')
    
    # Layer 1 connections
    dag.edge('layer1_input_stage0', 'layer1_mha_query_stage0')
    dag.edge('layer1_input_stage0', 'layer1_mha_key_stage0')
    dag.edge('layer1_input_stage0', 'layer1_mha_value_stage0')
    dag.edge('layer1_mha_query_stage0', 'layer1_mha_qkt_stage0')
    dag.edge('layer1_mha_key_stage0', 'layer1_mha_qkt_stage0')
    dag.edge('layer1_mha_qkt_stage0', 'layer1_mha_weights_stage0')
    dag.edge('layer1_mha_weights_stage0', 'layer1_mha_apply_stage0')
    dag.edge('layer1_mha_value_stage0', 'layer1_mha_apply_stage0')
    dag.edge('layer1_mha_apply_stage0', 'layer1_mha_out_stage0')
    dag.edge('layer1_mha_out_stage0', 'layer1_mha_allreduce_stage0')
    dag.edge('layer1_mha_allreduce_stage0', 'layer1_residual1_stage0')
    dag.edge('layer1_input_stage0', 'layer1_residual1_stage0')
    dag.edge('layer1_residual1_stage0', 'layer1_layernorm1_stage0')
    
    dag.edge('layer1_layernorm1_stage0', 'layer1_mlp_gate_stage0')
    dag.edge('layer1_layernorm1_stage0', 'layer1_mlp_up_stage0')
    dag.edge('layer1_mlp_gate_stage0', 'layer1_mlp_gelu_stage0')
    dag.edge('layer1_mlp_gelu_stage0', 'layer1_mlp_down_stage0')
    dag.edge('layer1_mlp_up_stage0', 'layer1_mlp_down_stage0')
    dag.edge('layer1_mlp_down_stage0', 'layer1_mlp_allreduce_stage0')
    dag.edge('layer1_mlp_allreduce_stage0', 'layer1_residual2_stage0')
    dag.edge('layer1_layernorm1_stage0', 'layer1_residual2_stage0')
    dag.edge('layer1_residual2_stage0', 'layer1_layernorm2_stage0')

    # Continue with remaining layers in stage 0
    for layer_num in range(2, 9):
        prev = layer_num - 1
        dag.edge(f'layer{prev}_layernorm2_stage0', f'layer{layer_num}_input_stage0')
        
        # Add all layer connections (simplified pattern)
        dag.edge(f'layer{layer_num}_input_stage0', f'layer{layer_num}_mha_query_stage0')
        dag.edge(f'layer{layer_num}_input_stage0', f'layer{layer_num}_mha_key_stage0')
        dag.edge(f'layer{layer_num}_input_stage0', f'layer{layer_num}_mha_value_stage0')
        dag.edge(f'layer{layer_num}_mha_query_stage0', f'layer{layer_num}_mha_qkt_stage0')
        dag.edge(f'layer{layer_num}_mha_key_stage0', f'layer{layer_num}_mha_qkt_stage0')
        dag.edge(f'layer{layer_num}_mha_qkt_stage0', f'layer{layer_num}_mha_weights_stage0')
        dag.edge(f'layer{layer_num}_mha_weights_stage0', f'layer{layer_num}_mha_apply_stage0')
        dag.edge(f'layer{layer_num}_mha_value_stage0', f'layer{layer_num}_mha_apply_stage0')
        dag.edge(f'layer{layer_num}_mha_apply_stage0', f'layer{layer_num}_mha_out_stage0')
        dag.edge(f'layer{layer_num}_mha_out_stage0', f'layer{layer_num}_mha_allreduce_stage0')
        dag.edge(f'layer{layer_num}_mha_allreduce_stage0', f'layer{layer_num}_residual1_stage0')
        dag.edge(f'layer{layer_num}_input_stage0', f'layer{layer_num}_residual1_stage0')
        dag.edge(f'layer{layer_num}_residual1_stage0', f'layer{layer_num}_layernorm1_stage0')
        
        dag.edge(f'layer{layer_num}_layernorm1_stage0', f'layer{layer_num}_mlp_gate_stage0')
        dag.edge(f'layer{layer_num}_layernorm1_stage0', f'layer{layer_num}_mlp_up_stage0')
        dag.edge(f'layer{layer_num}_mlp_gate_stage0', f'layer{layer_num}_mlp_gelu_stage0')
        dag.edge(f'layer{layer_num}_mlp_gelu_stage0', f'layer{layer_num}_mlp_down_stage0')
        dag.edge(f'layer{layer_num}_mlp_up_stage0', f'layer{layer_num}_mlp_down_stage0')
        dag.edge(f'layer{layer_num}_mlp_down_stage0', f'layer{layer_num}_mlp_allreduce_stage0')
        dag.edge(f'layer{layer_num}_mlp_allreduce_stage0', f'layer{layer_num}_residual2_stage0')
        dag.edge(f'layer{layer_num}_layernorm1_stage0', f'layer{layer_num}_residual2_stage0')
        dag.edge(f'layer{layer_num}_residual2_stage0', f'layer{layer_num}_layernorm2_stage0')

    # Pipeline communication
    dag.edge('layer8_layernorm2_stage0', 'pipeline_communication')
    dag.edge('pipeline_communication', 'layer9_input_stage1')

    # Continue with stage 1 layers
    for layer_num in range(9, 17):
        dag.edge(f'layer{layer_num}_input_stage1', f'layer{layer_num}_mha_query_stage1')
        dag.edge(f'layer{layer_num}_input_stage1', f'layer{layer_num}_mha_key_stage1')
        dag.edge(f'layer{layer_num}_input_stage1', f'layer{layer_num}_mha_value_stage1')
        dag.edge(f'layer{layer_num}_mha_query_stage1', f'layer{layer_num}_mha_qkt_stage1')
        dag.edge(f'layer{layer_num}_mha_key_stage1', f'layer{layer_num}_mha_qkt_stage1')
        dag.edge(f'layer{layer_num}_mha_qkt_stage1', f'layer{layer_num}_mha_weights_stage1')
        dag.edge(f'layer{layer_num}_mha_weights_stage1', f'layer{layer_num}_mha_apply_stage1')
        dag.edge(f'layer{layer_num}_mha_value_stage1', f'layer{layer_num}_mha_apply_stage1')
        dag.edge(f'layer{layer_num}_mha_apply_stage1', f'layer{layer_num}_mha_out_stage1')
        dag.edge(f'layer{layer_num}_mha_out_stage1', f'layer{layer_num}_mha_allreduce_stage1')
        dag.edge(f'layer{layer_num}_mha_allreduce_stage1', f'layer{layer_num}_residual1_stage1')
        dag.edge(f'layer{layer_num}_input_stage1', f'layer{layer_num}_residual1_stage1')
        dag.edge(f'layer{layer_num}_residual1_stage1', f'layer{layer_num}_layernorm1_stage1')
        
        dag.edge(f'layer{layer_num}_layernorm1_stage1', f'layer{layer_num}_mlp_gate_stage1')
        dag.edge(f'layer{layer_num}_layernorm1_stage1', f'layer{layer_num}_mlp_up_stage1')
        dag.edge(f'layer{layer_num}_mlp_gate_stage1', f'layer{layer_num}_mlp_gelu_stage1')
        dag.edge(f'layer{layer_num}_mlp_gelu_stage1', f'layer{layer_num}_mlp_down_stage1')
        dag.edge(f'layer{layer_num}_mlp_up_stage1', f'layer{layer_num}_mlp_down_stage1')
        dag.edge(f'layer{layer_num}_mlp_down_stage1', f'layer{layer_num}_mlp_allreduce_stage1')
        dag.edge(f'layer{layer_num}_mlp_allreduce_stage1', f'layer{layer_num}_residual2_stage1')
        dag.edge(f'layer{layer_num}_layernorm1_stage1', f'layer{layer_num}_residual2_stage1')
        dag.edge(f'layer{layer_num}_residual2_stage1', f'layer{layer_num}_layernorm2_stage1')
        
        # Connect to next layer or output
        if layer_num < 16:
            dag.edge(f'layer{layer_num}_layernorm2_stage1', f'layer{layer_num+1}_input_stage1')
        else:
            dag.edge('layer16_layernorm2_stage1', 'output')

    return dag

if __name__ == '__main__':
    dag = create_baseline_dag()
    
    # Save files
    output_dir = '../outputs/2025-10-19-20-32-46'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DOT file
    dot_path = os.path.join(output_dir, 'baseline_tensor_pipeline_parallel.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG
    svg_path = os.path.join(output_dir, 'baseline_tensor_pipeline_parallel.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Baseline DAG generated:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")