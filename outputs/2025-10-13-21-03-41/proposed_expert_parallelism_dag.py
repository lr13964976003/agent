#!/usr/bin/env python3
"""
Cross-Node Expert Parallelism DAG for MoE Model
EP=16, one expert per GPU
"""

import graphviz

def create_proposed_expert_parallelism_dag():
    """Create complete DAG for proposed cross-node expert parallelism"""
    
    # Create directed graph
    dot = graphviz.Digraph('proposed_expert_parallelism', 
                          comment='Cross-Node Expert Parallelism (EP=16)')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,40', compound='true')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define colors for different components
    colors = {
        'input': 'lightblue',
        'attention': 'lightgreen',
        'gate': 'lightyellow',
        'expert': 'lightcoral',
        'communication': 'lightgray',
        'aggregation': 'lightpink',
        'output': 'lightblue'
    }
    
    # Model dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    mha_heads = 16
    head_dim = 512
    ffn_hidden = 32768
    num_tokens = batch_size * seq_len
    
    # =================================================================================
    # LAYER 0
    # =================================================================================
    
    with dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0', style='rounded', color='black', bgcolor='aliceblue')
        
        # Input to Layer 0
        layer0.node('input_l0', f'Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor=colors['input'])
        
        # Multi-Head Attention for Layer 0
        layer0.node('mha_qkv_l0', f'MHA QKV Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer0.node('mha_attn_l0', f'MHA Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer0.node('mha_out_l0', f'MHA Output Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer0.node('res_add_l0', f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer0.node('norm_l0', f'Layer Norm\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Gate for Layer 0
        layer0.node('gate_l0', f'Gate Network\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['gate'])
        
        # Expert routing and communication
        layer0.node('route_l0', f'Token Routing\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [distributed across 16 GPUs]\nGPU: all GPUs',
                   shape='parallelogram', style='filled, dashed', fillcolor=colors['communication'])
        
        # Create experts for Layer 0 (one per GPU)
        for expert_id in range(16):
            gpu_id = expert_id
            
            # Expert input communication
            layer0.node(f'expert{expert_id}_comm_l0', 
                       f'Expert {expert_id} Communication\nInput: [tokens selected for expert {expert_id}]\nOutput: [tokens on GPU {gpu_id}]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
            
            # Expert computation
            layer0.node(f'expert{expert_id}_l0',
                       f'Expert {expert_id} MLP\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                       shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            # Expert output communication
            layer0.node(f'expert{expert_id}_return_l0',
                       f'Expert {expert_id} Return\nInput: [processed_tokens, hidden={hidden_dim}]\nOutput: [tokens back to original positions]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
        
        # Aggregation after experts
        layer0.node('aggregate_l0', f'Expert Aggregation\nInput: [processed_tokens from all experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        
        layer0.node('res_add2_l0', f'Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer0.node('norm2_l0', f'Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Connect Layer 0 nodes
        layer0.edge('input_l0', 'mha_qkv_l0')
        layer0.edge('mha_qkv_l0', 'mha_attn_l0')
        layer0.edge('mha_attn_l0', 'mha_out_l0')
        layer0.edge('input_l0', 'res_add_l0')  # residual connection
        layer0.edge('mha_out_l0', 'res_add_l0')
        layer0.edge('res_add_l0', 'norm_l0')
        layer0.edge('norm_l0', 'gate_l0')
        layer0.edge('norm_l0', 'route_l0')
        
        # Connect routing to each expert
        for expert_id in range(16):
            layer0.edge('route_l0', f'expert{expert_id}_comm_l0')
            layer0.edge(f'expert{expert_id}_comm_l0', f'expert{expert_id}_l0')
            layer0.edge(f'expert{expert_id}_l0', f'expert{expert_id}_return_l0')
            layer0.edge(f'expert{expert_id}_return_l0', 'aggregate_l0')
        
        layer0.edge('aggregate_l0', 'res_add2_l0')
        layer0.edge('norm_l0', 'res_add2_l0')  # residual connection
        layer0.edge('res_add2_l0', 'norm2_l0')
    
    # =================================================================================
    # LAYER 1 (similar structure to Layer 0)
    # =================================================================================
    
    with dot.subgraph(name='cluster_layer1') as layer1:
        layer1.attr(label='Layer 1', style='rounded', color='black', bgcolor='honeydew')
        
        # Input to Layer 1
        layer1.node('input_l1', f'Layer 1 Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor=colors['input'])
        
        # Multi-Head Attention for Layer 1
        layer1.node('mha_qkv_l1', f'MHA QKV Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer1.node('mha_attn_l1', f'MHA Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer1.node('mha_out_l1', f'MHA Output Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer1.node('res_add_l1', f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer1.node('norm_l1', f'Layer Norm\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Gate for Layer 1
        layer1.node('gate_l1', f'Gate Network\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['gate'])
        
        # Expert routing and communication
        layer1.node('route_l1', f'Token Routing\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [distributed across 16 GPUs]\nGPU: all GPUs',
                   shape='parallelogram', style='filled, dashed', fillcolor=colors['communication'])
        
        # Create experts for Layer 1
        for expert_id in range(16):
            gpu_id = expert_id
            
            layer1.node(f'expert{expert_id}_comm_l1', 
                       f'Expert {expert_id} Communication\nInput: [tokens selected for expert {expert_id}]\nOutput: [tokens on GPU {gpu_id}]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
            
            layer1.node(f'expert{expert_id}_l1',
                       f'Expert {expert_id} MLP\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                       shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            layer1.node(f'expert{expert_id}_return_l1',
                       f'Expert {expert_id} Return\nInput: [processed_tokens, hidden={hidden_dim}]\nOutput: [tokens back to original positions]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
        
        layer1.node('aggregate_l1', f'Expert Aggregation\nInput: [processed_tokens from all experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        
        layer1.node('res_add2_l1', f'Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer1.node('norm2_l1', f'Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Connect Layer 1 nodes
        layer1.edge('input_l1', 'mha_qkv_l1')
        layer1.edge('mha_qkv_l1', 'mha_attn_l1')
        layer1.edge('mha_attn_l1', 'mha_out_l1')
        layer1.edge('input_l1', 'res_add_l1')
        layer1.edge('mha_out_l1', 'res_add_l1')
        layer1.edge('res_add_l1', 'norm_l1')
        layer1.edge('norm_l1', 'gate_l1')
        layer1.edge('norm_l1', 'route_l1')
        
        for expert_id in range(16):
            layer1.edge('route_l1', f'expert{expert_id}_comm_l1')
            layer1.edge(f'expert{expert_id}_comm_l1', f'expert{expert_id}_l1')
            layer1.edge(f'expert{expert_id}_l1', f'expert{expert_id}_return_l1')
            layer1.edge(f'expert{expert_id}_return_l1', 'aggregate_l1')
        
        layer1.edge('aggregate_l1', 'res_add2_l1')
        layer1.edge('norm_l1', 'res_add2_l1')
        layer1.edge('res_add2_l1', 'norm2_l1')
    
    # =================================================================================
    # LAYER 2 (similar structure)
    # =================================================================================
    
    with dot.subgraph(name='cluster_layer2') as layer2:
        layer2.attr(label='Layer 2', style='rounded', color='black', bgcolor='lavender')
        
        layer2.node('input_l2', f'Layer 2 Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor=colors['input'])
        
        layer2.node('mha_qkv_l2', f'MHA QKV Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('mha_attn_l2', f'MHA Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('mha_out_l2', f'MHA Output Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('res_add_l2', f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('norm_l2', f'Layer Norm\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('gate_l2', f'Gate Network\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['gate'])
        
        layer2.node('route_l2', f'Token Routing\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [distributed across 16 GPUs]\nGPU: all GPUs',
                   shape='parallelogram', style='filled, dashed', fillcolor=colors['communication'])
        
        for expert_id in range(16):
            gpu_id = expert_id
            
            layer2.node(f'expert{expert_id}_comm_l2', 
                       f'Expert {expert_id} Communication\nInput: [tokens selected for expert {expert_id}]\nOutput: [tokens on GPU {gpu_id}]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
            
            layer2.node(f'expert{expert_id}_l2',
                       f'Expert {expert_id} MLP\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                       shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            layer2.node(f'expert{expert_id}_return_l2',
                       f'Expert {expert_id} Return\nInput: [processed_tokens, hidden={hidden_dim}]\nOutput: [tokens back to original positions]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
        
        layer2.node('aggregate_l2', f'Expert Aggregation\nInput: [processed_tokens from all experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        
        layer2.node('res_add2_l2', f'Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer2.node('norm2_l2', f'Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Connect Layer 2 nodes
        layer2.edge('input_l2', 'mha_qkv_l2')
        layer2.edge('mha_qkv_l2', 'mha_attn_l2')
        layer2.edge('mha_attn_l2', 'mha_out_l2')
        layer2.edge('input_l2', 'res_add_l2')
        layer2.edge('mha_out_l2', 'res_add_l2')
        layer2.edge('res_add_l2', 'norm_l2')
        layer2.edge('norm_l2', 'gate_l2')
        layer2.edge('norm_l2', 'route_l2')
        
        for expert_id in range(16):
            layer2.edge('route_l2', f'expert{expert_id}_comm_l2')
            layer2.edge(f'expert{expert_id}_comm_l2', f'expert{expert_id}_l2')
            layer2.edge(f'expert{expert_id}_l2', f'expert{expert_id}_return_l2')
            layer2.edge(f'expert{expert_id}_return_l2', 'aggregate_l2')
        
        layer2.edge('aggregate_l2', 'res_add2_l2')
        layer2.edge('norm_l2', 'res_add2_l2')
        layer2.edge('res_add2_l2', 'norm2_l2')
    
    # =================================================================================
    # LAYER 3 (similar structure)
    # =================================================================================
    
    with dot.subgraph(name='cluster_layer3') as layer3:
        layer3.attr(label='Layer 3', style='rounded', color='black', bgcolor='mistyrose')
        
        layer3.node('input_l3', f'Layer 3 Input\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor=colors['input'])
        
        layer3.node('mha_qkv_l3', f'MHA QKV Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('mha_attn_l3', f'MHA Attention\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('mha_out_l3', f'MHA Output Linear\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={mha_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('res_add_l3', f'Residual Add\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('norm_l3', f'Layer Norm\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('gate_l3', f'Gate Network\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts=16]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['gate'])
        
        layer3.node('route_l3', f'Token Routing\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [distributed across 16 GPUs]\nGPU: all GPUs',
                   shape='parallelogram', style='filled, dashed', fillcolor=colors['communication'])
        
        for expert_id in range(16):
            gpu_id = expert_id
            
            layer3.node(f'expert{expert_id}_comm_l3', 
                       f'Expert {expert_id} Communication\nInput: [tokens selected for expert {expert_id}]\nOutput: [tokens on GPU {gpu_id}]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
            
            layer3.node(f'expert{expert_id}_l3',
                       f'Expert {expert_id} MLP\nInput: [variable_tokens, hidden={hidden_dim}]\nOutput: [variable_tokens, hidden={hidden_dim}]\nGPU: {gpu_id}',
                       shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            layer3.node(f'expert{expert_id}_return_l3',
                       f'Expert {expert_id} Return\nInput: [processed_tokens, hidden={hidden_dim}]\nOutput: [tokens back to original positions]\nGPU: {gpu_id}',
                       shape='ellipse', style='filled', fillcolor=colors['communication'])
        
        layer3.node('aggregate_l3', f'Expert Aggregation\nInput: [processed_tokens from all experts]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        
        layer3.node('res_add2_l3', f'Residual Add 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}] (x2)\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        layer3.node('norm2_l3', f'Layer Norm 2\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='rectangle', style='filled', fillcolor=colors['attention'])
        
        # Final output
        layer3.node('final_output', f'Final Output\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nGPU: all GPUs',
                   shape='ellipse', style='filled', fillcolor=colors['output'])
        
        # Connect Layer 3 nodes
        layer3.edge('input_l3', 'mha_qkv_l3')
        layer3.edge('mha_qkv_l3', 'mha_attn_l3')
        layer3.edge('mha_attn_l3', 'mha_out_l3')
        layer3.edge('input_l3', 'res_add_l3')
        layer3.edge('mha_out_l3', 'res_add_l3')
        layer3.edge('res_add_l3', 'norm_l3')
        layer3.edge('norm_l3', 'gate_l3')
        layer3.edge('norm_l3', 'route_l3')
        
        for expert_id in range(16):
            layer3.edge('route_l3', f'expert{expert_id}_comm_l3')
            layer3.edge(f'expert{expert_id}_comm_l3', f'expert{expert_id}_l3')
            layer3.edge(f'expert{expert_id}_l3', f'expert{expert_id}_return_l3')
            layer3.edge(f'expert{expert_id}_return_l3', 'aggregate_l3')
        
        layer3.edge('aggregate_l3', 'res_add2_l3')
        layer3.edge('norm_l3', 'res_add2_l3')
        layer3.edge('res_add2_l3', 'norm2_l3')
        layer3.edge('norm2_l3', 'final_output')
    
    # =================================================================================
    # INTER-LAYER CONNECTIONS
    # =================================================================================
    
    # Connect layers
    dot.edge('norm2_l0', 'input_l1')
    dot.edge('norm2_l1', 'input_l2')
    dot.edge('norm2_l2', 'input_l3')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_expert_parallelism_dag()
    
    # Save DOT file
    dag.save(directory='./outputs/2025-10-13-21-03-41', filename='proposed_expert_parallelism_dag.dot')
    
    # Render to SVG
    dag.render(directory='./outputs/2025-10-13-21-03-41', filename='proposed_expert_parallelism_dag', format='svg', cleanup=True)
    
    print("Proposed Expert Parallelism DAG generated successfully!")
    print("Files saved:")
    print("- proposed_expert_parallelism_dag.dot")
    print("- proposed_expert_parallelism_dag.svg")