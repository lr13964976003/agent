#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def generate_moe_dag():
    """Generate complete DAG for 61-layer MoE with large-scale cross-node expert parallelism"""
    
    # Create DAG
    dot = Digraph(comment='Large-Scale Cross-Node Expert Parallelism MoE DAG')
    dot.attr(rankdir='TB', size='50,50')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    batch_size = "batch_size"
    seq_len = "seq_len"
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    hidden_size = 2048
    num_experts = 16
    total_layers = 61
    dense_layers = 3
    moe_layers = 58
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('input', 
               f'INPUT\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Dense layers (first 3 layers)
    for layer in range(1, dense_layers + 1):
        with dot.subgraph(name=f'cluster_dense_{layer}') as c:
            c.attr(label=f'Dense Layer {layer}', style='rounded,filled', fillcolor='lightgray')
            
            # MHA computation
            c.node(f'dense_{layer}_mha', 
                   f'MHA Computation\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # FFN computation
            c.node(f'dense_{layer}_ffn', 
                   f'FFN Computation\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Residual connections
            c.node(f'dense_{layer}_res1', 
                   f'Residual Add 1\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            c.node(f'dense_{layer}_res2', 
                   f'Residual Add 2\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
    
    # MoE layers (layers 4-61)
    for layer in range(4, total_layers + 1):
        with dot.subgraph(name=f'cluster_moe_{layer}') as c:
            c.attr(label=f'MoE Layer {layer}', style='rounded,filled', fillcolor='lightgray')
            
            # MHA computation (same as dense layers)
            c.node(f'moe_{layer}_mha', 
                   f'MHA Computation\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Gating network
            c.node(f'moe_{layer}_gate', 
                   f'Gating Network\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Expert routing (communication)
            c.node(f'moe_{layer}_route', 
                   f'Token Routing\\nGPU: All GPUs\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: Distributed to experts',
                   shape='ellipse', fillcolor='lightblue')
            
            # Individual experts (one per GPU)
            for expert_id in range(num_experts):
                gpu_id = expert_id % 16  # 16 GPUs per layer
                c.node(f'moe_{layer}_expert_{expert_id}', 
                       f'Expert {expert_id} MLP\\nGPU: {gpu_id}\\nInput: [batch_size_subset, token_dim={token_dim}]\\nOutput: [batch_size_subset, token_dim={token_dim}]',
                       shape='rectangle', fillcolor='lightgreen')
            
            # Expert output aggregation
            c.node(f'moe_{layer}_aggregate', 
                   f'Expert Output Aggregation\\nGPU: All GPUs\\nInput: [batch_size_subset, token_dim={token_dim}] x 16\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='ellipse', fillcolor='lightblue')
            
            # FFN computation (after MoE)
            c.node(f'moe_{layer}_ffn', 
                   f'FFN Computation\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Residual connections
            c.node(f'moe_{layer}_res1', 
                   f'Residual Add 1\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            c.node(f'moe_{layer}_res2', 
                   f'Residual Add 2\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('output', 
               f'OUTPUT\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to first dense layer
    dot.edge('input', 'dense_1_mha')
    
    # Dense layer connections
    for layer in range(1, dense_layers + 1):
        # MHA -> Res1 -> FFN -> Res2
        dot.edge(f'dense_{layer}_mha', f'dense_{layer}_res1')
        dot.edge(f'dense_{layer}_res1', f'dense_{layer}_ffn')
        dot.edge(f'dense_{layer}_ffn', f'dense_{layer}_res2')
        
        # Connect to next layer
        if layer < dense_layers:
            dot.edge(f'dense_{layer}_res2', f'dense_{layer+1}_mha')
        else:
            dot.edge(f'dense_{layer}_res2', 'moe_4_mha')
    
    # MoE layer connections
    for layer in range(4, total_layers + 1):
        # MHA -> Res1 -> Gate -> Route -> Experts -> Aggregate -> FFN -> Res2
        dot.edge(f'moe_{layer}_mha', f'moe_{layer}_res1')
        dot.edge(f'moe_{layer}_res1', f'moe_{layer}_gate')
        dot.edge(f'moe_{layer}_gate', f'moe_{layer}_route')
        
        # Route to all experts
        for expert_id in range(num_experts):
            dot.edge(f'moe_{layer}_route', f'moe_{layer}_expert_{expert_id}')
        
        # All experts to aggregate
        for expert_id in range(num_experts):
            dot.edge(f'moe_{layer}_expert_{expert_id}', f'moe_{layer}_aggregate')
        
        dot.edge(f'moe_{layer}_aggregate', f'moe_{layer}_ffn')
        dot.edge(f'moe_{layer}_ffn', f'moe_{layer}_res2')
        
        # Connect to next layer
        if layer < total_layers:
            dot.edge(f'moe_{layer}_res2', f'moe_{layer+1}_mha')
        else:
            dot.edge(f'moe_{layer}_res2', 'output')
    
    return dot

def generate_simplified_moe_dag():
    """Generate simplified DAG showing key components"""
    
    dot = Digraph(comment='Simplified Large-Scale Cross-Node Expert Parallelism MoE DAG')
    dot.attr(rankdir='TB', size='30,30')
    dot.attr('node', fontname='Arial', fontsize='12')
    
    # Model parameters
    batch_size = "batch_size"
    seq_len = "seq_len"
    token_dim = 7168
    num_experts = 16
    
    # Input
    dot.node('input', f'Input\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Dense layers block
    dot.node('dense_block', f'Dense Layers (1-3)\\nReplicated Across GPUs\\nMHA + FFN', 
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # MoE layer template
    dot.node('mha', f'Multi-Head Attention\\nReplicated\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    dot.node('gate', f'Gating Network\\nTop-K Selection\\n[batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]', 
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    dot.node('route', f'Token Routing\\nCross-Node Communication', 
             shape='ellipse', fillcolor='lightblue', style='filled')
    
    # Expert cluster
    with dot.subgraph(name='cluster_experts') as c:
        c.attr(label='Expert Parallelism (16 Experts)', style='rounded,filled', fillcolor='lightgray')
        for i in range(4):  # Show 4 experts as example
            c.node(f'expert_{i}', f'Expert {i}\\nGPU {i}\\nOne Expert Per GPU', 
                   shape='rectangle', fillcolor='lightgreen', style='filled')
    
    dot.node('aggregate', f'Expert Aggregation\\nCross-Node Communication', 
             shape='ellipse', fillcolor='lightblue', style='filled')
    
    dot.node('ffn', f'FFN Layer\\nReplicated\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Output
    dot.node('output', f'Output\\n[batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Connections
    dot.edge('input', 'dense_block')
    dot.edge('dense_block', 'mha')
    dot.edge('mha', 'gate')
    dot.edge('gate', 'route')
    dot.edge('route', 'expert_0')
    dot.edge('route', 'expert_1')
    dot.edge('route', 'expert_2')
    dot.edge('route', 'expert_3')
    dot.edge('expert_0', 'aggregate')
    dot.edge('expert_1', 'aggregate')
    dot.edge('expert_2', 'aggregate')
    dot.edge('expert_3', 'aggregate')
    dot.edge('aggregate', 'ffn')
    dot.edge('ffn', 'output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../outputs/2025-11-26-14-20-42', exist_ok=True)
    
    # Generate complete DAG
    print("Generating complete MoE DAG...")
    complete_dag = generate_moe_dag()
    
    # Save DOT file
    complete_dag.save('../outputs/2025-11-26-14-20-42/complete_moe_dag.dot')
    
    # Render to SVG
    try:
        complete_dag.render('../outputs/2025-11-26-14-20-42/complete_moe_dag', format='svg', cleanup=True)
        print(f"Complete DAG saved to: ../outputs/2025-11-26-14-20-42/complete_moe_dag.svg")
    except Exception as e:
        print(f"Error rendering complete DAG: {e}")
    
    # Generate simplified DAG
    print("Generating simplified MoE DAG...")
    simplified_dag = generate_simplified_moe_dag()
    
    # Save DOT file
    simplified_dag.save('../outputs/2025-11-26-14-20-42/simplified_moe_dag.dot')
    
    # Render to SVG
    try:
        simplified_dag.render('../outputs/2025-11-26-14-20-42/simplified_moe_dag', format='svg', cleanup=True)
        print(f"Simplified DAG saved to: ../outputs/2025-11-26-14-20-42/simplified_moe_dag.svg")
    except Exception as e:
        print(f"Error rendering simplified DAG: {e}")
    
    print("DAG generation complete!")