#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def generate_single_layer_moe_dag():
    """Generate detailed DAG for a single MoE layer showing expert parallelism"""
    
    dot = Digraph(comment='Single MoE Layer - Large-Scale Cross-Node Expert Parallelism')
    dot.attr(rankdir='TB', size='25,25')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Model parameters
    batch_size = "batch_size"
    seq_len = "seq_len"
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    num_experts = 16
    hidden_size = 2048
    
    # Input to layer
    dot.node('layer_input', 
             f'Layer Input\\nGPU: Distributed\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # MHA computation
    dot.node('mha', 
             f'Multi-Head Attention\\nGPU: Replicated Across Nodes\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # First residual connection
    dot.node('residual1', 
             f'Residual Add 1\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Layer normalization
    dot.node('ln1', 
             f'Layer Norm 1\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Gating network
    dot.node('gate', 
             f'Gating Network (Top-2)\\nGPU: Replicated per node\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Token routing (communication step)
    dot.node('token_split', 
             f'Token Split by Expert\\nGPU: All GPUs\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: Expert-specific token subsets',
             shape='ellipse', fillcolor='lightblue', style='filled')
    
    # Cross-node communication
    dot.node('comm_node', 
             f'Cross-Node Token Transfer\\nGPU: NCCL Async\\nInput: Token subsets\\nOutput: Routed to destination GPUs',
             shape='ellipse', fillcolor='lightblue', style='filled')
    
    # Expert computation cluster
    with dot.subgraph(name='cluster_experts') as c:
        c.attr(label='Expert Parallelism - One Expert Per GPU', style='rounded,filled', fillcolor='lightgray')
        
        # Create 16 expert nodes (4x4 grid representation)
        for expert_id in range(num_experts):
            gpu_id = expert_id  # Each expert on dedicated GPU
            c.node(f'expert_{expert_id}', 
                   f'Expert {expert_id} MLP\\nGPU: {gpu_id}\\nInput: [batch_size_subset, token_dim={token_dim}]\\nOutput: [batch_size_subset, token_dim={token_dim}]\\nHidden: {hidden_size}',
                   shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Expert output aggregation
    dot.node('expert_aggregate', 
             f'Expert Output Aggregation\\nGPU: All GPUs\\nInput: [batch_size_subset, token_dim={token_dim}] x 16\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', fillcolor='lightblue', style='filled')
    
    # Weighted combination based on gate scores
    dot.node('weighted_combine', 
             f'Weighted Combination\\nGPU: Replicated\\nInput: Expert outputs + Gate weights\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # FFN layer after MoE
    dot.node('ffn', 
             f'Feed-Forward Network\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Second residual connection
    dot.node('residual2', 
             f'Residual Add 2\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}] x 2\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Layer normalization
    dot.node('ln2', 
             f'Layer Norm 2\\nGPU: Replicated\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Layer output
    dot.node('layer_output', 
             f'Layer Output\\nGPU: Distributed\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='rectangle', fillcolor='lightgreen', style='filled')
    
    # Connect all nodes
    dot.edge('layer_input', 'mha')
    dot.edge('mha', 'residual1')
    dot.edge('residual1', 'ln1')
    dot.edge('ln1', 'gate')
    dot.edge('gate', 'token_split')
    dot.edge('token_split', 'comm_node')
    
    # Connect communication to all experts
    for expert_id in range(num_experts):
        dot.edge('comm_node', f'expert_{expert_id}')
    
    # Connect all experts to aggregation
    for expert_id in range(num_experts):
        dot.edge(f'expert_{expert_id}', 'expert_aggregate')
    
    dot.edge('expert_aggregate', 'weighted_combine')
    dot.edge('gate', 'weighted_combine')  # Gate scores for weighting
    dot.edge('weighted_combine', 'ffn')
    dot.edge('ffn', 'residual2')
    dot.edge('residual2', 'ln2')
    dot.edge('ln2', 'layer_output')
    
    # Add dashed line for gate to expert selection
    for expert_id in range(num_experts):
        dot.edge('gate', f'expert_{expert_id}', style='dashed', 
                label='selects' if expert_id == 0 else '')
    
    return dot

def generate_performance_comparison_dag():
    """Generate comparison DAG between baseline and proposed approach"""
    
    dot = Digraph(comment='Performance Comparison: Baseline vs Proposed Expert Parallelism')
    dot.attr(rankdir='LR', size='20,15')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Baseline approach
    with dot.subgraph(name='cluster_baseline') as c:
        c.attr(label='Baseline: Multiple Experts per GPU', style='rounded,filled', fillcolor='lightcoral')
        
        c.node('baseline_input', 'Input Tokens', shape='rectangle', fillcolor='lightgreen')
        c.node('baseline_route', 'Token Routing\n(Within Node)', shape='ellipse', fillcolor='lightblue')
        c.node('baseline_gpu0', 'GPU 0\n4 Experts\nContention', shape='rectangle', fillcolor='orange')
        c.node('baseline_gpu1', 'GPU 1\n4 Experts\nContention', shape='rectangle', fillcolor='orange')
        c.node('baseline_gpu2', 'GPU 2\n4 Experts\nContention', shape='rectangle', fillcolor='orange')
        c.node('baseline_gpu3', 'GPU 3\n4 Experts\nContention', shape='rectangle', fillcolor='orange')
        c.node('baseline_aggregate', 'Aggregation\n(Within Node)', shape='ellipse', fillcolor='lightblue')
        c.node('baseline_output', 'Output\nLower Throughput', shape='rectangle', fillcolor='lightgreen')
        
        c.edge('baseline_input', 'baseline_route')
        c.edge('baseline_route', 'baseline_gpu0')
        c.edge('baseline_route', 'baseline_gpu1')
        c.edge('baseline_route', 'baseline_gpu2')
        c.edge('baseline_route', 'baseline_gpu3')
        c.edge('baseline_gpu0', 'baseline_aggregate')
        c.edge('baseline_gpu1', 'baseline_aggregate')
        c.edge('baseline_gpu2', 'baseline_aggregate')
        c.edge('baseline_gpu3', 'baseline_aggregate')
        c.edge('baseline_aggregate', 'baseline_output')
    
    # Proposed approach
    with dot.subgraph(name='cluster_proposed') as c:
        c.attr(label='Proposed: One Expert per GPU', style='rounded,filled', fillcolor='lightgreen')
        
        c.node('proposed_input', 'Input Tokens', shape='rectangle', fillcolor='lightgreen')
        c.node('proposed_route', 'Token Routing\n(Cross-Node)', shape='ellipse', fillcolor='lightblue')
        c.node('proposed_gpu0', 'GPU 0\nExpert 0\nNo Contention', shape='rectangle', fillcolor='lightgreen')
        c.node('proposed_gpu1', 'GPU 1\nExpert 1\nNo Contention', shape='rectangle', fillcolor='lightgreen')
        c.node('proposed_gpu2', 'GPU 2\nExpert 2\nNo Contention', shape='rectangle', fillcolor='lightgreen')
        c.node('proposed_gpu15', 'GPU 15\nExpert 15\nNo Contention', shape='rectangle', fillcolor='lightgreen')
        c.node('proposed_aggregate', 'Aggregation\n(Cross-Node)', shape='ellipse', fillcolor='lightblue')
        c.node('proposed_output', 'Output\nHigher Throughput', shape='rectangle', fillcolor='lightgreen')
        
        c.edge('proposed_input', 'proposed_route')
        c.edge('proposed_route', 'proposed_gpu0')
        c.edge('proposed_route', 'proposed_gpu1')
        c.edge('proposed_route', 'proposed_gpu2')
        c.edge('proposed_route', 'proposed_gpu15')
        c.edge('proposed_gpu0', 'proposed_aggregate')
        c.edge('proposed_gpu1', 'proposed_aggregate')
        c.edge('proposed_gpu2', 'proposed_aggregate')
        c.edge('proposed_gpu15', 'proposed_aggregate')
        c.edge('proposed_aggregate', 'proposed_output')
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../outputs/2025-11-26-14-20-42', exist_ok=True)
    
    # Generate single layer DAG
    print("Generating single layer MoE DAG...")
    single_layer_dag = generate_single_layer_moe_dag()
    
    # Save DOT file
    single_layer_dag.save('../outputs/2025-11-26-14-20-42/single_layer_moe_dag.dot')
    
    # Render to SVG
    try:
        single_layer_dag.render('../outputs/2025-11-26-14-20-42/single_layer_moe_dag', format='svg', cleanup=True)
        print(f"Single layer DAG saved to: ../outputs/2025-11-26-14-20-42/single_layer_moe_dag.svg")
    except Exception as e:
        print(f"Error rendering single layer DAG: {e}")
    
    # Generate comparison DAG
    print("Generating performance comparison DAG...")
    comparison_dag = generate_performance_comparison_dag()
    
    # Save DOT file
    comparison_dag.save('../outputs/2025-11-26-14-20-42/performance_comparison_dag.dot')
    
    # Render to SVG
    try:
        comparison_dag.render('../outputs/2025-11-26-14-20-42/performance_comparison_dag', format='svg', cleanup=True)
        print(f"Performance comparison DAG saved to: ../outputs/2025-11-26-14-20-42/performance_comparison_dag.svg")
    except Exception as e:
        print(f"Error rendering comparison DAG: {e}")
    
    print("Additional DAG generation complete!")