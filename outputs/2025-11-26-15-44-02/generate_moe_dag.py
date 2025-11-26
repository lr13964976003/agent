#!/usr/bin/env python3
"""
Generate complete DAG for Large-Scale Cross-Node Expert Parallelism MoE Model
Based on the paper with 61 layers (3 dense + 58 MoE), single expert per GPU
"""

import os
from graphviz import Digraph

def create_moe_dag():
    """Create comprehensive DAG for MoE model with expert parallelism"""
    
    # Create DAG
    dot = Digraph(comment='Large-Scale Cross-Node Expert Parallelism MoE Model')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    
    # Define node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Model configuration from paper
    batch_size = "B"
    seq_len = "S"
    hidden_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    num_experts = 16  # EP >= 16 as per paper
    num_layers = 61
    dense_layers = 3
    moe_layers = 58
    
    # Define colors
    compute_color = 'lightblue'
    comm_color = 'lightgreen'
    routing_color = 'lightyellow'
    aggregate_color = 'lightcoral'
    
    # Input node
    dot.node('input', 
             f'INPUT\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', style='filled', fillcolor=comm_color)
    
    prev_node = 'input'
    
    # Process each layer
    for layer_idx in range(num_layers):
        layer_type = 'dense' if layer_idx < dense_layers else 'moe'
        
        if layer_type == 'dense':
            # Dense layer (standard transformer)
            layer_prefix = f'layer{layer_idx}_dense'
            
            # Multi-Head Attention
            mha_node = f'{layer_prefix}_mha'
            dot.node(mha_node,
                     f'MHA L{layer_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # MLP for dense layer
            mlp_node = f'{layer_prefix}_mlp'
            dot.node(mlp_node,
                     f'MLP L{layer_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Connect within layer
            dot.edge(prev_node, mha_node)
            dot.edge(mha_node, mlp_node)
            prev_node = mlp_node
            
        else:  # MoE layer
            layer_prefix = f'layer{layer_idx}_moe'
            
            # Multi-Head Attention (same as dense)
            mha_node = f'{layer_prefix}_mha'
            dot.node(mha_node,
                     f'MHA L{layer_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='rectangle', style='filled', fillcolor=compute_color)
            
            # Gate computation (routing decision)
            gate_node = f'{layer_prefix}_gate'
            dot.node(gate_node,
                     f'GATE L{layer_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                     shape='parallelogram', style='filled', fillcolor=routing_color)
            
            # Token splitting based on routing
            split_node = f'{layer_prefix}_split'
            dot.node(split_node,
                     f'SPLIT TOKENS L{layer_idx}\\nGPU: 0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: Multiple expert-specific batches',
                     shape='parallelogram', style='filled', fillcolor=aggregate_color)
            
            # Expert computations (one per GPU)
            expert_nodes = []
            for expert_id in range(num_experts):
                expert_node = f'{layer_prefix}_expert{expert_id}'
                gpu_id = expert_id % 16  # Distribute across 16 GPUs
                
                dot.node(expert_node,
                         f'EXPERT {expert_id} L{layer_idx}\\nGPU: {gpu_id}\\nInput: [batch_size=B_expert{expert_id}, seq_len=S_expert{expert_id}, hidden_dim={hidden_dim}]\\nOutput: [batch_size=B_expert{expert_id}, seq_len=S_expert{expert_id}, hidden_dim={hidden_dim}]',
                         shape='rectangle', style='filled', fillcolor=compute_color)
                expert_nodes.append(expert_node)
                
                # Communication from split to expert
                comm_node = f'{layer_prefix}_comm_to_expert{expert_id}'
                dot.node(comm_node,
                         f'SEND TO EXPERT {expert_id}\\nGPU: {gpu_id}\\nTokens routed to expert {expert_id}',
                         shape='ellipse', style='filled,dashed', fillcolor=comm_color)
                dot.edge(split_node, comm_node, style='dashed')
                dot.edge(comm_node, expert_node)
            
            # Expert aggregation
            aggregate_node = f'{layer_prefix}_aggregate'
            dot.node(aggregate_node,
                     f'AGGREGATE EXPERTS L{layer_idx}\\nGPU: 0\\nInput: Multiple expert outputs\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
                     shape='parallelogram', style='filled', fillcolor=aggregate_color)
            
            # Communications from experts to aggregation
            for expert_id, expert_node in enumerate(expert_nodes):
                comm_back_node = f'{layer_prefix}_comm_from_expert{expert_id}'
                dot.node(comm_back_node,
                         f'RECV FROM EXPERT {expert_id}\\nGPU: 0\\nOutput from expert {expert_id}',
                         shape='ellipse', style='filled', fillcolor=comm_color)
                dot.edge(expert_node, comm_back_node)
                dot.edge(comm_back_node, aggregate_node)
            
            # Connect within MoE layer
            dot.edge(prev_node, mha_node)
            dot.edge(mha_node, gate_node)
            dot.edge(gate_node, split_node)
            prev_node = aggregate_node
    
    # Final output
    dot.node('output',
             f'OUTPUT\\nGPU: Host\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', style='filled', fillcolor=comm_color)
    
    dot.edge(prev_node, 'output')
    
    return dot

def create_simplified_moe_dag():
    """Create simplified DAG showing key concepts"""
    
    dot = Digraph(comment='Simplified MoE Expert Parallelism')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    batch_size = "B"
    seq_len = "S"
    hidden_dim = 7168
    num_experts = 16
    
    # Colors
    compute_color = 'lightblue'
    comm_color = 'lightgreen'
    routing_color = 'lightyellow'
    aggregate_color = 'lightcoral'
    
    # Input
    dot.node('input', f'INPUT\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', style='filled', fillcolor=comm_color)
    
    # MHA
    dot.node('mha', f'Multi-Head Attention\\nGPU: 0\\n[batch_size={batch_size}, seq_len={seq_len}, heads=128, d_k=128]',
             shape='rectangle', style='filled', fillcolor=compute_color)
    
    # Gate
    dot.node('gate', f'Gate (Router)\\nGPU: 0\\n[batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
             shape='parallelogram', style='filled', fillcolor=routing_color)
    
    # Token split
    dot.node('split', f'Token Split\\nGPU: 0\\nRoute tokens to experts',
             shape='parallelogram', style='filled', fillcolor=aggregate_color)
    
    # Experts (showing first few)
    experts = []
    for i in range(4):  # Show first 4 experts
        expert = f'expert{i}'
        gpu_id = i
        dot.node(expert, f'Expert {i}\\nGPU: {gpu_id}\\nMLP Expert Computation',
                 shape='rectangle', style='filled', fillcolor=compute_color)
        experts.append(expert)
        
        # Communication to expert
        comm_to = f'comm_to{i}'
        dot.node(comm_to, f'Send Tokens\\nGPU: {gpu_id}',
                 shape='ellipse', style='filled,dashed', fillcolor=comm_color)
        dot.edge('split', comm_to, style='dashed')
        dot.edge(comm_to, expert)
    
    # Show ellipsis for remaining experts
    dot.node('ellipsis1', '...\\n12 more experts\\nGPUs 4-15',
             shape='ellipse', style='dashed', fillcolor='white')
    
    # Aggregation
    dot.node('aggregate', f'Aggregate Experts\\nGPU: 0\\nCombine expert outputs',
             shape='parallelogram', style='filled', fillcolor=aggregate_color)
    
    # Communications from experts
    for i, expert in enumerate(experts):
        comm_from = f'comm_from{i}'
        dot.node(comm_from, f'Receive Output\\nGPU: 0',
                 shape='ellipse', style='filled', fillcolor=comm_color)
        dot.edge(expert, comm_from)
        dot.edge(comm_from, 'aggregate')
    
    # Output
    dot.node('output', f'OUTPUT\\n[batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]',
             shape='ellipse', style='filled', fillcolor=comm_color)
    
    # Connect main flow
    dot.edge('input', 'mha')
    dot.edge('mha', 'gate')
    dot.edge('gate', 'split')
    dot.edge('aggregate', 'output')
    
    return dot

def create_parallel_strategies_dag():
    """Create DAG showing all parallel strategies integration"""
    
    dot = Digraph(comment='MoE Parallel Strategies Integration')
    dot.attr(rankdir='LR', splines='ortho', bgcolor='white')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    batch_size = "B"
    seq_len = "S"
    hidden_dim = 7168
    dp_degree = 2
    ep_degree = 16
    tp_degree = 2
    
    # Colors
    dp_color = 'lightblue'
    ep_color = 'lightgreen'
    tp_color = 'lightyellow'
    comm_color = 'lightcoral'
    
    # Data Parallelism replicas
    for dp_rank in range(dp_degree):
        with dot.subgraph(name=f'cluster_dp{dp_rank}') as c:
            c.attr(label=f'Data Parallel Replica {dp_rank}', style='rounded', fillcolor=dp_color)
            
            # Within each DP replica, show EP and TP
            # Expert Parallelism across GPUs
            for ep_rank in range(ep_degree):
                gpu_id = dp_rank * ep_degree + ep_rank
                
                # Tensor Parallelism within expert if needed
                if tp_degree > 1:
                    # Show TP split within expert
                    for tp_rank in range(tp_degree):
                        node_id = f'dp{dp_rank}_ep{ep_rank}_tp{tp_rank}'
                        c.node(node_id,
                               f'Expert {ep_rank} TP{tp_rank}\\nGPU: {gpu_id}\\nTP Dimension: {hidden_dim//tp_degree}',
                               shape='rectangle', style='filled', fillcolor=tp_color)
                else:
                    node_id = f'dp{dp_rank}_ep{ep_rank}'
                    c.node(node_id,
                           f'Expert {ep_rank}\\nGPU: {gpu_id}\\nFull Expert',
                           shape='rectangle', style='filled', fillcolor=ep_color)
    
    # Communication nodes between DP replicas
    for dp_rank in range(dp_degree):
        for other_dp in range(dp_degree):
            if dp_rank != other_dp:
                comm_node = f'dp_comm_{dp_rank}_to_{other_dp}'
                dot.node(comm_node,
                         f'DP Sync\\nReplica {dp_rank} ↔ {other_dp}',
                         shape='ellipse', style='filled', fillcolor=comm_color)
    
    return dot

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../outputs/2025-11-26-15-44-02', exist_ok=True)
    
    # Generate comprehensive DAG
    print("Generating comprehensive MoE DAG...")
    comprehensive_dag = create_moe_dag()
    comprehensive_dag.render('../outputs/2025-11-26-15-44-02/moe_comprehensive_dag', format='svg', cleanup=True)
    comprehensive_dag.render('../outputs/2025-11-26-15-44-02/moe_comprehensive_dag', format='dot', cleanup=True)
    
    # Generate simplified DAG
    print("Generating simplified MoE DAG...")
    simplified_dag = create_simplified_moe_dag()
    simplified_dag.render('../outputs/2025-11-26-15-44-02/moe_simplified_dag', format='svg', cleanup=True)
    simplified_dag.render('../outputs/2025-11-26-15-44-02/moe_simplified_dag', format='dot', cleanup=True)
    
    # Generate parallel strategies DAG
    print("Generating parallel strategies DAG...")
    parallel_dag = create_parallel_strategies_dag()
    parallel_dag.render('../outputs/2025-11-26-15-44-02/moe_parallel_strategies', format='svg', cleanup=True)
    parallel_dag.render('../outputs/2025-11-26-15-44-02/moe_parallel_strategies', format='dot', cleanup=True)
    
    print("All DAGs generated successfully!")
    
    # List generated files
    import glob
    files = glob.glob('../outputs/2025-11-26-15-44-02/*.svg') + glob.glob('../outputs/2025-11-26-15-44-02/*.dot')
    print("Generated files:")
    for f in files:
        print(f"  {f}")