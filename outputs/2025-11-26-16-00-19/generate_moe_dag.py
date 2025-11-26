#!/usr/bin/env python3
"""
Generate complete DAG for large-scale cross-node expert parallelism
Each node represents a GPU with detailed operator-level decomposition
"""

import graphviz

def create_moe_dag():
    """Create a complete DAG for one MoE layer with large-scale expert parallelism"""
    
    # Initialize directed graph
    dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
    dot.attr(rankdir='TB', size='100,100', splines='ortho')
    dot.attr('node', fontsize='10', height='0.8', width='2.5')
    
    # Define colors for different node types
    dot.attr('node', fillcolor='lightblue', style='filled')  # Computation nodes
    dot.attr('node', fillcolor='lightgreen', style='filled')  # Communication nodes
    dot.attr('node', fillcolor='lightyellow', style='filled')  # Routing nodes
    dot.attr('node', fillcolor='lightcoral', style='filled')  # Aggregation nodes
    
    # Global parameters
    batch_size = 32
    seq_len = 2048
    hidden_dim = 7168
    num_heads = 128
    head_dim = 128
    num_experts = 32
    expert_hidden = 2048
    top_k = 2
    
    # ============================= INPUT NODE =============================
    dot.node('input', 
             f'Input Layer\nGPU: ALL\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]',
             fillcolor='lightgray', shape='box')
    
    # ============================= MHA DECOMPOSITION =============================
    
    # MHA: Query projection (Q)
    dot.node('q_proj', 
             f'Q Projection\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='box')
    
    # MHA: Key projection (K)
    dot.node('k_proj', 
             f'K Projection\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='box')
    
    # MHA: Value projection (V)
    dot.node('v_proj', 
             f'V Projection\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]',
             shape='box')
    
    # MHA: Attention computation
    dot.node('attention', 
             f'Multi-Head Attention\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]',
             shape='box')
    
    # MHA: Output projection
    dot.node('mha_out_proj', 
             f'MHA Output Projection\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]',
             shape='box')
    
    # ============================= GATING NETWORK =============================
    
    # Gating network
    dot.node('gating', 
             f'Gating Network\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, experts={num_experts}, top_k={top_k}]',
             shape='parallelogram')
    
    # Token routing decision
    dot.node('routing', 
             f'Token Routing Decision\nGPU: Shared (All GPUs)\nInput: [batch_size={batch_size}, seq_len={seq_len}, experts={num_experts}, top_k={top_k}]\nOutput: Token routing masks',
             shape='parallelogram')
    
    # ============================= TOKEN DISTRIBUTION =============================
    
    # Communication: Token scatter to experts
    dot.node('token_scatter', 
             f'Token Scatter Communication\nGPU: All-to-all\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: Distributed tokens per expert',
             shape='ellipse', fillcolor='lightgreen')
    
    # ============================= EXPERT COMPUTATION =============================
    
    expert_nodes = []
    for expert_id in range(16):  # Show first 16 experts for clarity
        
        # Expert gate computation
        gate_node = f'expert_{expert_id}_gate'
        dot.node(gate_node, 
                 f'Expert {expert_id} Gate\nGPU: {expert_id}\nInput: [batch_size=?, seq_len=?, hidden={hidden_dim}]\nOutput: [batch_size=?, seq_len=?, expert_hidden={expert_hidden}]',
                 shape='box')
        
        # Expert expert computation
        expert_node = f'expert_{expert_id}_expert'
        dot.node(expert_node, 
                 f'Expert {expert_id} Expert\nGPU: {expert_id}\nInput: [batch_size=?, seq_len=?, expert_hidden={expert_hidden}]\nOutput: [batch_size=?, seq_len=?, hidden={hidden_dim}]',
                 shape='box')
        
        # Expert multiply-add (SiLU * gate)
        multiply_node = f'expert_{expert_id}_multiply'
        dot.node(multiply_node, 
                 f'Expert {expert_id} Multiply\nGPU: {expert_id}\nInput: [batch_size=?, seq_len=?, hidden={hidden_dim}], [batch_size=?, seq_len=?, expert_hidden={expert_hidden}]\nOutput: [batch_size=?, seq_len=?, hidden={hidden_dim}]',
                 shape='box')
        
        expert_nodes.append((gate_node, expert_node, multiply_node))
    
    # ============================= AGGREGATION =============================
    
    # Communication: Token gather from experts
    dot.node('token_gather', 
             f'Token Gather Communication\nGPU: All-to-all\nInput: Expert outputs\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]',
             shape='ellipse', fillcolor='lightcoral')
    
    # ============================= RESIDUAL CONNECTION =============================
    
    # Final layer output
    dot.node('layer_output', 
             f'Layer Output\nGPU: ALL\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]',
             fillcolor='lightgray', shape='box')
    
    # ============================= EDGES =============================
    
    # Input to MHA projections
    dot.edge('input', 'q_proj')
    dot.edge('input', 'k_proj')
    dot.edge('input', 'v_proj')
    
    # MHA projections to attention
    dot.edge('q_proj', 'attention')
    dot.edge('k_proj', 'attention')
    dot.edge('v_proj', 'attention')
    
    # Attention to output and gating
    dot.edge('attention', 'mha_out_proj')
    dot.edge('attention', 'gating')
    
    # Gating to routing
    dot.edge('gating', 'routing')
    dot.edge('routing', 'token_scatter')
    
    # Token scatter to all experts
    for expert_id in range(16):
        gate_node, expert_node, multiply_node = expert_nodes[expert_id]
        dot.edge('token_scatter', gate_node)
        dot.edge(gate_node, expert_node)
        dot.edge(expert_node, multiply_node)
        dot.edge(multiply_node, 'token_gather')
        
        # Dashed line for gating decision (token routing)
        dot.edge('routing', gate_node, style='dashed')
    
    # Expert outputs to gather
    for expert_id in range(16):
        _, _, multiply_node = expert_nodes[expert_id]
        dot.edge(multiply_node, 'token_gather')
    
    # Gather to output
    dot.edge('token_gather', 'layer_output')
    dot.edge('mha_out_proj', 'layer_output')
    
    return dot

def create_detailed_expert_dag():
    """Create a more detailed DAG showing all 32 experts"""
    
    dot = graphviz.Digraph(comment='Complete 32-Expert Large-Scale Parallelism')
    dot.attr(rankdir='LR', size='150,50')
    
    # Global parameters
    batch_size = 32
    seq_len = 2048
    hidden_dim = 7168
    expert_hidden = 2048
    
    # Input node
    dot.node('input', f'Input\n[batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]', 
             shape='box', fillcolor='lightgray')
    
    # Attention computation (shared across all GPUs)
    dot.node('mha', f'Multi-Head Attention\nGPU: Shared\n[batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]',
             shape='box', fillcolor='lightblue')
    
    # Gating network
    dot.node('gate', f'Gating Network\nGPU: Shared\n[batch={batch_size}, seq={seq_len}, experts=32]',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Token routing
    dot.node('route', f'Token Routing\nGPU: Shared',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Communication - scatter
    dot.node('scatter', f'Token Scatter\nAll-to-all Comm',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create 32 expert clusters
    expert_cluster_nodes = []
    for gpu_id in range(32):
        cluster = f'cluster_gpu_{gpu_id}'
        
        with dot.subgraph(name=cluster) as c:
            c.attr(label=f'GPU {gpu_id}', style='dotted')
            
            # Expert gate
            gate = f'expert_gate_{gpu_id}'
            c.node(gate, f'Expert Gate\n[batch=?, seq=?, hidden={expert_hidden}]',
                   shape='box', fillcolor='lightblue')
            
            # Expert expert
            expert = f'expert_expert_{gpu_id}'
            c.node(expert, f'Expert Expert\n[batch=?, seq=?, hidden={hidden_dim}]',
                   shape='box', fillcolor='lightblue')
            
            # Multiply-add
            multiply = f'expert_multiply_{gpu_id}'
            c.node(multiply, f'Multiply\n[batch=?, seq=?, hidden={hidden_dim}]',
                   shape='box', fillcolor='lightblue')
            
            expert_cluster_nodes.append((gate, expert, multiply))
    
    # Communication - gather
    dot.node('gather', f'Token Gather\nAll-to-all Comm',
             shape='ellipse', fillcolor='lightcoral')
    
    # Output
    dot.node('output', f'Layer Output\n[batch={batch_size}, seq={seq_len}, hidden={hidden_dim}]',
             shape='box', fillcolor='lightgray')
    
    # Connect everything
    dot.edge('input', 'mha')
    dot.edge('input', 'gate')
    dot.edge('gate', 'route')
    dot.edge('route', 'scatter')
    dot.edge('mha', 'gather')  # Skip connection
    
    for gpu_id in range(32):
        gate, expert, multiply = expert_cluster_nodes[gpu_id]
        dot.edge('scatter', gate)
        dot.edge(gate, expert)
        dot.edge(expert, multiply)
        dot.edge(multiply, 'gather')
        dot.edge('route', gate, style='dashed', constraint='false')
    
    dot.edge('gather', 'output')
    
    return dot

if __name__ == "__main__":
    # Generate both DAGs
    simple_dag = create_moe_dag()
    detailed_dag = create_detailed_expert_dag()
    
    # Save DOT files
    simple_dag.save('../outputs/2025-11-26-16-00-19/moe_parallelism_dag.dot')
    detailed_dag.save('../outputs/2025-11-26-16-00-19/moe_detailed_32_experts.dot')
    
    # Save SVG files
    simple_dag.render('../outputs/2025-11-26-16-00-19/moe_parallelism_dag', format='svg', cleanup=True)
    detailed_dag.render('../outputs/2025-11-26-16-00-19/moe_detailed_32_experts', format='svg', cleanup=True)
    
    print("DAGs generated successfully:")
    print("- Simple DAG: ../outputs/2025-11-26-16-00-19/moe_parallelism_dag.dot")
    print("- Detailed DAG: ../outputs/2025-11-26-16-00-19/moe_detailed_32_experts.dot")
    print("- Simple SVG: ../outputs/2025-11-26-16-00-19/moe_parallelism_dag.svg")
    print("- Detailed SVG: ../outputs/2025-11-26-16-00-19/moe_detailed_32_experts.svg")