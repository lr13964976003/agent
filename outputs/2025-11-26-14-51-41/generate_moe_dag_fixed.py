import graphviz
import os

# Create a directed graph for MoE deployment DAG
def create_moe_dag():
    dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism DAG')
    dot.attr(rankdir='TB', size='30,40', fontname='Arial')
    
    # Constants
    batch_size = 4
    seq_len = 2048
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    num_experts = 16
    top_k = 2
    
    # Input node
    dot.node('input', f'INPUT\\nGPU: N/A\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightcyan')
    
    # Layer norm 1
    dot.node('ln1', f'LayerNorm\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Multi-Head Attention components
    dot.node('q_proj', f'Q Projection\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('k_proj', f'K Projection\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('v_proj', f'V Projection\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Communication for attention
    dot.node('comm_q', f'All-Gather Q\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//8}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.node('comm_k', f'All-Gather K\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//8}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightyellow')
    dot.node('comm_v', f'All-Gather V\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads//8}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Attention computation
    dot.node('attn_score', f'Attention Score\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, seq_len={seq_len}, num_heads={num_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, seq_len={seq_len}, num_heads={num_heads}]',
             style='filled', fillcolor='lightblue')
    dot.node('attn_softmax', f'Softmax\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, seq_len={seq_len}, num_heads={num_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, seq_len={seq_len}, num_heads={num_heads}]',
             style='filled', fillcolor='lightblue')
    dot.node('attn_weight', f'Weighted Values\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, seq_len={seq_len}, num_heads={num_heads}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Output projection
    dot.node('o_proj', f'O Projection\\nGPU: 0-7\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Residual connection 1
    dot.node('res1', f'Residual Add 1\\nGPU: All\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # MoE routing
    dot.node('gate', f'Gate Network\\nGPU: Routing Node\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k={top_k}]', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Token routing based on expert assignment
    for i in range(num_experts):
        node_id = i // 4
        gpu_id = i % 4
        dot.node(f'split_{i}', f'Token Split\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, token_dim={token_dim}]', 
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Expert processing - each expert has full MLP
    for i in range(num_experts):
        node_id = i // 4
        gpu_id = i % 4
        
        # Expert gate linear
        dot.node(f'expert_gate_{i}', f'Expert {i} Gate\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, dynamic_seq_len, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]',
                 style='filled', fillcolor='lightblue')
        
        # Expert up linear
        dot.node(f'expert_up_{i}', f'Expert {i} Up\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, dynamic_seq_len, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]',
                 style='filled', fillcolor='lightblue')
        
        # Expert activation
        dot.node(f'expert_act_{i}', f'Expert {i} Activation\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]',
                 style='filled', fillcolor='lightblue')
        
        # Expert down linear
        dot.node(f'expert_down_{i}', f'Expert {i} Down\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, token_dim={token_dim}]',
                 style='filled', fillcolor='lightblue')
        
        # Expert gate multiply
        dot.node(f'expert_mul_{i}', f'Expert {i} Gate Mul\\nGPU: {node_id}_{gpu_id}\\nInput1: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]\\nInput2: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]\\nOutput: [batch_size={batch_size}, dynamic_seq_len, mlp_hidden={mlp_hidden}]',
                 style='filled', fillcolor='lightblue')
    
    # Token aggregation
    for i in range(num_experts):
        node_id = i // 4
        gpu_id = i % 4
        dot.node(f'agg_{i}', f'Token Aggregate\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, dynamic_seq_len, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Final aggregation
    dot.node('final_agg', f'Final Aggregation\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Residual connection 2
    dot.node('res2', f'Residual Add 2\\nGPU: All\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Output
    dot.node('output', f'OUTPUT\\nGPU: N/A\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightcyan')
    
    # Edges showing data flow
    dot.edge('input', 'ln1')
    dot.edge('ln1', 'q_proj')
    dot.edge('ln1', 'k_proj')
    dot.edge('ln1', 'v_proj')
    
    dot.edge('q_proj', 'comm_q')
    dot.edge('k_proj', 'comm_k')
    dot.edge('v_proj', 'comm_v')
    
    dot.edge('comm_q', 'attn_score')
    dot.edge('comm_k', 'attn_score')
    dot.edge('attn_score', 'attn_softmax')
    dot.edge('attn_softmax', 'attn_weight')
    dot.edge('comm_v', 'attn_weight')
    
    dot.edge('attn_weight', 'o_proj')
    dot.edge('o_proj', 'res1')
    dot.edge('input', 'res1')
    
    dot.edge('res1', 'gate')
    
    # Expert routing with dashed lines for gating decisions
    for i in range(num_experts):
        dot.edge('gate', f'split_{i}', style='dashed')
        dot.edge(f'split_{i}', f'expert_gate_{i}')
        dot.edge(f'split_{i}', f'expert_up_{i}')
        dot.edge(f'expert_gate_{i}', f'expert_act_{i}')
        dot.edge(f'expert_up_{i}', f'expert_mul_{i}')
        dot.edge(f'expert_act_{i}', f'expert_mul_{i}')
        dot.edge(f'expert_mul_{i}', f'expert_down_{i}')
        dot.edge(f'expert_down_{i}', f'agg_{i}')
        dot.edge(f'agg_{i}', 'final_agg')
    
    dot.edge('final_agg', 'res2')
    dot.edge('res1', 'res2')
    dot.edge('res2', 'output')
    
    return dot

# Create simplified layer DAG for better visualization
def create_single_layer_moe_dag():
    dot = graphviz.Digraph(comment='Single Layer MoE with Expert Parallelism')
    dot.attr(rankdir='TB', size='20,30', fontname='Arial')
    
    batch_size = 4
    seq_len = 2048
    token_dim = 7168
    num_heads = 128
    head_dim = 128
    mlp_hidden = 2048
    num_experts = 16
    
    # Layer inputs
    dot.node('layer_input', f'Layer Input\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]', 
             shape='ellipse', style='filled', fillcolor='lightcyan')
    
    # MHA Block
    dot.node('mha_ln', f'MHA LayerNorm\\nGPU: TP-8\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('mha_qkv', f'QKV Projection\\nGPU: TP-8\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('mha_attn', f'Multi-Head Attention\\nGPU: TP-8\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('mha_res', f'MHA Residual\\nGPU: All\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Expert routing
    dot.node('moe_ln', f'MoE LayerNorm\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    dot.node('gate', f'Gate Network\\nGPU: Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]',
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # Expert nodes - distributed across 4 nodes, 4 GPUs each
    expert_nodes = []
    for node_id in range(4):
        for gpu_id in range(4):
            expert_idx = node_id * 4 + gpu_id
            
            # Expert computation chain
            expert_gate = f'expert_{expert_idx}_gate'
            expert_up = f'expert_{expert_idx}_up'
            expert_act = f'expert_{expert_idx}_act'
            expert_down = f'expert_{expert_idx}_down'
            expert_mul = f'expert_{expert_idx}_mul'
            
            dot.node(expert_gate, f'Expert {expert_idx} Gate\\nGPU: {node_id}_{gpu_id}\\nInput: [dynamic_batch, dynamic_seq, token_dim={token_dim}]\\nOutput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]',
                     style='filled', fillcolor='lightblue')
            dot.node(expert_up, f'Expert {expert_idx} Up\\nGPU: {node_id}_{gpu_id}\\nInput: [dynamic_batch, dynamic_seq, token_dim={token_dim}]\\nOutput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]',
                     style='filled', fillcolor='lightblue')
            dot.node(expert_act, f'Expert {expert_idx} Activation\\nGPU: {node_id}_{gpu_id}\\nInput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]\\nOutput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]',
                     style='filled', fillcolor='lightblue')
            dot.node(expert_mul, f'Expert {expert_idx} Gate Mul\\nGPU: {node_id}_{gpu_id}\\nInput1: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]\\nInput2: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]\\nOutput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]',
                     style='filled', fillcolor='lightblue')
            dot.node(expert_down, f'Expert {expert_idx} Down\\nGPU: {node_id}_{gpu_id}\\nInput: [dynamic_batch, dynamic_seq, mlp_hidden={mlp_hidden}]\\nOutput: [dynamic_batch, dynamic_seq, token_dim={token_dim}]',
                     style='filled', fillcolor='lightblue')
            
            expert_nodes.append((expert_gate, expert_up, expert_act, expert_mul, expert_down))
    
    # Token routing nodes
    for i in range(num_experts):
        node_id = i // 4
        gpu_id = i % 4
        dot.node(f'split_{i}', f'Token Split {i}\\nGPU: {node_id}_{gpu_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [dynamic_batch, dynamic_seq, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
        dot.node(f'agg_{i}', f'Token Aggregate {i}\\nGPU: {node_id}_{gpu_id}\\nInput: [dynamic_batch, dynamic_seq, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # MoE aggregation
    dot.node('moe_agg', f'MoE Aggregation\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    dot.node('moe_res', f'MoE Residual\\nGPU: All\\nInput1: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nInput2: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             style='filled', fillcolor='lightblue')
    
    # Layer output
    dot.node('layer_output', f'Layer Output\\nGPU: All\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]',
             shape='ellipse', style='filled', fillcolor='lightcyan')
    
    # Edges
    dot.edge('layer_input', 'mha_ln')
    dot.edge('mha_ln', 'mha_qkv')
    dot.edge('mha_qkv', 'mha_attn')
    dot.edge('mha_attn', 'mha_res')
    dot.edge('layer_input', 'mha_res')
    
    dot.edge('mha_res', 'moe_ln')
    dot.edge('moe_ln', 'gate')
    
    # Expert routing
    for i in range(num_experts):
        gate_str = f'split_{i}'
        dot.edge('gate', gate_str, style='dashed')
        dot.edge(gate_str, expert_nodes[i][0])
        dot.edge(gate_str, expert_nodes[i][1])
        dot.edge(expert_nodes[i][0], expert_nodes[i][2])
        dot.edge(expert_nodes[i][1], expert_nodes[i][4])
        dot.edge(expert_nodes[i][2], expert_nodes[i][3])
        dot.edge(expert_nodes[i][4], expert_nodes[i][3])
        dot.edge(expert_nodes[i][3], f'agg_{i}')
        dot.edge(f'agg_{i}', 'moe_agg')
    
    dot.edge('moe_agg', 'moe_res')
    dot.edge('mha_res', 'moe_res')
    dot.edge('moe_res', 'layer_output')
    
    return dot

# Create communication DAG
def create_communication_dag():
    dot = graphviz.Digraph(comment='Cross-Node Communication Patterns')
    dot.attr(rankdir='LR', size='15,20', fontname='Arial')
    
    batch_size = 4
    seq_len = 2048
    token_dim = 7168
    
    # Node 0
    for gpu in range(4):
        dot.node(f'n0_gpu{gpu}', f'Node 0 GPU {gpu}\\nExpert {gpu}\\nMemory: 64GB\\nCompute: 400TFLOPS')
    
    # Node 1
    for gpu in range(4):
        dot.node(f'n1_gpu{gpu}', f'Node 1 GPU {gpu}\\nExpert {4+gpu}\\nMemory: 64GB\\nCompute: 400TFLOPS')
    
    # Node 2
    for gpu in range(4):
        dot.node(f'n2_gpu{gpu}', f'Node 2 GPU {gpu}\\nExpert {8+gpu}\\nMemory: 64GB\\nCompute: 400TFLOPS')
    
    # Node 3
    for gpu in range(4):
        dot.node(f'n3_gpu{gpu}', f'Node 3 GPU {gpu}\\nExpert {12+gpu}\\nMemory: 64GB\\nCompute: 400TFLOPS')
    
    # Routing node
    dot.node('router', f'Token Router\\nGPU: Routing Node\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: Distributed to experts', 
             shape='parallelogram', style='filled', fillcolor='lightgreen')
    
    # NVLink connections (intra-node)
    for node in range(4):
        for i in range(4):
            for j in range(i+1, 4):
                dot.edge(f'n{node}_gpu{i}', f'n{node}_gpu{j}', 
                        label='NVLink 900GB/s', 
                        color='blue', 
                        style='dashed')
    
    # InfiniBand connections (inter-node)
    for src_node in range(4):
        for dst_node in range(4):
            if src_node != dst_node:
                for gpu in range(4):
                    dot.edge(f'n{src_node}_gpu{gpu}', f'n{dst_node}_gpu{gpu}', 
                            label='InfiniBand 400Gbps', 
                            color='red', 
                            style='dotted')
    
    # Router to all experts
    for node in range(4):
        for gpu in range(4):
            dot.edge('router', f'n{node}_gpu{gpu}', style='dashed')
    
    return dot

if __name__ == "__main__":
    output_dir = "../outputs/2025-11-26-14-51-41"
    
    # Create complete MoE DAG
    dag1 = create_moe_dag()
    dag1.render(os.path.join(output_dir, 'complete_moe_dag'), format='svg', cleanup=False)
    dag1.save(os.path.join(output_dir, 'complete_moe_dag.dot'))
    
    # Create single layer MoE DAG
    dag2 = create_single_layer_moe_dag()
    dag2.render(os.path.join(output_dir, 'single_layer_moe_dag'), format='svg', cleanup=False)
    dag2.save(os.path.join(output_dir, 'single_layer_moe_dag.dot'))
    
    # Create communication DAG
    dag3 = create_communication_dag()
    dag3.render(os.path.join(output_dir, 'communication_pattern_dag'), format='svg', cleanup=False)
    dag3.save(os.path.join(output_dir, 'communication_pattern_dag.dot'))
    
    print(f"DAGs generated in {output_dir}")
    print("Files created:")
    print("- complete_moe_dag.svg")
    print("- single_layer_moe_dag.svg") 
    print("- communication_pattern_dag.svg")
    print("- complete_moe_dag.dot")
    print("- single_layer_moe_dag.dot")
    print("- communication_pattern_dag.dot")