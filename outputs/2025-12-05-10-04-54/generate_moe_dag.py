#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_moe_dag():
    # Create a new directed graph
    dot = Digraph(comment='MoE Hybrid Parallel DAG')
    dot.attr(rankdir='TB', size='100,200', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Configuration parameters
    total_gpus = 128
    ep_degree = 8  # Expert parallelism degree
    tp_degree = 2  # Tensor parallelism degree
    pp_degree = 2  # Pipeline parallelism degree
    dp_degree = 4  # Data parallelism degree
    layers = 16
    experts_per_layer = 64
    experts_per_gpu = 8
    attention_heads = 16
    head_dim = 64
    token_dim = 1024
    moe_hidden_size = 2048
    batch_size = 128
    micro_batch_size = 32
    seq_len = 1024  # Using a representative sequence length
    
    # GPU organization: EP(8) × TP(2) × PP(2) × DP(4) = 128 GPUs
    # Let's represent a simplified but complete view focusing on one data parallel group
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch={batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={token_dim}]', 
             shape='octagon', fillcolor='lightcoral')
    
    # Let's create a representative slice showing the full pipeline for one DP group
    # We'll show 4 layers (2 from each stage) to demonstrate the pattern
    
    current_node = 'input'
    
    for stage in range(pp_degree):
        for layer in range(2):  # Show 2 layers per stage for clarity
            layer_id = stage * 8 + layer  # Actual layer ID
            
            # Layer input distribution (communication)
            dist_node = f'dist_stage{stage}_layer{layer}'
            dot.node(dist_node, f'Distribute Layer {layer_id}\\nInput: [batch={batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                     shape='ellipse', fillcolor='lightblue')
            dot.edge(current_node, dist_node)
            
            # Attention computation - broken down into detailed steps
            # Step 1: QKV projection with tensor parallelism
            for tp_id in range(tp_degree):
                qkv_node = f'qkv_stage{stage}_layer{layer}_tp{tp_id}'
                gpu_id = f'GPU-{stage*64 + tp_id}'  # Simplified GPU mapping
                dot.node(qkv_node, f'QKV Projection TP{tp_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, heads={attention_heads//2}, d_k={head_dim}]', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(dist_node, qkv_node)
            
            # Step 2: Attention score computation
            for tp_id in range(tp_degree):
                score_node = f'score_stage{stage}_layer{layer}_tp{tp_id}'
                gpu_id = f'GPU-{stage*64 + tp_id}'
                dot.node(score_node, f'Attention Scores TP{tp_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size}, seq={seq_len}, heads={attention_heads//2}, d_k={head_dim}]\\nOutput: [batch={micro_batch_size}, heads={attention_heads//2}, seq={seq_len}, seq={seq_len}]', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(f'qkv_stage{stage}_layer{layer}_tp{tp_id}', score_node)
            
            # Step 3: Attention weights (softmax)
            for tp_id in range(tp_degree):
                softmax_node = f'softmax_stage{stage}_layer{layer}_tp{tp_id}'
                gpu_id = f'GPU-{stage*64 + tp_id}'
                dot.node(softmax_node, f'Attention Softmax TP{tp_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size}, heads={attention_heads//2}, seq={seq_len}, seq={seq_len}]\\nOutput: [batch={micro_batch_size}, heads={attention_heads//2}, seq={seq_len}, seq={seq_len}]', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(score_node, softmax_node)
            
            # Step 4: Attention output (weighted sum)
            for tp_id in range(tp_degree):
                attn_out_node = f'attn_out_stage{stage}_layer{layer}_tp{tp_id}'
                gpu_id = f'GPU-{stage*64 + tp_id}'
                dot.node(attn_out_node, f'Attention Output TP{tp_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size}, heads={attention_heads//2}, seq={seq_len}, d_k={head_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim//2}]', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(f'softmax_stage{stage}_layer{layer}_tp{tp_id}', attn_out_node)
                dot.edge(f'qkv_stage{stage}_layer{layer}_tp{tp_id}', attn_out_node)
            
            # Step 5: Attention output projection and aggregation
            attn_agg_node = f'attn_agg_stage{stage}_layer{layer}'
            dot.node(attn_agg_node, f'Attention Aggregation\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim//2}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                     shape='parallelogram', fillcolor='lightyellow')
            
            for tp_id in range(tp_degree):
                dot.edge(f'attn_out_stage{stage}_layer{layer}_tp{tp_id}', attn_agg_node)
            
            # MoE Routing (Gate)
            gate_node = f'gate_stage{stage}_layer{layer}'
            dot.node(gate_node, f'MoE Gate (Top-2)\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, experts=2]', 
                     shape='parallelogram', fillcolor='lightyellow', style='dashed')
            dot.edge(attn_agg_node, gate_node, style='dashed')
            
            # MoE Experts (8 experts per GPU, distributed across EP groups)
            for ep_id in range(ep_degree):
                for expert_id in range(experts_per_gpu):
                    expert_node = f'expert_stage{stage}_layer{layer}_ep{ep_id}_ex{expert_id}'
                    gpu_id = f'GPU-{stage*64 + ep_id*8 + expert_id}'  # Simplified mapping
                    dot.node(expert_node, f'Expert {expert_id} EP{ep_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size//8}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size//8}, seq={seq_len}, hidden={token_dim}]', 
                             shape='rectangle', fillcolor='lightgreen')
                    
                    # Connect gate to experts (dashed for routing)
                    dot.edge(gate_node, expert_node, style='dashed')
                    # Connect attention to experts (solid line)
                    dot.edge(attn_agg_node, expert_node)
            
            # MoE Expert Aggregation
            expert_agg_node = f'expert_agg_stage{stage}_layer{layer}'
            dot.node(expert_agg_node, f'Expert Aggregation\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                     shape='parallelogram', fillcolor='lightyellow')
            
            for ep_id in range(ep_degree):
                for expert_id in range(experts_per_gpu):
                    expert_node = f'expert_stage{stage}_layer{layer}_ep{ep_id}_ex{expert_id}'
                    dot.edge(expert_node, expert_agg_node)
            
            # MLP Layer (after MoE)
            for tp_id in range(tp_degree):
                mlp_node = f'mlp_stage{stage}_layer{layer}_tp{tp_id}'
                gpu_id = f'GPU-{stage*64 + tp_id}'
                dot.node(mlp_node, f'MLP Layer TP{tp_id}\\n{gpu_id}\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                         shape='rectangle', fillcolor='lightgreen')
                dot.edge(expert_agg_node, mlp_node)
            
            # MLP Aggregation
            mlp_agg_node = f'mlp_agg_stage{stage}_layer{layer}'
            dot.node(mlp_agg_node, f'MLP Aggregation\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim//2}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                     shape='parallelogram', fillcolor='lightyellow')
            
            for tp_id in range(tp_degree):
                dot.edge(f'mlp_stage{stage}_layer{layer}_tp{tp_id}', mlp_agg_node)
            
            # Layer normalization
            layernorm_node = f'layernorm_stage{stage}_layer{layer}'
            dot.node(layernorm_node, f'LayerNorm {layer_id}\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                     shape='rectangle', fillcolor='lightgreen')
            dot.edge(mlp_agg_node, layernorm_node)
            
            current_node = layernorm_node
            
            # Add communication between stages
            if stage < pp_degree - 1 and layer == 1:  # Between stages
                comm_node = f'comm_stage{stage}_to_{stage+1}'
                dot.node(comm_node, f'Pipeline Communication\\nStage {stage} → {stage+1}\\nInput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={micro_batch_size}, seq={seq_len}, hidden={token_dim}]', 
                         shape='ellipse', fillcolor='lightblue')
                dot.edge(current_node, comm_node)
                current_node = comm_node
    
    # Final output
    dot.node('output', f'Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={token_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={token_dim}]', 
             shape='doubleoctagon', fillcolor='lightcoral')
    dot.edge(current_node, 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file = '../outputs/2025-12-05-10-04-54/moe_hybrid_parallel_dag.dot'
    dag.save(dot_file)
    
    # Save as SVG
    svg_file = '../outputs/2025-12-05-10-04-54/moe_hybrid_parallel_dag.svg'
    dag.render(format='svg', filename='../outputs/2025-12-05-10-04-54/moe_hybrid_parallel_dag', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")