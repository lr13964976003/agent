#!/usr/bin/env python3
"""
MA Separation Model DAG Generator
12 Attention GPUs (0-11) + 4 MoE GPUs (12-15)
"""

import graphviz

def create_ma_separation_dag():
    dot = graphviz.Digraph('ma_separation')
    dot.attr(rankdir='TB', compound='true', ranksep='2.0', nodesep='0.8')
    
    # Model parameters
    batch_size = 1024
    seq_len = 2048
    hidden_dim = 4096
    num_heads = 32
    head_dim = 128
    vocab_size = 50265
    num_experts = 16
    experts_per_gpu = 4
    
    # Add subgraphs for attention and MoE
    with dot.subgraph(name='cluster_attention') as attention:
        attention.attr(label='Attention Computation (GPUs 0-11)', style='rounded', color='blue', bgcolor='lightblue')
        attention.node_attr.update(shape='rectangle', style='filled', fillcolor='white')
        
        # Initial input
        attention.node('input', 'Model Input\\nInput: [batch_size=1024, seq_len=2048]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: all GPUs',
                      shape='ellipse', style='filled', fillcolor='lightgreen')
        
        # Token embedding
        attention.node('embed', 'Token Embedding\\nInput: [batch_size=1024, seq_len=2048]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11')
        
        # Process each layer
        for layer in range(4):
            # Layer norm before attention
            attention.node(f'ln1_l{layer}', f'Layer {layer} - LayerNorm\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11')
            
            # QKV projections on attention GPUs
            for gpu_id in range(12):
                if gpu_id < 8:  # GPUs 0-7 get 3 heads each
                    heads_per_gpu = 3
                    head_start = gpu_id * 3
                    head_end = head_start + 3
                else:  # GPUs 8-11 get 2 heads each
                    heads_per_gpu = 2
                    head_start = 24 + (gpu_id - 8) * 2
                    head_end = head_start + 2
                
                attention.node(f'qkv_l{layer}_gpu{gpu_id}', f'QKV Projection\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, head_dim=128, 3]\\nGPU: {gpu_id}')
            
            # Attention scores on each GPU
            for gpu_id in range(12):
                if gpu_id < 8:
                    heads_per_gpu = 3
                else:
                    heads_per_gpu = 2
                
                attention.node(f'scores_l{layer}_gpu{gpu_id}', f'Attention Scores\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, seq_len=2048]\\nGPU: {gpu_id}')
                
                attention.node(f'softmax_l{layer}_gpu{gpu_id}', f'Softmax\\nGPU {gpu_id}\\nInput/Output: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, seq_len=2048]\\nGPU: {gpu_id}')
                
                attention.node(f'attn_out_l{layer}_gpu{gpu_id}', f'Attention Output\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, head_dim=128]\\nOutput: [batch_size=1024, seq_len=2048, heads={heads_per_gpu}, head_dim=128]\\nGPU: {gpu_id}')
            
            # Output projection on each GPU
            for gpu_id in range(12):
                if gpu_id < 8:
                    heads_per_gpu = 3
                    output_dim = 3 * head_dim
                else:
                    heads_per_gpu = 2
                    output_dim = 2 * head_dim
                
                attention.node(f'o_proj_l{layer}_gpu{gpu_id}', f'O Projection\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, dim={output_dim}]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim={hidden_dim//12}]\\nGPU: {gpu_id}')
            
            # All-reduce for attention
            attention.node(f'all_reduce_attn_l{layer}', f'Attention All-Reduce\\nHierarchical\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11',
                          shape='ellipse', style='filled', fillcolor='yellow')
            
            # Residual add
            attention.node(f'res1_l{layer}', f'Residual Add L{layer}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 0-11',
                          shape='ellipse', style='filled', fillcolor='pink')
            
            # Broadcast to MoE GPUs
            attention.node(f'broadcast_l{layer}', f'Broadcast to MoE\\nLayer {layer}\\nFrom: GPUs 0-11\\nTo: GPUs 12-15\\nData: [batch_size=1024, seq_len=2048, hidden_dim=4096]',
                          shape='parallelogram', style='filled', fillcolor='lightcyan')

    with dot.subgraph(name='cluster_moe') as moe:
        moe.attr(label='MoE Computation (GPUs 12-15)', style='rounded', color='red', bgcolor='lightcoral')
        moe.node_attr.update(shape='rectangle', style='filled', fillcolor='white')
        
        for layer in range(4):
            # Receive from attention
            moe.node(f'recv_moe_l{layer}', f'Receive from Attention\\nLayer {layer}\\nFrom: GPUs 0-11\\nTo: GPUs 12-15\\nData: [batch_size=1024, seq_len=2048, hidden_dim=4096]',
                    shape='parallelogram', style='filled', fillcolor='lightcyan')
            
            # Layer norm on MoE GPUs
            for gpu_id in range(12, 16):
                moe.node(f'ln_moe_l{layer}_gpu{gpu_id}', f'LayerNorm\\nGPU {gpu_id}\\nInput/Output: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: {gpu_id}')
                
                # Gate network
                moe.node(f'gate_l{layer}_gpu{gpu_id}', f'Gate Network\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, num_experts=16]\\nGPU: {gpu_id}',
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # Expert routing (dashed lines)
                expert_start = (gpu_id - 12) * experts_per_gpu
                for expert_id in range(expert_start, expert_start + experts_per_gpu):
                    moe.node(f'expert_l{layer}_{expert_id}_gpu{gpu_id}', f'Expert {expert_id}\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: {gpu_id}')
                
                # Expert aggregation
                moe.node(f'expert_agg_l{layer}_gpu{gpu_id}', f'Expert Aggregation\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096, top_k=2]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: {gpu_id}',
                        shape='ellipse', style='filled', fillcolor='lightyellow')
                
                # All-to-all communication between MoE GPUs
                moe.node(f'all_to_all_l{layer}_gpu{gpu_id}', f'All-to-All\\nGPU {gpu_id}\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nGPU: 12-15',
                        shape='ellipse', style='filled', fillcolor='yellow')
            
            # Return to attention GPUs
            moe.node(f'return_l{layer}', f'Return to Attention\\nLayer {layer}\\nFrom: GPUs 12-15\\nTo: GPUs 0-11\\nData: [batch_size=1024, seq_len=2048, hidden_dim=4096]',
                    shape='parallelogram', style='filled', fillcolor='lightcyan')
    
    # Add edges for the flow
    dot.edge('input', 'embed')
    
    for layer in range(4):
        if layer == 0:
            dot.edge('embed', f'ln1_l{layer}')
        
        # Attention computation flow
        dot.edge(f'ln1_l{layer}', f'qkv_l{layer}_gpu0')
        
        # Process each GPU in attention
        for gpu_id in range(12):
            dot.edge(f'qkv_l{layer}_gpu{gpu_id}', f'scores_l{layer}_gpu{gpu_id}')
            dot.edge(f'scores_l{layer}_gpu{gpu_id}', f'softmax_l{layer}_gpu{gpu_id}')
            dot.edge(f'softmax_l{layer}_gpu{gpu_id}', f'attn_out_l{layer}_gpu{gpu_id}')
            dot.edge(f'attn_out_l{layer}_gpu{gpu_id}', f'o_proj_l{layer}_gpu{gpu_id}')
            
            # Connect to all-reduce
            if gpu_id == 0:
                dot.edge(f'o_proj_l{layer}_gpu{gpu_id}', f'all_reduce_attn_l{layer}')
            else:
                dot.edge(f'o_proj_l{layer}_gpu{gpu_id}', f'all_reduce_attn_l{layer}', style='dashed')
        
        dot.edge(f'all_reduce_attn_l{layer}', f'res1_l{layer}')
        dot.edge(f'res1_l{layer}', f'broadcast_l{layer}')
        
        # MoE computation flow
        dot.edge(f'broadcast_l{layer}', f'recv_moe_l{layer}', lhead='cluster_moe', ltail='cluster_attention')
        
        # Process each GPU in MoE
        for gpu_id in range(12, 16):
            dot.edge(f'recv_moe_l{layer}', f'ln_moe_l{layer}_gpu{gpu_id}')
            dot.edge(f'ln_moe_l{layer}_gpu{gpu_id}', f'gate_l{layer}_gpu{gpu_id}')
            
            expert_start = (gpu_id - 12) * experts_per_gpu
            for expert_id in range(expert_start, expert_start + experts_per_gpu):
                dot.edge(f'gate_l{layer}_gpu{gpu_id}', f'expert_l{layer}_{expert_id}_gpu{gpu_id}', style='dashed')
                dot.edge(f'recv_moe_l{layer}', f'expert_l{layer}_{expert_id}_gpu{gpu_id}')
                dot.edge(f'expert_l{layer}_{expert_id}_gpu{gpu_id}', f'expert_agg_l{layer}_gpu{gpu_id}')
            
            dot.edge(f'expert_agg_l{layer}_gpu{gpu_id}', f'all_to_all_l{layer}_gpu{gpu_id}')
        
        # Return to attention
        dot.edge(f'all_to_all_l{layer}_gpu12', f'return_l{layer}', lhead='cluster_attention', ltail='cluster_moe')
        
        if layer < 3:
            dot.edge(f'return_l{layer}', f'ln1_l{layer+1}')
        else:
            # Final output
            dot.node('final_output', 'Final Output\\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\\nOutput: [batch_size=1024, seq_len=2048, vocab_size=50265]\\nGPU: 0-11',
                    shape='ellipse', style='filled', fillcolor='lightgreen')
            dot.edge(f'return_l{layer}', 'final_output')
    
    return dot

if __name__ == '__main__':
    dag = create_ma_separation_dag()
    dag.render('../outputs/2025-10-16-20-29-23/ma_separation_dag', format='svg', cleanup=True)
    dag.save('../outputs/2025-10-16-20-29-23/ma_separation_dag.dot')
    print("MA Separation DAG generated successfully")