#!/usr/bin/env python3
"""
Complete MoE transformation DAG with exact tensor dimensions and communication details
"""

import graphviz

def create_complete_moe_dag():
    """Create complete DAG with exact tensor dimensions and all transformations"""
    
    dot = graphviz.Digraph(comment='Complete Large-Scale MoE Transformation DAG')
    dot.attr(rankdir='TB', size='200,150', splines='ortho')
    dot.attr('node', fontsize='8', height='0.6', width='2.0')
    
    # Global parameters
    B = 32        # batch_size
    S = 2048      # seq_len  
    H = 7168      # hidden_dim
    Nh = 128      # num_heads
    Dh = 128      # head_dim
    Ne = 32       # num_experts
    He = 2048     # expert_hidden
    K = 2         # top_k
    
    # ============================= INPUT =============================
    dot.node('input', 
             f'Input Tokens\nGPU: All\nINPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightgray', style='filled')
    
    # ============================= MHA DECOMPOSITION =============================
    
    # Layer Normalization
    dot.node('layernorm1', 
             f'LayerNorm\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Query projection
    dot.node('q_proj', 
             f'Query Projection (Linear)\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Key projection  
    dot.node('k_proj', 
             f'Key Projection (Linear)\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Value projection
    dot.node('v_proj', 
             f'Value Projection (Linear)\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Attention score computation
    dot.node('attention_scores', 
             f'Attention Scores\nGPU: Shared\nINPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]×[B={B}, S={S}, Nh={Dh}, Dh={Nh}]\nOUTPUT: [B={B}, Nh={Nh}, S={S}, S={S}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Attention weights
    dot.node('attention_weights', 
             f'Softmax\nGPU: Shared\nINPUT: [B={B}, Nh={Nh}, S={S}, S={S}]\nOUTPUT: [B={B}, Nh={Nh}, S={S}, S={S}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Attention output
    dot.node('attention_output', 
             f'Attention Output\nGPU: Shared\nINPUT: [B={B}, Nh={Nh}, S={S}, S={S}]×[B={B}, S={S}, Nh={Nh}, Dh={Dh}]\nOUTPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Reshape back to hidden
    dot.node('attention_reshape', 
             f'Reshape Attention\nGPU: Shared\nINPUT: [B={B}, S={S}, Nh={Nh}, Dh={Dh}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # MHA output projection
    dot.node('mha_out_proj', 
             f'MHA Output Projection\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Residual connection
    dot.node('mha_residual', 
             f'MHA Residual Add\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}] + [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # ============================= GATING NETWORK =============================
    
    # Gating network
    dot.node('gating', 
             f'Gating Network (Linear)\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, Ne={Ne}]',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Top-K selection
    dot.node('topk_gate', 
             f'Top-K Selection (K={K})\nGPU: Shared\nINPUT: [B={B}, S={S}, Ne={Ne}]\nOUTPUT: [B={B}, S={S}, K={K}] indices',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # ============================= TOKEN DISTRIBUTION =============================
    
    # Compute token counts per expert
    dot.node('token_counts', 
             f'Compute Token Counts\nGPU: Shared\nINPUT: [B={B}, S={S}, K={K}] indices\nOUTPUT: Expert token counts',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # Token packing
    dot.node('token_pack', 
             f'Pack Tokens by Expert\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [Σtokens, H={H}] per expert',
             shape='parallelogram', fillcolor='lightyellow', style='filled')
    
    # All-to-all communication - scatter
    dot.node('scatter_comm', 
             f'All-to-All Scatter\nCommunication: NCCL\nINPUT: [Σtokens, H={H}] per expert\nOUTPUT: [Σtokens_gpu, H={H}] per GPU',
             shape='ellipse', fillcolor='lightgreen', style='filled')
    
    # ============================= EXPERT COMPUTATION =============================
    
    expert_nodes = []
    for gpu_id in range(32):
        # Create cluster for each GPU
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
            c.attr(label=f'GPU {gpu_id}', style='dotted')
            
            # Layer normalization
            ln_node = f'expert_{gpu_id}_ln'
            c.node(ln_node, 
                   f'Expert {gpu_id} LayerNorm\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, H={H}]\nOUTPUT: [Σtokens_gpu, H={H}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # Gate network
            gate_node = f'expert_{gpu_id}_gate'
            c.node(gate_node, 
                   f'Expert {gpu_id} Gate\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, H={H}]\nOUTPUT: [Σtokens_gpu, He={He}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # SiLU activation
            gate_act_node = f'expert_{gpu_id}_gate_act'
            c.node(gate_act_node, 
                   f'Expert {gpu_id} SiLU\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, He={He}]\nOUTPUT: [Σtokens_gpu, He={He}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # Expert network
            expert_node = f'expert_{gpu_id}_expert'
            c.node(expert_node, 
                   f'Expert {gpu_id} Expert\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, H={H}]\nOUTPUT: [Σtokens_gpu, He={He}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # Expert activation
            expert_act_node = f'expert_{gpu_id}_expert_act'
            c.node(expert_act_node, 
                   f'Expert {gpu_id} Expert Act\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, He={He}]\nOUTPUT: [Σtokens_gpu, He={He}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # Multiply gate and expert
            multiply_node = f'expert_{gpu_id}_multiply'
            c.node(multiply_node, 
                   f'Expert {gpu_id} Multiply\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, He={He}] × [Σtokens_gpu, He={He}]\nOUTPUT: [Σtokens_gpu, He={He}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            # Expert output projection
            out_proj_node = f'expert_{gpu_id}_out_proj'
            c.node(out_proj_node, 
                   f'Expert {gpu_id} Output Proj\nGPU: {gpu_id}\nINPUT: [Σtokens_gpu, He={He}]\nOUTPUT: [Σtokens_gpu, H={H}]',
                   shape='rect', fillcolor='lightblue', style='filled')
            
            expert_nodes.append((ln_node, gate_node, gate_act_node, expert_node, expert_act_node, multiply_node, out_proj_node))
    
    # ============================= AGGREGATION =============================
    
    # All-to-all communication - gather
    dot.node('gather_comm', 
             f'All-to-All Gather\nCommunication: NCCL\nINPUT: [Σtokens_gpu, H={H}] per GPU\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='ellipse', fillcolor='lightcoral', style='filled')
    
    # Token unpacking
    dot.node('token_unpack', 
             f'Unpack Tokens\nGPU: Shared\nINPUT: [Σtokens, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='parallelogram', fillcolor='lightcoral', style='filled')
    
    # ============================= FINAL OUTPUT =============================
    
    # Second layer normalization
    dot.node('layernorm2', 
             f'LayerNorm\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # FFN residual connection
    dot.node('ffn_residual', 
             f'FFN Residual Add\nGPU: Shared\nINPUT: [B={B}, S={S}, H={H}] + [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightblue', style='filled')
    
    # Final output
    dot.node('final_output', 
             f'Layer Output\nGPU: ALL\nINPUT: [B={B}, S={S}, H={H}]\nOUTPUT: [B={B}, S={S}, H={H}]',
             shape='rect', fillcolor='lightgray', style='filled')
    
    # ============================= CONNECTIONS =============================
    
    # MHA path
    dot.edge('input', 'layernorm1')
    dot.edge('layernorm1', 'q_proj')
    dot.edge('layernorm1', 'k_proj')
    dot.edge('layernorm1', 'v_proj')
    dot.edge('q_proj', 'attention_scores')
    dot.edge('k_proj', 'attention_scores')
    dot.edge('attention_scores', 'attention_weights')
    dot.edge('attention_weights', 'attention_output')
    dot.edge('v_proj', 'attention_output')
    dot.edge('attention_output', 'attention_reshape')
    dot.edge('attention_reshape', 'mha_out_proj')
    dot.edge('mha_out_proj', 'mha_residual')
    dot.edge('input', 'mha_residual')
    
    # Gating path
    dot.edge('mha_residual', 'gating')
    dot.edge('gating', 'topk_gate')
    dot.edge('topk_gate', 'token_counts')
    dot.edge('token_counts', 'token_pack')
    dot.edge('mha_residual', 'token_pack')
    dot.edge('token_pack', 'scatter_comm')
    
    # Expert computation
    for gpu_id, nodes in enumerate(expert_nodes):
        ln_node, gate_node, gate_act_node, expert_node, expert_act_node, multiply_node, out_proj_node = nodes
        dot.edge('scatter_comm', ln_node)
        dot.edge(ln_node, gate_node)
        dot.edge(gate_node, gate_act_node)
        dot.edge(ln_node, expert_node)
        dot.edge(expert_node, expert_act_node)
        dot.edge(gate_act_node, multiply_node)
        dot.edge(expert_act_node, multiply_node)
        dot.edge(multiply_node, out_proj_node)
        dot.edge(out_proj_node, 'gather_comm')
        
        # Dashed routing lines
        dot.edge('token_pack', ln_node, style='dashed', constraint='false')
    
    # Final aggregation
    dot.edge('gather_comm', 'token_unpack')
    dot.edge('token_unpack', 'layernorm2')
    dot.edge('layernorm2', 'ffn_residual')
    dot.edge('mha_residual', 'ffn_residual')
    dot.edge('ffn_residual', 'final_output')
    
    return dot

if __name__ == "__main__":
    dag = create_complete_moe_dag()
    
    # Save files
    dag.save('../outputs/2025-11-26-16-00-19/complete_moe_dag.dot')
    dag.render('../outputs/2025-11-26-16-00-19/complete_moe_dag', format='svg', cleanup=True)
    
    print("Complete DAG generated successfully:")
    print("- DOT file: ../outputs/2025-11-26-16-00-19/complete_moe_dag.dot")
    print("- SVG file: ../outputs/2025-11-26-16-00-19/complete_moe_dag.svg")