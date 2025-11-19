#!/usr/bin/env python3
"""
Generate baseline MoE DAG with TP=8, PP=2
"""

import graphviz
import os

def create_baseline_dag():
    """Create baseline DAG with Tensor Parallelism=8 and Pipeline Parallelism=2"""
    
    # Create directed graph
    dot = graphviz.Digraph('baseline_moe', 
                          comment='Baseline MoE with TP=8, PP=2',
                          graph_attr={
                              'rankdir': 'TB',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12'
                          })
    
    # Define node attributes
    compute_attrs = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue', 'fontname': 'Arial'}
    comm_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow', 'fontname': 'Arial'}
    route_attrs = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen', 'fontname': 'Arial'}
    input_attrs = {'shape': 'plaintext', 'fontname': 'Arial'}
    output_attrs = {'shape': 'plaintext', 'fontname': 'Arial'}
    
    # Model dimensions
    batch_size = 128
    seq_len = 10000
    hidden_dim = 4096
    ffn_hidden = 32768
    num_heads = 32
    head_dim = 128
    num_layers = 4
    num_experts = 16
    
    # Input nodes
    dot.node('input', f'''<<b>Model Input</b><br/>
    Batch Size: {batch_size}<br/>
    Sequence Length: {seq_len}<br/>
    Hidden Dim: {hidden_dim}>''', input_attrs)
    
    # Pipeline Stage 0 (Devices 0-7)
    dot.node('pipeline_split', f'''<<b>Pipeline Split</b><br/>
    Stage 0: Devices 0-7<br/>
    Stage 1: Devices 8-15>''', route_attrs)
    
    # Layer 0 - Stage 0
    for layer_idx in [0, 1]:
        # Layer norm (replicated across tensor parallel group)
        dot.node(f'layernorm_{layer_idx}_s0', f'''<<b>LayerNorm (Layer {layer_idx})</b><br/>
        GPU: 0-7<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Multi-Head Attention (Tensor Parallel)
        dot.node(f'mha_qkv_{layer_idx}_s0', f'''<<b>MHA QKV Linear (TP=8)</b><br/>
        GPU: 0-7 (column parallel)<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]>''', 
                compute_attrs)
        
        dot.node(f'mha_attn_{layer_idx}_s0', f'''<<b>MHA Attention (TP=8)</b><br/>
        GPU: 0-7<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//8}]>''', 
                compute_attrs)
        
        dot.node(f'mha_out_{layer_idx}_s0', f'''<<b>MHA Output Linear (TP=8)</b><br/>
        GPU: 0-7 (row parallel)<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//8}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Attention residual
        dot.node(f'attn_res_{layer_idx}_s0', f'''<<b>Attention Residual Add</b><br/>
        GPU: 0-7<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # MoE Layer
        dot.node(f'moe_gating_{layer_idx}_s0', f'''<<b>MoE Gating Network</b><br/>
        GPU: 0-7<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [routing decisions, expert assignments]>''', 
                route_attrs)
        
        # Expert processing (8 experts per GPU)
        for gpu_idx in range(8):
            expert_start = gpu_idx * 2
            expert_end = expert_start + 1
            
            dot.node(f'expert_{layer_idx}_gpu{gpu_idx}_s0', f'''<<b>Expert MLP (Layer {layer_idx})</b><br/>
            GPU: {gpu_idx}<br/>
            Experts: {expert_start}-{expert_end}<br/>
            Input: [variable tokens, hidden_dim={hidden_dim}]<br/>
            Output: [variable tokens, hidden_dim={hidden_dim}]<br/>
            Params: 512MB total>''', 
                    compute_attrs)
        
        # Expert aggregation
        dot.node(f'expert_agg_{layer_idx}_s0', f'''<<b>Expert Output Aggregation</b><br/>
        GPU: 0-7<br/>
        Input: [expert outputs from 16 experts]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                route_attrs)
        
        # MoE residual
        dot.node(f'moe_res_{layer_idx}_s0', f'''<<b>MoE Residual Add</b><br/>
        GPU: 0-7<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
    
    # Pipeline communication
    dot.node('pipeline_comm', f'''<<b>Pipeline Communication</b><br/>
    Stage 0 → Stage 1<br/>
    Transfer: 4 layers output<br/>
    Async communication>''', comm_attrs)
    
    # Layer 2-3 - Stage 1 (similar structure)
    for layer_idx in [2, 3]:
        # Layer norm
        dot.node(f'layernorm_{layer_idx}_s1', f'''<<b>LayerNorm (Layer {layer_idx})</b><br/>
        GPU: 8-15<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Multi-Head Attention
        dot.node(f'mha_qkv_{layer_idx}_s1', f'''<<b>MHA QKV Linear (TP=8)</b><br/>
        GPU: 8-15 (column parallel)<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]>''', 
                compute_attrs)
        
        dot.node(f'mha_attn_{layer_idx}_s1', f'''<<b>MHA Attention (TP=8)</b><br/>
        GPU: 8-15<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//8}]>''', 
                compute_attrs)
        
        dot.node(f'mha_out_{layer_idx}_s1', f'''<<b>MHA Output Linear (TP=8)</b><br/>
        GPU: 8-15 (row parallel)<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//8}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        dot.node(f'attn_res_{layer_idx}_s1', f'''<<b>Attention Residual Add</b><br/>
        GPU: 8-15<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # MoE Layer
        dot.node(f'moe_gating_{layer_idx}_s1', f'''<<b>MoE Gating Network</b><br/>
        GPU: 8-15<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [routing decisions, expert assignments]>''', 
                route_attrs)
        
        # Expert processing
        for gpu_idx in range(8, 16):
            expert_start = (gpu_idx - 8) * 2
            expert_end = expert_start + 1
            
            dot.node(f'expert_{layer_idx}_gpu{gpu_idx}_s1', f'''<<b>Expert MLP (Layer {layer_idx})</b><br/>
            GPU: {gpu_idx}<br/>
            Experts: {expert_start}-{expert_end}<br/>
            Input: [variable tokens, hidden_dim={hidden_dim}]<br/>
            Output: [variable tokens, hidden_dim={hidden_dim}]<br/>
            Params: 512MB total>''', 
                    compute_attrs)
        
        dot.node(f'expert_agg_{layer_idx}_s1', f'''<<b>Expert Output Aggregation</b><br/>
        GPU: 8-15<br/>
        Input: [expert outputs from 16 experts]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                route_attrs)
        
        dot.node(f'moe_res_{layer_idx}_s1', f'''<<b>MoE Residual Add</b><br/>
        GPU: 8-15<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
    
    # Output
    dot.node('output', f'''<<b>Model Output</b><br/>
    Batch Size: {batch_size}<br/>
    Sequence Length: {seq_len}<br/>
    Hidden Dim: {hidden_dim}>''', output_attrs)
    
    # Connect nodes
    dot.edge('input', 'pipeline_split')
    
    # Stage 0 connections
    for layer_idx in [0, 1]:
        prev_node = 'pipeline_split' if layer_idx == 0 else f'moe_res_0_s0'
        dot.edge(prev_node, f'layernorm_{layer_idx}_s0')
        dot.edge(f'layernorm_{layer_idx}_s0', f'mha_qkv_{layer_idx}_s0')
        dot.edge(f'mha_qkv_{layer_idx}_s0', f'mha_attn_{layer_idx}_s0')
        dot.edge(f'mha_attn_{layer_idx}_s0', f'mha_out_{layer_idx}_s0')
        dot.edge(f'mha_out_{layer_idx}_s0', f'attn_res_{layer_idx}_s0')
        dot.edge(f'layernorm_{layer_idx}_s0', f'attn_res_{layer_idx}_s0')  # residual
        dot.edge(f'attn_res_{layer_idx}_s0', f'moe_gating_{layer_idx}_s0')
        
        # Connect to experts
        for gpu_idx in range(8):
            dot.edge(f'moe_gating_{layer_idx}_s0', f'expert_{layer_idx}_gpu{gpu_idx}_s0')
            dot.edge(f'expert_{layer_idx}_gpu{gpu_idx}_s0', f'expert_agg_{layer_idx}_s0')
        
        dot.edge(f'expert_agg_{layer_idx}_s0', f'moe_res_{layer_idx}_s0')
        dot.edge(f'attn_res_{layer_idx}_s0', f'moe_res_{layer_idx}_s0')  # residual
    
    # Pipeline communication
    dot.edge(f'moe_res_1_s0', 'pipeline_comm')
    dot.edge('pipeline_comm', 'layernorm_2_s1')
    
    # Stage 1 connections
    for layer_idx in [2, 3]:
        if layer_idx == 2:
            dot.edge('pipeline_comm', f'layernorm_{layer_idx}_s1')
        else:
            dot.edge(f'moe_res_2_s1', f'layernorm_{layer_idx}_s1')
        
        dot.edge(f'layernorm_{layer_idx}_s1', f'mha_qkv_{layer_idx}_s1')
        dot.edge(f'mha_qkv_{layer_idx}_s1', f'mha_attn_{layer_idx}_s1')
        dot.edge(f'mha_attn_{layer_idx}_s1', f'mha_out_{layer_idx}_s1')
        dot.edge(f'mha_out_{layer_idx}_s1', f'attn_res_{layer_idx}_s1')
        dot.edge(f'layernorm_{layer_idx}_s1', f'attn_res_{layer_idx}_s1')  # residual
        dot.edge(f'attn_res_{layer_idx}_s1', f'moe_gating_{layer_idx}_s1')
        
        # Connect to experts
        for gpu_idx in range(8, 16):
            dot.edge(f'moe_gating_{layer_idx}_s1', f'expert_{layer_idx}_gpu{gpu_idx}_s1')
            dot.edge(f'expert_{layer_idx}_gpu{gpu_idx}_s1', f'expert_agg_{layer_idx}_s1')
        
        dot.edge(f'expert_agg_{layer_idx}_s1', f'moe_res_{layer_idx}_s1')
        dot.edge(f'attn_res_{layer_idx}_s1', f'moe_res_{layer_idx}_s1')  # residual
    
    dot.edge(f'moe_res_3_s1', 'output')
    
    return dot

if __name__ == "__main__":
    os.makedirs('../outputs/2025-11-19-09-06-23', exist_ok=True)
    
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    
    # Save files
    baseline_dag.render('../outputs/2025-11-19-09-06-23/baseline_moe_dag', format='svg', cleanup=True)
    with open('../outputs/2025-11-19-09-06-23/baseline_moe_dag.dot', 'w') as f:
        f.write(str(baseline_dag.source))
    
    print("Baseline DAG generated successfully")