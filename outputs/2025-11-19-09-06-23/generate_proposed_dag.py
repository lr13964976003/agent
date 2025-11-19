#!/usr/bin/env python3
"""
Generate proposed cross-node expert parallelism DAG with EP=16
"""

import graphviz
import os

def create_proposed_dag():
    """Create proposed DAG with Expert Parallelism=16 (1 expert per GPU)"""
    
    # Create directed graph
    dot = graphviz.Digraph('proposed_moe', 
                          comment='Proposed Large-Scale Expert Parallelism with EP=16',
                          graph_attr={
                              'rankdir': 'TB',
                              'bgcolor': 'white',
                              'fontname': 'Arial',
                              'fontsize': '12',
                              'splines': 'ortho'
                          })
    
    # Define node attributes
    compute_attrs = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue', 'fontname': 'Arial'}
    comm_attrs = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightyellow', 'fontname': 'Arial'}
    route_attrs = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightgreen', 'fontname': 'Arial'}
    input_attrs = {'shape': 'plaintext', 'fontname': 'Arial'}
    output_attrs = {'shape': 'plaintext', 'fontname': 'Arial'}
    expert_attrs = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightcoral', 'fontname': 'Arial'}
    
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
    
    # Global routing for all layers
    dot.node('global_router', f'''<<b>Global Token Router</b><br/>
    All GPUs<br/>
    Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
    Output: [token routing decisions per expert]>''', route_attrs)
    
    # Process each layer
    for layer_idx in range(num_layers):
        # Layer norm (single GPU per token)
        dot.node(f'layernorm_{layer_idx}', f'''<<b>LayerNorm (Layer {layer_idx})</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Multi-Head Attention (no tensor parallelism)
        dot.node(f'mha_qkv_{layer_idx}', f'''<<b>MHA QKV Linear</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]>''', 
                compute_attrs)
        
        dot.node(f'mha_attn_{layer_idx}', f'''<<b>MHA Attention</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        dot.node(f'mha_out_{layer_idx}', f'''<<b>MHA Output Linear</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Attention residual
        dot.node(f'attn_res_{layer_idx}', f'''<<b>Attention Residual Add</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # Local gating for expert distribution
        dot.node(f'local_gating_{layer_idx}', f'''<<b>Layer {layer_idx} Local Gating</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]<br/>
        Output: [token assignments → expert destinations]>''', route_attrs)
        
        # Token batching and communication
        dot.node(f'token_batch_{layer_idx}', f'''<<b>Token Batching & Async Send</b><br/>
        All GPUs<br/>
        Input: [tokens with expert assignments]<br/>
        Output: [batches per expert]>''', comm_attrs)
        
        # Individual experts (1 per GPU)
        for expert_id in range(num_experts):
            gpu_id = expert_id
            dot.node(f'expert_{layer_idx}_{expert_id}', f'''<<b>Expert {expert_id} (Layer {layer_idx})</b><br/>
            GPU: {gpu_id}<br/>
            Expert ID: {expert_id}<br/>
            Input: [token batch, hidden_dim={hidden_dim}]<br/>
            Output: [processed tokens, hidden_dim={hidden_dim}]<br/>
            Memory: 512MB<br/>
            FFN: 4096→32768→4096>''', 
                    expert_attrs)
        
        # Expert results aggregation
        dot.node(f'expert_results_{layer_idx}', f'''<<b>Expert Results Collection</b><br/>
        All GPUs<br/>
        Input: [processed tokens from 16 experts]<br/>
        Output: [reordered tokens, batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                comm_attrs)
        
        # Final MoE layer norm and residual
        dot.node(f'moe_combine_{layer_idx}', f'''<<b>MoE Output Combine</b><br/>
        All GPUs<br/>
        Input: [expert outputs, gating weights]<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
        
        # MoE residual
        dot.node(f'moe_res_{layer_idx}', f'''<<b>MoE Residual Add</b><br/>
        All GPUs<br/>
        Input: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}] (x2)<br/>
        Output: [batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]>''', 
                compute_attrs)
    
    # Output
    dot.node('output', f'''<<b>Model Output</b><br/>
    Batch Size: {batch_size}<br/>
    Sequence Length: {seq_len}<br/>
    Hidden Dim: {hidden_dim}>''', output_attrs)
    
    # Connect the DAG
    dot.edge('input', 'global_router')
    
    # Layer connections
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            dot.edge('global_router', f'layernorm_{layer_idx}')
        else:
            dot.edge(f'moe_res_{layer_idx-1}', f'layernorm_{layer_idx}')
        
        # Attention path
        dot.edge(f'layernorm_{layer_idx}', f'mha_qkv_{layer_idx}')
        dot.edge(f'mha_qkv_{layer_idx}', f'mha_attn_{layer_idx}')
        dot.edge(f'mha_attn_{layer_idx}', f'mha_out_{layer_idx}')
        dot.edge(f'mha_out_{layer_idx}', f'attn_res_{layer_idx}')
        dot.edge(f'layernorm_{layer_idx}', f'attn_res_{layer_idx}')  # residual
        
        # MoE path
        dot.edge(f'attn_res_{layer_idx}', f'local_gating_{layer_idx}')
        dot.edge(f'local_gating_{layer_idx}', f'token_batch_{layer_idx}')
        
        # Connect to individual experts
        for expert_id in range(num_experts):
            dot.edge(f'token_batch_{layer_idx}', f'expert_{layer_idx}_{expert_id}')
            dot.edge(f'expert_{layer_idx}_{expert_id}', f'expert_results_{layer_idx}')
        
        # Aggregation path
        dot.edge(f'expert_results_{layer_idx}', f'moe_combine_{layer_idx}')
        dot.edge(f'moe_combine_{layer_idx}', f'moe_res_{layer_idx}')
        dot.edge(f'attn_res_{layer_idx}', f'moe_res_{layer_idx}')  # residual
    
    dot.edge(f'moe_res_{3}', 'output')
    
    return dot

if __name__ == "__main__":
    os.makedirs('../outputs/2025-11-19-09-06-23', exist_ok=True)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    
    # Save files
    proposed_dag.render('../outputs/2025-11-19-09-06-23/proposed_moe_dag', format='svg', cleanup=True)
    with open('../outputs/2025-11-19-09-06-23/proposed_moe_dag.dot', 'w') as f:
        f.write(str(proposed_dag.source))
    
    print("Proposed DAG generated successfully")