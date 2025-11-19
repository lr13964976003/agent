#!/usr/bin/env python3
"""
Generate proposed cross-node expert parallelism DAG with EP=16 - Fixed version
"""

import graphviz
import os

def create_proposed_dag_fixed():
    """Create proposed DAG with Expert Parallelism=16 (1 expert per GPU)"""
    
    # Create directed graph
    dot = graphviz.Digraph('proposed_moe_fixed', 
                          comment='Proposed Large-Scale Expert Parallelism with EP=16',
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
    
    # Input
    dot.node('input', f'Model Input\\nBatch: {batch_size}, Seq: {seq_len}, Dim: {hidden_dim}', input_attrs)
    
    # Global routing
    dot.node('global_router', 'Global Token Router\\nAll GPUs', route_attrs)
    
    # Process each layer
    for layer_idx in range(num_layers):
        # Layer norm
        dot.node(f'layernorm_{layer_idx}', f'LayerNorm L{layer_idx}\\nAll GPUs', compute_attrs)
        
        # Multi-Head Attention
        dot.node(f'mha_qkv_{layer_idx}', f'MHA QKV Linear L{layer_idx}\\nAll GPUs', compute_attrs)
        dot.node(f'mha_attn_{layer_idx}', f'MHA Attention L{layer_idx}\\nAll GPUs', compute_attrs)
        dot.node(f'mha_out_{layer_idx}', f'MHA Output Linear L{layer_idx}\\nAll GPUs', compute_attrs)
        
        # Attention residual
        dot.node(f'attn_res_{layer_idx}', f'Attention Residual L{layer_idx}\\nAll GPUs', compute_attrs)
        
        # Local gating
        dot.node(f'local_gating_{layer_idx}', f'Layer Gating L{layer_idx}\\nAll GPUs', route_attrs)
        
        # Token routing and communication
        dot.node(f'token_route_{layer_idx}', f'Token Routing L{layer_idx}\\nAll GPUs', comm_attrs)
        
        # Individual experts (1 per GPU)
        for expert_id in range(num_experts):
            gpu_id = expert_id
            dot.node(f'expert_{layer_idx}_{expert_id}', 
                    f'Expert {expert_id} L{layer_idx}\\nGPU {gpu_id}\\nFFN 4096→32768→4096', 
                    expert_attrs)
        
        # Expert results aggregation
        dot.node(f'expert_agg_{layer_idx}', f'Expert Aggregation L{layer_idx}\\nAll GPUs', comm_attrs)
        
        # Final MoE processing
        dot.node(f'moe_combine_{layer_idx}', f'MoE Output Combine L{layer_idx}\\nAll GPUs', compute_attrs)
        
        # MoE residual
        dot.node(f'moe_res_{layer_idx}', f'MoE Residual L{layer_idx}\\nAll GPUs', compute_attrs)
    
    # Output
    dot.node('output', f'Model Output\\nBatch: {batch_size}, Seq: {seq_len}, Dim: {hidden_dim}', output_attrs)
    
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
        dot.edge(f'local_gating_{layer_idx}', f'token_route_{layer_idx}')
        
        # Connect to experts
        for expert_id in range(num_experts):
            dot.edge(f'token_route_{layer_idx}', f'expert_{layer_idx}_{expert_id}')
            dot.edge(f'expert_{layer_idx}_{expert_id}', f'expert_agg_{layer_idx}')
        
        # Aggregation path
        dot.edge(f'expert_agg_{layer_idx}', f'moe_combine_{layer_idx}')
        dot.edge(f'moe_combine_{layer_idx}', f'moe_res_{layer_idx}')
        dot.edge(f'attn_res_{layer_idx}', f'moe_res_{layer_idx}')  # residual
    
    dot.edge(f'moe_res_{3}', 'output')
    
    return dot

if __name__ == "__main__":
    os.makedirs('../outputs/2025-11-19-09-06-23', exist_ok=True)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag_fixed()
    
    # Save files
    proposed_dag.render('../outputs/2025-11-19-09-06-23/proposed_moe_dag_fixed', format='svg', cleanup=True)
    with open('../outputs/2025-11-19-09-06-23/proposed_moe_dag_fixed.dot', 'w') as f:
        f.write(str(proposed_dag.source))
    
    print("Proposed DAG (fixed) generated successfully")