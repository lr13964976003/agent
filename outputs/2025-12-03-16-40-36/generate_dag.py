#!/usr/bin/env python3

import graphviz

def create_moe_dag():
    """
    Create a complete DAG for MoE LLM deployment with EP64_TP2_PP1 strategy
    """
    dot = graphviz.Digraph(comment='MoE LLM Deployment DAG - EP64_TP2_PP1')
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    batch_size = 128
    seq_len = 1024
    token_dim = 1024
    moe_hidden = 2048
    num_layers = 16
    num_experts = 64
    num_gpus = 128
    gpus_per_group = 2
    num_groups = 64
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('input', 
               f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Process each layer
    for layer_idx in range(num_layers):
        with dot.subgraph(name=f'cluster_layer_{layer_idx}') as c:
            c.attr(label=f'Layer {layer_idx}', style='rounded,filled', fillcolor='lightcyan')
            
            # Layer Norm (Attention path)
            c.node(f'layernorm_att_{layer_idx}',
                   f'LayerNorm (Attention)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Multi-Head Attention with Tensor Parallelism (2-way split)
            # QKV projection split column-wise
            c.node(f'qkv_proj_0_{layer_idx}',
                   f'QKV Projection GPU-0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'qkv_proj_1_{layer_idx}',
                   f'QKV Projection GPU-1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Attention computation
            c.node(f'attention_0_{layer_idx}',
                   f'Attention GPU-0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'attention_1_{layer_idx}',
                   f'Attention GPU-1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Output projection (row parallel)
            c.node(f'attn_out_0_{layer_idx}',
                   f'Attention Output GPU-0\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim//2}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'attn_out_1_{layer_idx}',
                   f'Attention Output GPU-1\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim//2}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # All-reduce for attention output
            c.node(f'attn_allreduce_{layer_idx}',
                   f'All-Reduce Attention\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim//2}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='ellipse', fillcolor='lightblue')
            
            # Residual add
            c.node(f'residual_att_{layer_idx}',
                   f'Residual Add (Attention)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Layer Norm (MoE path)
            c.node(f'layernorm_moe_{layer_idx}',
                   f'LayerNorm (MoE)\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Gate computation (routing)
            c.node(f'gate_{layer_idx}',
                   f'Gate Network\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]',
                   shape='rectangle', fillcolor='lightgreen')
            
            # Expert selection (routing)
            c.node(f'expert_select_{layer_idx}',
                   f'Expert Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, top_k=2]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Process each expert group (64 groups total)
            for group_idx in range(num_groups):
                with dot.subgraph(name=f'cluster_expert_group_{group_idx}_layer_{layer_idx}') as ec:
                    ec.attr(label=f'Expert Group {group_idx} (GPUs {group_idx*2}-{group_idx*2+1})', style='rounded,filled', fillcolor='lightpink')
                    
                    # Expert 0 in group (GPU pair)
                    ec.node(f'expert_0_{group_idx}_{layer_idx}',
                           f'Expert 0 Group {group_idx}\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert 1 in group (GPU pair)  
                    ec.node(f'expert_1_{group_idx}_{layer_idx}',
                           f'Expert 1 Group {group_idx}\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Expert processing with tensor parallelism
                    # First linear (column parallel)
                    ec.node(f'expert_linear1_0_{group_idx}_{layer_idx}',
                           f'Expert Linear1 GPU-0\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    ec.node(f'expert_linear1_1_{group_idx}_{layer_idx}',
                           f'Expert Linear1 GPU-1\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # GELU activation
                    ec.node(f'expert_gelu_0_{group_idx}_{layer_idx}',
                           f'GELU GPU-0\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    ec.node(f'expert_gelu_1_{group_idx}_{layer_idx}',
                           f'GELU GPU-1\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # Second linear (row parallel)
                    ec.node(f'expert_linear2_0_{group_idx}_{layer_idx}',
                           f'Expert Linear2 GPU-0\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    ec.node(f'expert_linear2_1_{group_idx}_{layer_idx}',
                           f'Expert Linear2 GPU-1\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, ffn_hidden={moe_hidden//2}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim//2}]',
                           shape='rectangle', fillcolor='lightgreen')
                    
                    # All-reduce for expert output
                    ec.node(f'expert_allreduce_{group_idx}_{layer_idx}',
                           f'All-Reduce Expert\\nInput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim//2}]\\nOutput: [batch_size={batch_size//num_groups}, seq_len={seq_len}, hidden={token_dim}]',
                           shape='ellipse', fillcolor='lightblue')
            
            # Expert aggregation (weighted sum)
            c.node(f'expert_agg_{layer_idx}',
                   f'Expert Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Final residual add
            c.node(f'residual_final_{layer_idx}',
                   f'Final Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('output', 
               f'Final Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to first layer
    dot.edge('input', 'layernorm_att_0')
    
    # Process each layer connections
    for layer_idx in range(num_layers):
        # Attention path
        dot.edge(f'layernorm_att_{layer_idx}', f'qkv_proj_0_{layer_idx}')
        dot.edge(f'layernorm_att_{layer_idx}', f'qkv_proj_1_{layer_idx}')
        
        dot.edge(f'qkv_proj_0_{layer_idx}', f'attention_0_{layer_idx}')
        dot.edge(f'qkv_proj_1_{layer_idx}', f'attention_1_{layer_idx}')
        
        dot.edge(f'attention_0_{layer_idx}', f'attn_out_0_{layer_idx}')
        dot.edge(f'attention_1_{layer_idx}', f'attn_out_1_{layer_idx}')
        
        dot.edge(f'attn_out_0_{layer_idx}', f'attn_allreduce_{layer_idx}')
        dot.edge(f'attn_out_1_{layer_idx}', f'attn_allreduce_{layer_idx}')
        
        # Residual connection for attention
        dot.edge(f'layernorm_att_{layer_idx}', f'residual_att_{layer_idx}')
        dot.edge(f'attn_allreduce_{layer_idx}', f'residual_att_{layer_idx}')
        
        # MoE path
        dot.edge(f'residual_att_{layer_idx}', f'layernorm_moe_{layer_idx}')
        dot.edge(f'layernorm_moe_{layer_idx}', f'gate_{layer_idx}')
        dot.edge(f'gate_{layer_idx}', f'expert_select_{layer_idx}')
        
        # Expert routing (dashed lines for selection)
        for group_idx in range(num_groups):
            dot.edge(f'expert_select_{layer_idx}', f'expert_linear1_0_{group_idx}_{layer_idx}', style='dashed')
            dot.edge(f'expert_select_{layer_idx}', f'expert_linear1_1_{group_idx}_{layer_idx}', style='dashed')
            
            # Expert computation flow
            dot.edge(f'expert_linear1_0_{group_idx}_{layer_idx}', f'expert_gelu_0_{group_idx}_{layer_idx}')
            dot.edge(f'expert_linear1_1_{group_idx}_{layer_idx}', f'expert_gelu_1_{group_idx}_{layer_idx}')
            
            dot.edge(f'expert_gelu_0_{group_idx}_{layer_idx}', f'expert_linear2_0_{group_idx}_{layer_idx}')
            dot.edge(f'expert_gelu_1_{group_idx}_{layer_idx}', f'expert_linear2_1_{group_idx}_{layer_idx}')
            
            dot.edge(f'expert_linear2_0_{group_idx}_{layer_idx}', f'expert_allreduce_{group_idx}_{layer_idx}')
            dot.edge(f'expert_linear2_1_{group_idx}_{layer_idx}', f'expert_allreduce_{group_idx}_{layer_idx}')
            
            # Connect expert outputs to aggregation
            dot.edge(f'expert_allreduce_{group_idx}_{layer_idx}', f'expert_agg_{layer_idx}')
        
        # Final residual and next layer
        dot.edge(f'residual_att_{layer_idx}', f'residual_final_{layer_idx}')
        dot.edge(f'expert_agg_{layer_idx}', f'residual_final_{layer_idx}')
        
        # Connect to next layer or output
        if layer_idx < num_layers - 1:
            dot.edge(f'residual_final_{layer_idx}', f'layernorm_att_{layer_idx+1}')
        else:
            dot.edge(f'residual_final_{layer_idx}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_moe_dag()
    
    # Save DOT file
    dag.save('../outputs/2025-12-03-16-40-36/moe_deployment_dag.dot')
    
    # Save SVG image
    dag.render('../outputs/2025-12-03-16-40-36/moe_deployment_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-03-16-40-36/moe_deployment_dag.dot")
    print(f"SVG file: ../outputs/2025-12-03-16-40-36/moe_deployment_dag.svg")