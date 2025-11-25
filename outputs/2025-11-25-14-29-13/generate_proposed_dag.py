#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """Create DAG for proposed model (EP=16) with expert-level parallelism"""
    
    dot = graphviz.Digraph(comment='MoE Proposed Model DAG (EP=16)')
    dot.attr(rankdir='TB', size='20,40')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input specifications
    batch_size = 128
    seq_len = 10000
    d_model = 4096
    num_heads = 32
    head_dim = 128
    experts_per_layer = 16
    
    # Create input node
    dot.node('input', 
             f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Create routing node for token distribution
    dot.node('router', 
             f'Token Router\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, k=2, d_model={d_model}]\\nGPU: All GPUs',
             shape='parallelogram', fillcolor='lightgray')
    
    # Process each layer
    for layer in range(16):
        # Layer norm before attention (shared across all GPUs)
        ln1_node = f'ln1_l{layer}'
        dot.node(ln1_node,
                 f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 fillcolor='lightyellow')
        
        # Multi-Head Attention (parallel across all GPUs)
        attn_node = f'attn_l{layer}'
        dot.node(attn_node,
                 f'Multi-Head Attention L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 fillcolor='lightcoral')
        
        # Residual Add 1
        residual1_node = f'residual1_l{layer}'
        dot.node(residual1_node,
                 f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 fillcolor='lightpink')
        
        # Layer norm before MoE
        ln2_node = f'ln2_l{layer}'
        dot.node(ln2_node,
                 f'LayerNorm L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 fillcolor='lightyellow')
        
        # Expert routing for this layer
        route_node = f'route_l{layer}'
        dot.node(route_node,
                 f'Expert Routing L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nGPU: All GPUs',
                 shape='parallelogram', fillcolor='lightgray')
        
        # Each expert on a separate GPU
        for expert_id in range(experts_per_layer):
            gpu_id = expert_id
            
            # Token receive - communication node
            recv_node = f'recv_l{layer}_e{expert_id}_gpu{gpu_id}'
            dot.node(recv_node,
                     f'Token Receive L{layer}E{expert_id}\\nInput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nOutput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nGPU: {gpu_id}',
                     shape='ellipse', fillcolor='lightgreen')
            
            # Expert computation
            expert_node = f'expert_l{layer}_e{expert_id}_gpu{gpu_id}'
            dot.node(expert_node,
                     f'Expert L{layer}E{expert_id}\\nInput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nOutput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nGPU: {gpu_id}',
                     fillcolor='lightsalmon')
            
            # Token send back - communication node
            send_node = f'send_l{layer}_e{expert_id}_gpu{gpu_id}'
            dot.node(send_node,
                     f'Token Send L{layer}E{expert_id}\\nInput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nOutput: [batch_size={batch_size}, tokens_per_expert, d_model={d_model}]\\nGPU: {gpu_id}',
                     shape='ellipse', fillcolor='lightgreen')
        
        # Expert aggregation after all experts complete
        agg_node = f'agg_l{layer}'
        dot.node(agg_node,
                 f'Expert Aggregation L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, k=2, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 shape='parallelogram', fillcolor='lightgray')
        
        # Residual Add 2
        residual2_node = f'residual2_l{layer}'
        dot.node(residual2_node,
                 f'ResidualAdd L{layer}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
                 fillcolor='lightpink')
    
    # Create output node
    dot.node('output', 
             f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}]\\nGPU: All GPUs',
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect nodes properly - ensuring no cycles
    
    # Input flow
    dot.edge('input', 'router')
    
    # Process each layer
    for layer in range(16):
        # Connect router for routing decisions
        dot.edge('router', f'route_l{layer}', style='dashed')
        
        # Connect layer components
        dot.edge('router', f'ln1_l{layer}')
        dot.edge(f'ln1_l{layer}', f'attn_l{layer}')
        dot.edge(f'attn_l{layer}', f'residual1_l{layer}')
        dot.edge(f'residual1_l{layer}', f'ln2_l{layer}')
        dot.edge(f'ln2_l{layer}', f'route_l{layer}')
        
        # Connect expert routing
        for expert_id in range(experts_per_layer):
            gpu_id = expert_id
            
            # Routing to expert
            dot.edge(f'route_l{layer}', f'recv_l{layer}_e{expert_id}_gpu{gpu_id}',
                     label=f'tokens_for_expert_{expert_id}')
            
            # Expert computation flow
            dot.edge(f'recv_l{layer}_e{expert_id}_gpu{gpu_id}', 
                     f'expert_l{layer}_e{expert_id}_gpu{gpu_id}')
            dot.edge(f'expert_l{layer}_e{expert_id}_gpu{gpu_id}', 
                     f'send_l{layer}_e{expert_id}_gpu{gpu_id}')
            
            # Expert results back to aggregation
            dot.edge(f'send_l{layer}_e{expert_id}_gpu{gpu_id}', f'agg_l{layer}')
        
        # Complete layer flow
        dot.edge(f'agg_l{layer}', f'residual2_l{layer}')
        
        # Connect to next layer or output
        if layer < 15:
            dot.edge(f'residual2_l{layer}', f'ln1_l{layer+1}')
        else:
            dot.edge(f'residual2_l{layer}', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    
    # Save DOT file
    dot_file_path = "../outputs/2025-11-25-14-29-13/proposed_model_dag_fixed.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG file
    svg_file_path = "../outputs/2025-11-25-14-29-13/proposed_model_dag_fixed.svg"
    dag.render(dot_file_path.replace('.dot', ''), format='svg', cleanup=True)
    
    print(f"Proposed DAG generated:")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")