#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_simple_dag():
    """Create a simplified version of the DAG to test basic functionality"""
    
    dot = Digraph(comment='7B MoE Model Deployment DAG - Simplified')
    dot.attr(rankdir='TB', size='100,50', dpi='300')
    
    # Model configuration
    batch_size = 128
    seq_len = 10240
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    num_experts = 64
    
    input_dims = f"batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    
    # Create input node
    dot.node('input', f'Total Input\\nInput: [{input_dims}]\\nOutput: [{input_dims}]', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # Create a few representative layers
    for layer in range(3):  # Just 3 layers for testing
        layer_prefix = f'layer{layer}'
        
        # Layer Input
        layer_input = f'{layer_prefix}_input'
        dot.node(layer_input, f'Layer {layer} Input\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        
        if layer == 0:
            dot.edge('input', layer_input)
        else:
            dot.edge(f'layer{layer-1}_output', layer_input)
        
        # MHA QKV Linear (representative of TP)
        qkv_node = f'{layer_prefix}_qkv'
        qkv_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim*3}"
        dot.node(qkv_node, f'QKV Linear (TP=4)\\nGPU: [0-3]\\nInput: [{input_dims}]\\nOutput: [{qkv_dims}]',
                 style='filled', fillcolor='lightcoral')
        dot.edge(layer_input, qkv_node)
        
        # Attention Compute
        attn_node = f'{layer_prefix}_attn'
        attn_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}"
        dot.node(attn_node, f'Attention Compute\\nGPU: [0-3]\\nInput: [{qkv_dims}]\\nOutput: [{attn_dims}]',
                 style='filled', fillcolor='lightcoral')
        dot.edge(qkv_node, attn_node)
        
        # Attention All-reduce
        attn_ar = f'{layer_prefix}_attn_allreduce'
        dot.node(attn_ar, f'Attention All-Reduce\\nGPU: All\\nInput: [{attn_dims}]\\nOutput: [{input_dims}]',
                 shape='ellipse', style='filled', fillcolor='lightpink')
        dot.edge(attn_node, attn_ar)
        
        # Attention + Residual
        attn_residual = f'{layer_prefix}_attn_residual'
        dot.node(attn_residual, f'Attention + Residual\\nGPU: All\\nInput: [{input_dims}], [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(layer_input, attn_residual)
        dot.edge(attn_ar, attn_residual)
        
        # MoE Gate
        gate_node = f'{layer_prefix}_gate'
        gate_dims = f"batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}"
        dot.node(gate_node, f'MoE Gate Network\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{gate_dims}]',
                 style='filled', fillcolor='lightgreen')
        dot.edge(attn_residual, gate_node)
        
        # Expert 0 (representative)
        expert_node = f'{layer_prefix}_expert0'
        expert_dims = f"batch_size={batch_size//16}, seq_len={seq_len}, hidden_dim={hidden_dim}"
        dot.node(expert_node, f'Expert 0\\nGPU: [0-3]\\nInput: [{expert_dims}]\\nOutput: [{expert_dims}]',
                 style='filled', fillcolor='lightsalmon')
        dot.edge(gate_node, expert_node, style='dashed')
        
        # All-to-all communication
        a2a_comm = f'{layer_prefix}_alltoall'
        dot.node(a2a_comm, f'All-to-All Expert Routing\\nGPU: All\\nInput: [{expert_dims}]\\nOutput: [{expert_dims}]',
                 shape='ellipse', style='filled', fillcolor='lightpink')
        dot.edge(expert_node, a2a_comm)
        
        # MoE Aggregation
        moe_agg = f'{layer_prefix}_moe_aggregate'
        dot.node(moe_agg, f'MoE Output Aggregation\\nGPU: All\\nInput: [{expert_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(a2a_comm, moe_agg)
        
        # MoE + Residual
        moe_residual = f'{layer_prefix}_moe_residual'
        dot.node(moe_residual, f'MoE + Residual\\nGPU: All\\nInput: [{input_dims}], [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(attn_residual, moe_residual, style='dashed')
        dot.edge(moe_agg, moe_residual)
        
        # Layer output
        layer_output = f'{layer_prefix}_output'
        dot.node(layer_output, f'Layer {layer} Output\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                 style='filled', fillcolor='lightblue')
        dot.edge(moe_residual, layer_output)
        
        if layer == 2:  # Last layer
            # Final output
            dot.node('final_output', f'Final Output\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                     shape='ellipse', style='filled', fillcolor='lightgreen')
            dot.edge(layer_output, 'final_output')
    
    return dot

def main():
    output_dir = "../outputs/2025-12-01-16-02-27"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating simplified 7B MoE Model Deployment DAG...")
    dag = create_simple_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "simple_moe_deployment_dag.dot")
    dag.save(dot_file)
    print(f"Saved DOT file: {dot_file}")
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "simple_moe_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved SVG image: {svg_file}")
    
    return {
        "dot_file": dot_file,
        "svg_file": svg_file
    }

if __name__ == "__main__":
    files = main()
    print("\nGenerated files:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")