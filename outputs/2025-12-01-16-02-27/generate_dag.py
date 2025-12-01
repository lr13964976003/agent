#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_model_deployment_dag():
    """
    Create a complete DAG for 7B MoE model deployment with TP=4, EP=16, PP=1
    64 GPUs total, 16 layers, each layer has MHA + MoE
    """
    
    # Create the DAG
    dot = Digraph(comment='7B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='300,200', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Model configuration
    batch_size = 128
    seq_len = 10240
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    num_experts = 64
    experts_per_gpu = 4
    tp_size = 4
    ep_size = 16
    total_gpus = 64
    num_layers = 16
    
    # Input dimensions
    input_dims = f"batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    attention_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}"
    expert_dims = f"batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
    
    # Create input node
    dot.node('input', f'Total Input\\nInput: [{input_dims}]\\nOutput: [{input_dims}]', 
             shape='ellipse', style='filled', fillcolor='lightgreen')
    
    prev_node = 'input'
    
    # Create each layer
    for layer in range(num_layers):
        layer_prefix = f'layer{layer}'
        
        # Layer Input
        layer_input = f'{layer_prefix}_input'
        dot.node(layer_input, f'Layer {layer} Input\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(prev_node, layer_input)
        
        # Multi-Head Attention (MHA) - Tensor Parallel
        ## QKV Linear - Column Parallel
        for tp in range(tp_size):
            qkv_node = f'{layer_prefix}_qkv_tp{tp}'
            output_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//tp_size}, d_k={head_dim*3}"
            
            dot.node(qkv_node, 
                     f'QKV Linear TP{tp}\\nGPU: [{tp},{tp+4},...,{tp+60}]\\nInput: [{input_dims}]\\nOutput: [{output_dims}]',
                     style='filled', fillcolor='lightcoral')
            dot.edge(layer_input, qkv_node)
        
        # All-reduce for QKV
        qkv_ar = f'{layer_prefix}_qkv_allreduce'
        qkv_output_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={num_heads//tp_size}, d_k={head_dim*3}"
        dot.node(qkv_ar, f'QKV All-Reduce\\nGPU: All\\nInput: [{qkv_output_dims}]\\nOutput: [{qkv_output_dims}]',
                 shape='ellipse', style='filled', fillcolor='lightpink')
        for tp in range(tp_size):
            dot.edge(f'{layer_prefix}_qkv_tp{tp}', qkv_ar)
        
        # Attention Computation - Head Parallel
        for tp in range(tp_size):
            attn_node = f'{layer_prefix}_attn_tp{tp}'
            heads_per_tp = num_heads // tp_size
            attn_dims = f"batch_size={batch_size}, seq_len={seq_len}, heads={heads_per_tp}, d_k={head_dim}"
            
            dot.node(attn_node,
                     f'Attention Compute TP{tp}\\nGPU: [{tp},{tp+4},...,{tp+60}]\\nInput: [{qkv_output_dims}]\\nOutput: [{attn_dims}]',
                     style='filled', fillcolor='lightcoral')
            dot.edge(qkv_ar, attn_node)
        
        # Attention Output Projection - Row Parallel
        for tp in range(tp_size):
            attn_out_node = f'{layer_prefix}_attn_out_tp{tp}'
            output_projection_dims = f"batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim//tp_size}"
            
            dot.node(attn_out_node,
                     f'Attention Output TP{tp}\\nGPU: [{tp},{tp+4},...,{tp+60}]\\nInput: [{attn_dims}]\\nOutput: [{output_projection_dims}]',
                     style='filled', fillcolor='lightcoral')
            dot.edge(f'{layer_prefix}_attn_tp{tp}', attn_out_node)
        
        # Attention All-reduce
        attn_ar = f'{layer_prefix}_attn_allreduce'
        dot.node(attn_ar, f'Attention All-Reduce\\nGPU: All\\nInput: [{output_projection_dims}]\\nOutput: [{input_dims}]',
                 shape='ellipse', style='filled', fillcolor='lightpink')
        for tp in range(tp_size):
            dot.edge(f'{layer_prefix}_attn_out_tp{tp}', attn_ar)
        
        # Residual Add for Attention
        attn_residual = f'{layer_prefix}_attn_residual'
        dot.node(attn_residual, f'Attention + Residual\\nGPU: All\\nInput: [{input_dims}], [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(layer_input, attn_residual)
        dot.edge(attn_ar, attn_residual)
        
        # Layer Norm after Attention
        attn_ln = f'{layer_prefix}_attn_layernorm'
        dot.node(attn_ln, f'Layer Norm (Post-Attn)\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                 style='filled', fillcolor='lightblue')
        dot.edge(attn_residual, attn_ln)
        
        # MoE Layer - Expert Parallel
        ## Gate Network (selects experts)
        gate_node = f'{layer_prefix}_gate'
        gate_dims = f"batch_size={batch_size}, seq_len={seq_len}, num_experts={num_experts}"
        dot.node(gate_node, f'MoE Gate Network\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{gate_dims}]',
                 style='filled', fillcolor='lightgreen')
        dot.edge(attn_ln, gate_node)
        
        # Expert computations - distributed across EP groups
        for ep in range(ep_size):
            for expert in range(experts_per_gpu):
                expert_id = ep * experts_per_gpu + expert
                expert_node = f'{layer_prefix}_expert{expert_id}_ep{ep}'
                expert_comp_dims = f"batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
                
                # Each EP group handles experts_per_gpu experts
                gpu_start = ep * tp_size
                gpu_end = (ep + 1) * tp_size
                
                dot.node(expert_node,
                         f'Expert {expert_id}\\nGPU: [{gpu_start}-{gpu_end-1}]\\nInput: [{expert_comp_dims}]\\nOutput: [{expert_comp_dims}]',
                         style='filled', fillcolor='lightsalmon')
                
                # Connect gate to expert with dashed line (routing)
                dot.edge(gate_node, expert_node, style='dashed')
        
        # All-to-all communication for expert routing
        a2a_comm = f'{layer_prefix}_alltoall'
        expert_comp_dims = f"batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden_dim={hidden_dim}"
        dot.node(a2a_comm, f'All-to-All Expert Routing\\nGPU: All\\nInput: [{expert_comp_dims}]\\nOutput: [{expert_comp_dims}]',
                 shape='ellipse', style='filled', fillcolor='lightpink')
        
        # Connect experts to all-to-all
        for ep in range(ep_size):
            for expert in range(experts_per_gpu):
                expert_id = ep * experts_per_gpu + expert
                dot.edge(f'{layer_prefix}_expert{expert_id}_ep{ep}', a2a_comm)
        
        # MoE Output Aggregation
        moe_agg = f'{layer_prefix}_moe_aggregate'
        dot.node(moe_agg, f'MoE Output Aggregation\\nGPU: All\\nInput: [{expert_comp_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(a2a_comm, moe_agg)
        
        # Residual Add for MoE
        moe_residual = f'{layer_prefix}_moe_residual'
        dot.node(moe_residual, f'MoE + Residual\\nGPU: All\\nInput: [{input_dims}], [{input_dims}]\\nOutput: [{input_dims}]',
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(attn_ln, moe_residual, style='dashed')
        dot.edge(moe_agg, moe_residual)
        
        # Layer Norm after MoE
        moe_ln = f'{layer_prefix}_moe_layernorm'
        dot.node(moe_ln, f'Layer Norm (Post-MoE)\\nGPU: All\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
                 style='filled', fillcolor='lightblue')
        dot.edge(moe_residual, moe_ln)
        
        # Set this as the previous node for next layer
        prev_node = moe_ln
    
    # Final output
    dot.node('output', f'Total Output\\nInput: [{input_dims}]\\nOutput: [{input_dims}]',
             shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.edge(prev_node, 'output')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    output_dir = "../outputs/2025-12-01-16-02-27"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    print("Generating 7B MoE Model Deployment DAG...")
    dag = create_model_deployment_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "moe_model_deployment_dag.dot")
    dag.save(dot_file)
    print(f"Saved DOT file: {dot_file}")
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "moe_model_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"Saved SVG image: {svg_file}")
    
    # Also save as PNG for easier viewing
    png_file = os.path.join(output_dir, "moe_model_deployment_dag.png")
    dag.render(png_file.replace('.png', ''), format='png', cleanup=True)
    print(f"Saved PNG image: {png_file}")
    
    # Return the paths
    return {
        "dot_file": dot_file,
        "svg_file": svg_file,
        "png_file": png_file
    }

if __name__ == "__main__":
    files = main()
    print("\nGenerated files:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")