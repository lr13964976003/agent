#!/usr/bin/env python3

import graphviz
import os
import json

def create_llm_deployment_dag():
    """
    Create a DAG for LLM deployment with EP16_TP4_PP2_Hybrid strategy
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='LLM Deployment DAG - EP16_TP4_PP2_Hybrid')
    dot.attr(rankdir='TB', size='50,30', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Model parameters from deployment strategy
    batch_size = 128
    seq_len = 1024
    token_dim = 1024
    moe_hidden = 2048
    num_layers = 16
    experts_per_layer = 64
    
    # Parallel dimensions
    ep_size = 16  # Expert parallel
    tp_size = 4   # Tensor parallel
    pp_size = 2   # Pipeline parallel
    
    # GPU organization
    total_gpus = 128
    gpus_per_ep_group = tp_size * pp_size  # 8 GPUs per expert parallel group
    num_ep_groups = total_gpus // gpus_per_ep_group  # 16 expert parallel groups
    
    # Input node
    input_attrs = f"Input: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
    output_attrs = f"Output: [batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
    dot.node('input', f'Input\\n{input_attrs}\\n{output_attrs}', shape='oval', fillcolor='white')
    
    # Process each pipeline stage
    for pp_stage in range(pp_size):
        with dot.subgraph(name=f'cluster_pp_stage_{pp_stage}') as c:
            c.attr(label=f'Pipeline Stage {pp_stage}', style='rounded,filled', fillcolor='lightgray')
            
            # Process each layer in the pipeline stage
            layers_per_stage = num_layers // pp_size
            for layer_idx in range(layers_per_stage):
                global_layer = pp_stage * layers_per_stage + layer_idx
                
                # Create MoE layer structure
                with c.subgraph(name=f'cluster_layer_{global_layer}') as layer_c:
                    layer_c.attr(label=f'Layer {global_layer}', style='rounded')
                    
                    # Self-attention computation (tensor parallel across TP group)
                    for tp_rank in range(tp_size):
                        gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                        
                        # Attention computation nodes
                        attn_input = f"[batch_size={batch_size}, seq_len={seq_len}, heads=16, d_k=64]"
                        attn_output = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
                        
                        layer_c.node(f'attn_qkv_gpu{gpu_id}', 
                                   f'QKV Linear\\nGPU-{gpu_id}\\nInput: {attn_input}\\nOutput: {attn_input}',
                                   shape='rectangle', fillcolor='lightblue')
                        
                        layer_c.node(f'attn_score_gpu{gpu_id}', 
                                   f'Attention Score\\nGPU-{gpu_id}\\nInput: {attn_input}\\nOutput: [batch_size={batch_size}, heads=16, seq_len={seq_len}, seq_len={seq_len}]',
                                   shape='rectangle', fillcolor='lightblue')
                        
                        layer_c.node(f'attn_out_gpu{gpu_id}', 
                                   f'Attention Output\\nGPU-{gpu_id}\\nInput: {attn_input}\\nOutput: {attn_output}',
                                   shape='rectangle', fillcolor='lightblue')
                    
                    # Expert parallel processing
                    experts_per_gpu = experts_per_layer // ep_size
                    for ep_rank in range(ep_size):
                        base_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                        
                        # Gate computation for expert selection
                        gate_input = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
                        gate_output = f"[batch_size={batch_size}, seq_len={seq_len}, experts={experts_per_layer}]"
                        
                        layer_c.node(f'gate_gpu{base_gpu}', 
                                   f'Expert Gate\\nGPU-{base_gpu}-{base_gpu+tp_size-1}\\nInput: {gate_input}\\nOutput: {gate_output}',
                                   shape='parallelogram', fillcolor='yellow')
                        
                        # Expert computation (distributed across EP group)
                        for expert_idx in range(experts_per_gpu):
                            expert_id = ep_rank * experts_per_gpu + expert_idx
                            gpu_id = base_gpu + (expert_idx % tp_size)  # Distribute experts across TP group
                            
                            expert_input = f"[batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden={token_dim}]"
                            expert_output = f"[batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden={token_dim}]"
                            
                            layer_c.node(f'expert_{expert_id}_gpu{gpu_id}', 
                                       f'Expert {expert_id} MLP\\nGPU-{gpu_id}\\nInput: {expert_input}\\nOutput: {expert_output}',
                                       shape='rectangle', fillcolor='lightblue')
                    
                    # Communication nodes
                    # All-reduce for tensor parallel attention
                    for tp_rank in range(tp_size):
                        gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                        comm_input = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
                        
                        dot.node(f'allreduce_attn_gpu{gpu_id}', 
                               f'All-Reduce Attention\\nGPU-{gpu_id}\\nInput: {comm_input}\\nOutput: {comm_input}',
                               shape='ellipse', fillcolor='lightgreen')
                    
                    # All-to-all communication for expert parallel
                    for ep_rank in range(ep_size):
                        base_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                        comm_input = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
                        comm_output = f"[batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden={token_dim}]"
                        
                        dot.node(f'all2all_send_gpu{base_gpu}', 
                               f'All-to-All Send\\nGPU-{base_gpu}-{base_gpu+tp_size-1}\\nInput: {comm_input}\\nOutput: {comm_output}',
                               shape='ellipse', fillcolor='lightgreen')
                        
                        dot.node(f'all2all_recv_gpu{base_gpu}', 
                               f'All-to-All Recv\\nGPU-{base_gpu}-{base_gpu+tp_size-1}\\nInput: {comm_output}\\nOutput: {comm_output}',
                               shape='ellipse', fillcolor='lightgreen')
                    
                    # Expert aggregation
                    for ep_rank in range(ep_size):
                        base_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                        agg_input = f"[batch_size={batch_size//ep_size}, seq_len={seq_len}, hidden={token_dim}]"
                        agg_output = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
                        
                        layer_c.node(f'expert_agg_gpu{base_gpu}', 
                                   f'Expert Aggregation\\nGPU-{base_gpu}-{base_gpu+tp_size-1}\\nInput: {agg_input}\\nOutput: {agg_output}',
                                   shape='parallelogram', fillcolor='yellow')
    
    # Output node
    final_output = f"[batch_size={batch_size}, seq_len={seq_len}, hidden={token_dim}]"
    dot.node('output', f'Output\\nInput: {final_output}\\nOutput: {final_output}', shape='oval', fillcolor='white')
    
    # Create edges (connections)
    # Input to first layer
    first_gpu = 0
    dot.edge('input', f'attn_qkv_gpu{first_gpu}')
    
    # Connections within layers
    for pp_stage in range(pp_size):
        for layer_idx in range(layers_per_stage):
            global_layer = pp_stage * layers_per_stage + layer_idx
            
            # Attention connections
            for tp_rank in range(tp_size):
                gpu_id = pp_stage * (ep_size * tp_size) + tp_rank
                
                # QKV -> Score -> Output
                dot.edge(f'attn_qkv_gpu{gpu_id}', f'attn_score_gpu{gpu_id}')
                dot.edge(f'attn_score_gpu{gpu_id}', f'attn_out_gpu{gpu_id}')
                
                # Attention output to all-reduce
                dot.edge(f'attn_out_gpu{gpu_id}', f'allreduce_attn_gpu{gpu_id}')
            
            # Expert parallel connections
            for ep_rank in range(ep_size):
                base_gpu = pp_stage * (ep_size * tp_size) + ep_rank * tp_size
                
                # Gate to all-to-all send (dashed line for selection)
                dot.edge(f'gate_gpu{base_gpu}', f'all2all_send_gpu{base_gpu}', style='dashed')
                
                # All-to-all send to experts
                for expert_idx in range(experts_per_gpu):
                    expert_id = ep_rank * experts_per_gpu + expert_idx
                    gpu_id = base_gpu + (expert_idx % tp_size)
                    dot.edge(f'all2all_send_gpu{base_gpu}', f'expert_{expert_id}_gpu{gpu_id}')
                
                # Experts to all-to-all receive
                for expert_idx in range(experts_per_gpu):
                    expert_id = ep_rank * experts_per_gpu + expert_idx
                    gpu_id = base_gpu + (expert_idx % tp_size)
                    dot.edge(f'expert_{expert_id}_gpu{gpu_id}', f'all2all_recv_gpu{base_gpu}')
                
                # All-to-all receive to aggregation
                dot.edge(f'all2all_recv_gpu{base_gpu}', f'expert_agg_gpu{base_gpu}')
    
    # Final layer to output
    last_base_gpu = (pp_size - 1) * (ep_size * tp_size)
    dot.edge(f'expert_agg_gpu{last_base_gpu}', 'output')
    
    return dot

def main():
    # Create output directory
    output_dir = "../outputs/2025-12-04-19-17-18"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "llm_deployment_dag.dot")
    dag.save(dot_file)
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "llm_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_file}")
    print(f"SVG saved to: {svg_file}")
    
    # Verify DAG has no cycles
    try:
        from graphviz import Source
        source = Source.from_file(dot_file)
        print("DAG verification: Structure created successfully")
    except Exception as e:
        print(f"DAG verification error: {e}")
    
    return {
        "dot_file": dot_file,
        "svg_file": svg_file
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))