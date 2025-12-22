#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def create_moe_deployment_dag():
    """
    Generate a comprehensive DAG for 30B MoE model deployment with hybrid parallelism
    Total GPUs: 2048 (EP=64 × TP=8 × PP=2 × DP=2)
    """
    
    # Create the main graph
    dot = Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='40,60', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    dot.attr('node', shape='ellipse', style='filled,dashed', fillcolor='pink')  # Communication
    
    # Global parameters
    batch_size = 128
    seq_len = "128-10240"
    hidden_dim = 1024
    num_heads = 16
    head_dim = 64
    experts_per_layer = 64
    num_layers = 16
    
    # GPU topology: PP=2 stages, each stage has TP=8 groups, each TP group has EP=64 experts, DP=2 replicas
    # Total: 2 × 8 × 64 × 2 = 2048 GPUs
    
    # Let's create a simplified but comprehensive view focusing on one pipeline stage
    # and showing the key parallel patterns
    
    with dot.subgraph(name='cluster_pipeline_stage_0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-7)', style='rounded,filled', fillcolor='lightgray', fontsize='14')
        
        # Input node
        input_attrs = f'INPUT DIMENSION: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}]'
        c.node('input', f'Input\\n{input_attrs}', shape='ellipse', fillcolor='lightblue')
        
        # Layer 0 - Attention with TP=8
        with c.subgraph(name='cluster_layer0_attention') as layer0:
            layer0.attr(label='Layer 0 - Multi-Head Attention (TP=8)', style='rounded', fontsize='12')
            
            # QKV Projection - Column parallel across 8 GPUs
            for tp_rank in range(8):
                gpu_id = f'PP0_TP{tp_rank}_GPU'
                qkv_attrs = f'INPUT: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOUTPUT: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]'
                layer0.node(f'layer0_qkv_tp{tp_rank}', f'QKV Projection\\nGPU: {gpu_id}\\n{qkv_attrs}', 
                           shape='box', fillcolor='lightgreen')
            
            # Attention Score Computation
            for tp_rank in range(8):
                gpu_id = f'PP0_TP{tp_rank}_GPU'
                attn_attrs = f'INPUT: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOUTPUT: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]'
                layer0.node(f'layer0_attn_tp{tp_rank}', f'Attention Score\\nGPU: {gpu_id}\\n{attn_attrs}', 
                           shape='box', fillcolor='lightgreen')
            
            # Attention Output - Row parallel
            for tp_rank in range(8):
                gpu_id = f'PP0_TP{tp_rank}_GPU'
                out_attrs = f'INPUT: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads//8}, d_k={head_dim}]\\nOUTPUT: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim//8}]'
                layer0.node(f'layer0_out_tp{tp_rank}', f'Attention Output\\nGPU: {gpu_id}\\n{out_attrs}', 
                           shape='box', fillcolor='lightgreen')
            
            # All-Reduce for attention output
            layer0.node('layer0_allreduce', f'All-Reduce Attention\\nTP Group All-Reduce\\n8 GPUs',                        shape='ellipse', fillcolor='pink', style='filled,dashed')
        
        # MoE Layer with EP=64
        with c.subgraph(name='cluster_layer0_moe') as moe0:
            moe0.attr(label='Layer 0 - MoE (EP=64)', style='rounded', fontsize='12')
            
            # Router - determines which experts to use
            router_attrs = f'INPUT: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOUTPUT: [batch_size={batch_size//2}, seq_len={seq_len}, top_k=2]'
            moe0.node('layer0_router', f'Router\\nTop-K Expert Selection\\nGPU: PP0_TP0_GPU\\n{router_attrs}', 
                     shape='parallelogram', fillcolor='lightyellow')
            
            # Expert dispatch - All-to-All communication
            dispatch_attrs = f'Dispatch tokens to 64 experts\\nAll-to-All communication\\n64 GPUs involved'
            moe0.node('layer0_dispatch', f'Expert Dispatch\\n{dispatch_attrs}', 
                     shape='ellipse', fillcolor='pink', style='filled,dashed')
            
            # Expert computations (showing first few as examples)
            for expert_id in range(4):  # Show first 4 experts as examples
                gpu_id = f'PP0_EP{expert_id}_GPU'
                expert_attrs = f'INPUT: [batch_size=~{batch_size//(2*64)}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOUTPUT: [batch_size=~{batch_size//(2*64)}, seq_len={seq_len}, hidden_dim={hidden_dim}]'
                moe0.node(f'layer0_expert{expert_id}', f'Expert {expert_id}\\nGPU: {gpu_id}\\n{expert_attrs}', 
                         shape='box', fillcolor='lightgreen')
            
            # Show ellipsis for remaining experts
            moe0.node('layer0_experts_ellipsis', f'...\\n60 more experts\\n distributed across\\n60 GPUs', 
                     shape='box', style='dashed', fillcolor='lightgray')
            
            # Expert combine - All-to-All communication
            combine_attrs = f'Combine expert outputs\\nAll-to-All communication\\n64 GPUs involved'
            moe0.node('layer0_combine', f'Expert Combine\\n{combine_attrs}', 
                     shape='ellipse', fillcolor='pink', style='filled,dashed')
        
        # Layer norm and residual connections
        layernorm_attrs = f'INPUT: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}]\\nOUTPUT: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}]'
        c.node('layer0_norm', f'Layer Normalization\\nGPU: PP0_TP0_GPU\\n{layernorm_attrs}', 
               shape='box', fillcolor='lightgreen')
        
        # Pipeline communication to next stage
        pipeline_attrs = f'Send activations to Stage 1\\nPipeline communication\\nGPU-to-GPU transfer'
        c.node('pipeline_send', f'Pipeline Send\\n{pipeline_attrs}', 
               shape='ellipse', fillcolor='orange', style='filled,dashed')
    
    # Pipeline Stage 1 (simplified representation)
    with dot.subgraph(name='cluster_pipeline_stage_1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 8-15)', style='rounded,filled', fillcolor='lightgray', fontsize='14')
        
        # Pipeline receive from stage 0
        receive_attrs = f'Receive activations from Stage 0\\nPipeline communication\\nGPU-to-GPU transfer'
        c.node('pipeline_receive', f'Pipeline Receive\\n{receive_attrs}', 
               shape='ellipse', fillcolor='orange', style='filled,dashed')
        
        # Similar structure for layers 8-15
        c.node('stage1_processing', f'Stage 1 Processing\\nLayers 8-15\\nSimilar TP+EP pattern\\n8 TP GPUs + 64 EP GPUs', 
               shape='box', fillcolor='lightgreen')
        
        # Final output
        output_attrs = f'OUTPUT DIMENSION: [batch_size={batch_size//2}, seq_len={seq_len}, hidden_dim={hidden_dim}, vocab_size=32000]'
        c.node('output', f'Final Output\\n{output_attrs}', shape='ellipse', fillcolor='lightcoral')
    
    # Data Parallelism representation
    with dot.subgraph(name='cluster_data_parallel') as dp:
        dp.attr(label='Data Parallelism (DP=2)', style='dashed,rounded', fontsize='12')
        dp.node('dp_replica', f'Data Parallel Replica\\nBatch {batch_size//2} sequences\\nIdentical computation graph\\nGradient sync in training', 
                shape='parallelogram', fillcolor='lightsteelblue')
    
    # Define edges (dependencies)
    # Input to first layer
    dot.edge('input', 'layer0_qkv_tp0')
    
    # Within attention layer
    for tp_rank in range(8):
        dot.edge(f'layer0_qkv_tp{tp_rank}', f'layer0_attn_tp{tp_rank}')
        dot.edge(f'layer0_attn_tp{tp_rank}', f'layer0_out_tp{tp_rank}')
        if tp_rank > 0:
            dot.edge(f'layer0_out_tp{tp_rank-1}', f'layer0_out_tp{tp_rank}')
    
    # All-Reduce after attention
    dot.edge('layer0_out_tp7', 'layer0_allreduce')
    
    # Attention to router
    dot.edge('layer0_allreduce', 'layer0_router')
    
    # Router to dispatch
    dot.edge('layer0_router', 'layer0_dispatch', style='dashed', label='expert selection')
    
    # Dispatch to experts
    for expert_id in range(4):
        dot.edge('layer0_dispatch', f'layer0_expert{expert_id}')
    dot.edge('layer0_dispatch', 'layer0_experts_ellipsis')
    
    # Experts to combine
    for expert_id in range(4):
        dot.edge(f'layer0_expert{expert_id}', 'layer0_combine')
    dot.edge('layer0_experts_ellipsis', 'layer0_combine')
    
    # Combine to layer norm
    dot.edge('layer0_combine', 'layer0_norm')
    
    # Layer norm to pipeline send
    dot.edge('layer0_norm', 'pipeline_send')
    
    # Pipeline communication
    dot.edge('pipeline_send', 'pipeline_receive')
    dot.edge('pipeline_receive', 'stage1_processing')
    
    # Final output
    dot.edge('stage1_processing', 'output')
    
    # Data parallelism connection
    dot.edge('input', 'dp_replica', style='dotted', label='DP replica')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    output_dir = "../outputs/2025-12-22-09-46-42"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the DAG
    dag = create_moe_deployment_dag()
    
    # Save as DOT file
    dot_file = os.path.join(output_dir, "moe_deployment_dag.dot")
    dag.save(dot_file)
    
    # Render as SVG
    svg_file = os.path.join(output_dir, "moe_deployment_dag.svg")
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    return {
        "dot_file": dot_file,
        "svg_file": svg_file
    }

if __name__ == "__main__":
    main()