#!/usr/bin/env python3

import graphviz
import os

def generate_llm_deployment_dag():
    """
    Generate a complete DAG for 30B MoE model deployment on 512 GPUs
    with 4 Pipeline Parallelism, 4 Tensor Parallelism, 8 Expert Parallelism, 4 Data Parallelism
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='30,40')  # Top to Bottom layout
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('input', 'Input Embedding\\nGPU: 0-511\\nInput: [batch_size=128, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=1024]')
    
    # Process each pipeline stage (4 stages)
    for pp_stage in range(4):
        with dot.subgraph(name=f'cluster_pp_stage_{pp_stage}') as c:
            c.attr(label=f'Pipeline Stage {pp_stage} (GPUs {pp_stage*128}-{pp_stage*128+127})', 
                   style='rounded,filled', fillcolor='lightcyan')
            
            # Each stage has 4 layers
            for layer in range(4):
                layer_id = pp_stage * 4 + layer
                
                # Create subgraph for each layer
                with c.subgraph(name=f'cluster_layer_{layer_id}') as layer_c:
                    layer_c.attr(label=f'Layer {layer_id}', style='rounded,filled', fillcolor='white')
                    
                    # Attention Block
                    # QKV Projection (Tensor Parallel - Column)
                    for tp_rank in range(4):
                        gpu_id = pp_stage * 128 + tp_rank * 32
                        qkv_node = f'layer_{layer_id}_qkv_tp_{tp_rank}'
                        layer_c.node(qkv_node, 
                                   f'QKV Projection (TP)\\nGPU: {gpu_id}-{gpu_id+31}\\nInput: [batch_size=32, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=32, seq_len=1024, hidden_size=768]')
                    
                    # QKV Communication (AllGather)
                    comm_node = f'layer_{layer_id}_qkv_comm'
                    layer_c.node(comm_node, 
                               f'QKV AllGather\\nGPUs: {pp_stage*128}-{pp_stage*128+127}\\nInput: [batch_size=32, seq_len=1024, hidden_size=768]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=768]',
                               shape='ellipse', fillcolor='lightgreen')
                    
                    # Attention Computation
                    for tp_rank in range(4):
                        gpu_id = pp_stage * 128 + tp_rank * 32
                        attn_node = f'layer_{layer_id}_attn_tp_{tp_rank}'
                        layer_c.node(attn_node, 
                                   f'Attention Computation\\nGPU: {gpu_id}-{gpu_id+31}\\nInput: [batch_size=32, seq_len=1024, hidden_size=192]\\nOutput: [batch_size=32, seq_len=1024, hidden_size=256]')
                    
                    # Attention Output Projection (Tensor Parallel - Row)
                    for tp_rank in range(4):
                        gpu_id = pp_stage * 128 + tp_rank * 32
                        attn_out_node = f'layer_{layer_id}_attn_out_tp_{tp_rank}'
                        layer_c.node(attn_out_node, 
                                   f'Attention Output (TP)\\nGPU: {gpu_id}-{gpu_id+31}\\nInput: [batch_size=32, seq_len=1024, hidden_size=256]\\nOutput: [batch_size=32, seq_len=1024, hidden_size=256]')
                    
                    # Attention AllReduce
                    attn_allreduce = f'layer_{layer_id}_attn_allreduce'
                    layer_c.node(attn_allreduce, 
                               f'Attention AllReduce\\nGPUs: {pp_stage*128}-{pp_stage*128+127}\\nInput: [batch_size=32, seq_len=1024, hidden_size=256]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=1024]',
                               shape='ellipse', fillcolor='lightgreen')
                    
                    # MoE Block
                    # Gate Computation
                    for tp_rank in range(4):
                        gpu_id = pp_stage * 128 + tp_rank * 32
                        gate_node = f'layer_{layer_id}_gate_tp_{tp_rank}'
                        layer_c.node(gate_node, 
                                   f'Gate Computation\\nGPU: {gpu_id}-{gpu_id+31}\\nInput: [batch_size=32, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=32, seq_len=1024, expert_scores=64]')
                    
                    # Expert Routing (dashed line for gate selection)
                    for ep_rank in range(8):
                        gpu_id = pp_stage * 128 + ep_rank * 16
                        route_node = f'layer_{layer_id}_route_ep_{ep_rank}'
                        layer_c.node(route_node, 
                                   f'Expert Routing\\nGPU: {gpu_id}-{gpu_id+15}\\nInput: [batch_size=16, seq_len=1024, expert_ids=2]\\nOutput: [batch_size=16, seq_len=1024, routed_tokens=512]',
                                   shape='parallelogram', fillcolor='lightyellow')
                    
                    # Expert Processing (8 experts per GPU)
                    for ep_rank in range(8):
                        for expert_id in range(8):
                            gpu_id = pp_stage * 128 + ep_rank * 16 + expert_id % 16
                            expert_node = f'layer_{layer_id}_expert_{expert_id}_ep_{ep_rank}'
                            layer_c.node(expert_node, 
                                       f'Expert {expert_id} (EP)\\nGPU: {gpu_id}\\nInput: [batch_size=2, seq_len=512, hidden_size=1024]\\nOutput: [batch_size=2, seq_len=512, hidden_size=1024]')
                    
                    # Expert Aggregation
                    for ep_rank in range(8):
                        gpu_id = pp_stage * 128 + ep_rank * 16
                        agg_node = f'layer_{layer_id}_agg_ep_{ep_rank}'
                        layer_c.node(agg_node, 
                                   f'Expert Aggregation\\nGPU: {gpu_id}-{gpu_id+15}\\nInput: [batch_size=16, seq_len=1024, expert_outputs=8]\\nOutput: [batch_size=16, seq_len=1024, hidden_size=1024]',
                                   shape='parallelogram', fillcolor='lightyellow')
                    
                    # MoE All-to-All Communication
                    moe_comm = f'layer_{layer_id}_moe_alltoall'
                    layer_c.node(moe_comm, 
                               f'MoE All-to-All\\nGPUs: {pp_stage*128}-{pp_stage*128+127}\\nInput: [batch_size=16, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden_size=1024]',
                               shape='ellipse', fillcolor='lightgreen')
                    
                    # Layer Normalization
                    for tp_rank in range(4):
                        gpu_id = pp_stage * 128 + tp_rank * 32
                        ln_node = f'layer_{layer_id}_ln_tp_{tp_rank}'
                        layer_c.node(ln_node, 
                                   f'Layer Norm\\nGPU: {gpu_id}-{gpu_id+31}\\nInput: [batch_size=32, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=32, seq_len=1024, hidden_size=1024]')
    
    # Output node
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer', style='rounded,filled', fillcolor='lightgray')
        c.node('output', 'Output Projection\\nGPU: 0-511\\nInput: [batch_size=128, seq_len=1024, hidden_size=1024]\\nOutput: [batch_size=128, seq_len=1024, vocab_size=32000]')
    
    # Connect nodes with edges
    # Input to first layer
    dot.edge('input', 'layer_0_qkv_tp_0', label='Data Parallel Split')
    dot.edge('input', 'layer_0_qkv_tp_1', label='Data Parallel Split')
    dot.edge('input', 'layer_0_qkv_tp_2', label='Data Parallel Split')
    dot.edge('input', 'layer_0_qkv_tp_3', label='Data Parallel Split')
    
    # Connect within each layer
    for layer_id in range(16):
        # QKV edges
        for tp_rank in range(4):
            dot.edge(f'layer_{layer_id}_qkv_tp_{tp_rank}', 'layer_{layer_id}_qkv_comm')
        
        # Attention edges
        dot.edge('layer_{layer_id}_qkv_comm', 'layer_{layer_id}_attn_tp_0')
        dot.edge('layer_{layer_id}_qkv_comm', 'layer_{layer_id}_attn_tp_1')
        dot.edge('layer_{layer_id}_qkv_comm', 'layer_{layer_id}_attn_tp_2')
        dot.edge('layer_{layer_id}_qkv_comm', 'layer_{layer_id}_attn_tp_3')
        
        for tp_rank in range(4):
            dot.edge(f'layer_{layer_id}_attn_tp_{tp_rank}', f'layer_{layer_id}_attn_out_tp_{tp_rank}')
            dot.edge(f'layer_{layer_id}_attn_out_tp_{tp_rank}', 'layer_{layer_id}_attn_allreduce')
        
        # Gate to routing (dashed line)
        for tp_rank in range(4):
            for ep_rank in range(8):
                dot.edge(f'layer_{layer_id}_gate_tp_{tp_rank}', f'layer_{layer_id}_route_ep_{ep_rank}', 
                        style='dashed', label='Gate Selection')
        
        # Routing to experts
        for ep_rank in range(8):
            for expert_id in range(8):
                dot.edge(f'layer_{layer_id}_route_ep_{ep_rank}', f'layer_{layer_id}_expert_{expert_id}_ep_{ep_rank}')
        
        # Experts to aggregation
        for ep_rank in range(8):
            for expert_id in range(8):
                dot.edge(f'layer_{layer_id}_expert_{expert_id}_ep_{ep_rank}', f'layer_{layer_id}_agg_ep_{ep_rank}')
            dot.edge(f'layer_{layer_id}_agg_ep_{ep_rank}', 'layer_{layer_id}_moe_alltoall')
        
        # Layer norm
        for tp_rank in range(4):
            dot.edge('layer_{layer_id}_attn_allreduce', f'layer_{layer_id}_gate_tp_{tp_rank}')
            dot.edge('layer_{layer_id}_moe_alltoall', f'layer_{layer_id}_ln_tp_{tp_rank}')
    
    # Connect layers
    for layer_id in range(15):
        for tp_rank in range(4):
            dot.edge(f'layer_{layer_id}_ln_tp_{tp_rank}', f'layer_{layer_id+1}_qkv_tp_{tp_rank}')
    
    # Final layer to output
    for tp_rank in range(4):
        dot.edge('layer_15_ln_tp_{tp_rank}', 'output')
    
    return dot

def main():
    # Generate the DAG
    dag = generate_llm_deployment_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-04-17-13-12/llm_deployment_dag.dot'
    dag.save(dot_file_path)
    
    # Render as SVG
    svg_file_path = '../outputs/2025-12-04-17-13-12/llm_deployment_dag.svg'
    dag.render('../outputs/2025-12-04-17-13-12/llm_deployment_dag', format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")
    
    # Return paths for submission
    return {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path
    }

if __name__ == "__main__":
    main()