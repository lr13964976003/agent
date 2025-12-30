#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_moe_parallel_dag():
    # Create the DAG
    dot = Digraph(comment='MoE Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input node
    dot.node('input', 'Input\n[batch_size=128, seq_len=var, dim=512]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Output node
    dot.node('output', 'Output\n[batch_size=128, seq_len=var, dim=512]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create subgraphs for each GPU
    for gpu_id in range(16):
        with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as gpu:
            gpu.attr(label=f'GPU {gpu_id}', style='rounded,filled', fillcolor='lightgray')
            
            # Input distribution for this GPU
            gpu.node(f'gpu{gpu_id}_dist', f'GPU {gpu_id} Input Dist\n[batch_size=8, seq_len=var, dim=512]', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Output aggregation for this GPU
            gpu.node(f'gpu{gpu_id}_agg', f'GPU {gpu_id} Output Agg\n[batch_size=8, seq_len=var, dim=512]', 
                    shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            # Process each layer
            for layer in range(16):
                # Attention with TP=4 breakdown
                gpu.node(f'gpu{gpu_id}_layer{layer}_attn_q', f'Layer {layer} Attention Q\nTP=4 Split\n[batch_size=8, seq_len=var, heads=4, d_k=32]', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                gpu.node(f'gpu{gpu_id}_layer{layer}_attn_k', f'Layer {layer} Attention K\nTP=4 Split\n[batch_size=8, seq_len=var, heads=4, d_k=32]', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                gpu.node(f'gpu{gpu_id}_layer{layer}_attn_v', f'Layer {layer} Attention V\nTP=4 Split\n[batch_size=8, seq_len=var, heads=4, d_k=32]', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                gpu.node(f'gpu{gpu_id}_layer{layer}_attn_out', f'Layer {layer} Attention Output\nTP=4 AllReduce\n[batch_size=8, seq_len=var, dim=512]', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Expert Gate (routing)
                gpu.node(f'gpu{gpu_id}_layer{layer}_gate', f'Layer {layer} Expert Gate\nGPU {gpu_id}\n[batch_size=8, seq_len=var, dim=512]', 
                        shape='diamond', style='filled', fillcolor='orange')
                
                # Expert computation (this GPU's expert)
                gpu.node(f'gpu{gpu_id}_layer{layer}_expert', f'Layer {layer} Expert {gpu_id}\nGPU {gpu_id}\n[batch_size=8, seq_len=var, dim=512→512]', 
                        shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Aggregate back to this GPU
                gpu.node(f'gpu{gpu_id}_layer{layer}_agg', f'Layer {layer} Aggregate\nGPU {gpu_id}\n[batch_size=8, seq_len=var, dim=512]', 
                        shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Connect input to all GPUs
    for gpu_id in range(16):
        dot.edge('input', f'gpu{gpu_id}_dist', label='Data Parallel Split\nB=128→8')
    
    # Connect each GPU's processing
    for gpu_id in range(16):
        # Input distribution to first layer
        dot.edge(f'gpu{gpu_id}_dist', f'gpu{gpu_id}_layer0_attn_q')
        dot.edge(f'gpu{gpu_id}_dist', f'gpu{gpu_id}_layer0_attn_k') 
        dot.edge(f'gpu{gpu_id}_dist', f'gpu{gpu_id}_layer0_attn_v')
        
        # Attention computation
        dot.edge(f'gpu{gpu_id}_layer0_attn_q', f'gpu{gpu_id}_layer0_attn_out')
        dot.edge(f'gpu{gpu_id}_layer0_attn_k', f'gpu{gpu_id}_layer0_attn_out')
        dot.edge(f'gpu{gpu_id}_layer0_attn_v', f'gpu{gpu_id}_layer0_attn_out')
        
        # Attention to gate
        dot.edge(f'gpu{gpu_id}_layer0_attn_out', f'gpu{gpu_id}_layer0_gate')
        
        # Gate to expert (with potential cross-GPU routing)
        dot.edge(f'gpu{gpu_id}_layer0_gate', f'gpu{gpu_id}_layer0_expert', 
                label=f'Route to Expert {gpu_id}', style='solid')
        
        # Expert to aggregate
        dot.edge(f'gpu{gpu_id}_layer0_expert', f'gpu{gpu_id}_layer0_agg')
        
        # Connect layers (simplified - showing key connections)
        for layer in range(1, 16):
            prev_layer = layer - 1
            dot.edge(f'gpu{gpu_id}_layer{prev_layer}_agg', f'gpu{gpu_id}_layer{layer}_attn_q')
            dot.edge(f'gpu{gpu_id}_layer{prev_layer}_agg', f'gpu{gpu_id}_layer{layer}_attn_k')
            dot.edge(f'gpu{gpu_id}_layer{prev_layer}_agg', f'gpu{gpu_id}_layer{layer}_attn_v')
            
            dot.edge(f'gpu{gpu_id}_layer{layer}_attn_q', f'gpu{gpu_id}_layer{layer}_attn_out')
            dot.edge(f'gpu{gpu_id}_layer{layer}_attn_k', f'gpu{gpu_id}_layer{layer}_attn_out')
            dot.edge(f'gpu{gpu_id}_layer{layer}_attn_v', f'gpu{gpu_id}_layer{layer}_attn_out')
            
            dot.edge(f'gpu{gpu_id}_layer{layer}_attn_out', f'gpu{gpu_id}_layer{layer}_gate')
            dot.edge(f'gpu{gpu_id}_layer{layer}_gate', f'gpu{gpu_id}_layer{layer}_expert')
            dot.edge(f'gpu{gpu_id}_layer{layer}_expert', f'gpu{gpu_id}_layer{layer}_agg')
        
        # Final layer to output aggregation
        dot.edge(f'gpu{gpu_id}_layer15_agg', f'gpu{gpu_id}_agg')
        dot.edge(f'gpu{gpu_id}_agg', 'output', label='Data Parallel Merge\nB=8→128')
    
    # Add cross-GPU expert routing (dashed lines)
    # This shows that tokens can be routed to experts on different GPUs
    for src_gpu in range(16):
        for dst_gpu in range(16):
            if src_gpu != dst_gpu:
                # Show potential routing from any GPU to any expert
                for layer in range(16):
                    dot.edge(f'gpu{src_gpu}_layer{layer}_gate', f'gpu{dst_gpu}_layer{layer}_expert', 
                            label=f'Route to Expert {dst_gpu}', 
                            style='dashed', color='red', constraint='false')
    
    # Add TP=4 communication within attention (ellipses for communication)
    for gpu_id in range(16):
        for layer in range(16):
            # TP=4 AllReduce communication
            dot.node(f'comm_gpu{gpu_id}_layer{layer}_attn', f'TP=4 AllReduce\nGPU {gpu_id} Layer {layer}\nAttention', 
                    shape='ellipse', style='filled', fillcolor='lightblue')
            dot.edge(f'gpu{gpu_id}_layer{layer}_attn_out', f'comm_gpu{gpu_id}_layer{layer}_attn', 
                    label='TP=4 Communication', style='dotted')
            dot.edge(f'comm_gpu{gpu_id}_layer{layer}_attn', f'gpu{gpu_id}_layer{layer}_gate')
    
    return dot

if __name__ == '__main__':
    dag = create_moe_parallel_dag()
    
    # Save as DOT file
    dag.save('./outputs/2025-12-30-11-08-23/moe_parallel_strategy_detailed.dot')
    
    # Render as SVG
    dag.render('./outputs/2025-12-30-11-08-23/moe_parallel_strategy_detailed', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files saved:")
    print("- moe_parallel_strategy_detailed.dot")
    print("- moe_parallel_strategy_detailed.svg")