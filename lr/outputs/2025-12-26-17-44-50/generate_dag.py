#!/usr/bin/env python3

import graphviz

def create_parallel_strategy_dag():
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Parallel Strategy Deployment DAG')
    dot.attr(rank_whole_graph='true')
    dot.attr(rankdir='TB')  # Top to Bottom direction
    
    # Set global node attributes
    dot.attr('node', fontsize='10', fontname='Arial')
    
    # Define node styles
    computation_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    communication_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightyellow'}
    
    # Input node
    dot.node('input', 'Input Embedding\\nGPU: 0-3 (TP Group 0)\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]', 
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Pipeline Stage 0 - Layers 0-7
    for layer in range(8):
        # Attention Layer - Tensor Parallel across GPUs 0-3
        attn_name = f'attn_{layer}'
        attn_label = f'Attention Layer {layer}\\nGPU: {layer%4} (TP Group {layer//4})\\nInput: [batch_size=128, seq_len=10240, heads=4, d_k=64]\\nOutput: [batch_size=128, seq_len=10240, heads=4, d_k=64]'
        dot.node(attn_name, attn_label, **computation_style)
        
        # Communication within TP group
        comm_name = f'comm_attn_{layer}'
        comm_label = f'TP All-Reduce\\nGPU: {layer%4} ↔ GPU: {(layer%4)+1}, {(layer%4)+2}, {(layer%4)+3}\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(comm_name, comm_label, **communication_style)
        
        # Expert Routing - Gate computation
        gate_name = f'gate_{layer}'
        gate_label = f'Expert Gate {layer}\\nGPU: {layer%4} (TP Group {layer//4})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, expert_ids=16]'
        dot.node(gate_name, gate_label, **routing_style)
        
        # Expert Parallelism - 16 experts across 8 GPUs (2 per GPU)
        for expert in range(16):
            gpu_id = 4 + (expert // 2)  # Experts 0-1 on GPU 4, 2-3 on GPU 5, etc.
            expert_name = f'expert_{layer}_{expert}'
            expert_label = f'Expert {expert} Layer {layer}\\nGPU: {gpu_id} (EP Group {layer//2})\\nInput: [batch_size=8, seq_len=10240, hidden=1024]\\nOutput: [batch_size=8, seq_len=10240, hidden=128]'
            dot.node(expert_name, expert_label, **computation_style)
        
        # All-to-All Communication for Expert Routing (dashed line)
        all2all_name = f'all2all_{layer}'
        all2all_label = f'Expert All-to-All\\nGPU: 0-7 (EP Group {layer//2})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(all2all_name, all2all_label, shape='ellipse', style='dashed', fillcolor='lightgreen')
        
        # Expert Aggregation
        agg_name = f'agg_{layer}'
        agg_label = f'Expert Aggregation {layer}\\nGPU: 0-3 (TP Group {layer//4})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(agg_name, agg_label, **routing_style)
    
    # Pipeline Stage 1 - Layers 8-15 (on GPUs 32-63)
    for layer in range(8, 16):
        # Attention Layer - Tensor Parallel across GPUs 32-35
        attn_name = f'attn_{layer}'
        attn_label = f'Attention Layer {layer}\\nGPU: {32+(layer%4)} (TP Group {(layer//4)-2})\\nInput: [batch_size=128, seq_len=10240, heads=4, d_k=64]\\nOutput: [batch_size=128, seq_len=10240, heads=4, d_k=64]'
        dot.node(attn_name, attn_label, **computation_style)
        
        # Communication within TP group
        comm_name = f'comm_attn_{layer}'
        comm_label = f'TP All-Reduce\\nGPU: {32+(layer%4)} ↔ GPU: {33+(layer%4)}, {34+(layer%4)}, {35+(layer%4)}\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(comm_name, comm_label, **communication_style)
        
        # Expert Routing - Gate computation
        gate_name = f'gate_{layer}'
        gate_label = f'Expert Gate {layer}\\nGPU: {32+(layer%4)} (TP Group {(layer//4)-2})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, expert_ids=16]'
        dot.node(gate_name, gate_label, **routing_style)
        
        # Expert Parallelism - 16 experts across 8 GPUs (2 per GPU)
        for expert in range(16):
            gpu_id = 36 + (expert // 2)  # Experts 0-1 on GPU 36, 2-3 on GPU 37, etc.
            expert_name = f'expert_{layer}_{expert}'
            expert_label = f'Expert {expert} Layer {layer}\\nGPU: {gpu_id} (EP Group {(layer//2)-4})\\nInput: [batch_size=8, seq_len=10240, hidden=1024]\\nOutput: [batch_size=8, seq_len=10240, hidden=128]'
            dot.node(expert_name, expert_label, **computation_style)
        
        # All-to-All Communication for Expert Routing (dashed line)
        all2all_name = f'all2all_{layer}'
        all2all_label = f'Expert All-to-All\\nGPU: 32-39 (EP Group {(layer//2)-4})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(all2all_name, all2all_label, shape='ellipse', style='dashed', fillcolor='lightgreen')
        
        # Expert Aggregation
        agg_name = f'agg_{layer}'
        agg_label = f'Expert Aggregation {layer}\\nGPU: {32+(layer%4)} (TP Group {(layer//4)-2})\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, hidden=128]'
        dot.node(agg_name, agg_label, **routing_style)
    
    # Output node
    dot.node('output', 'Output Projection\\nGPU: 32-35 (TP Group 2)\\nInput: [batch_size=128, seq_len=10240, hidden=128]\\nOutput: [batch_size=128, seq_len=10240, vocab_size=50000]', 
             shape='rectangle', style='filled', fillcolor='lightgray')
    
    # Pipeline Communication between stages
    pipe_comm_0 = f'pipe_comm_0'
    pipe_comm_label_0 = f'Pipeline Stage 0→1\\nGPU: 0-3 → GPU: 32-35\\nInput: [batch_size=128, seq_len=10240, hidden=512]\\nOutput: [batch_size=128, seq_len=10240, hidden=512]'
    dot.node(pipe_comm_0, pipe_comm_label_0, **communication_style)
    
    # Connect nodes
    # Input to first layer
    dot.edge('input', 'attn_0')
    
    # Layer 0 connections
    dot.edge('attn_0', 'comm_attn_0')
    dot.edge('comm_attn_0', 'gate_0')
    dot.edge('gate_0', 'all2all_0', style='dashed')  # Gate to all2all with dashed line
    
    # Connect experts for layer 0
    for expert in range(16):
        dot.edge('all2all_0', f'expert_0_{expert}')
        dot.edge(f'expert_0_{expert}', 'agg_0')
    
    # Connect layers within stage 0
    for layer in range(1, 8):
        prev_agg = f'agg_{layer-1}'
        curr_attn = f'attn_{layer}'
        dot.edge(prev_agg, curr_attn)
        
        # Standard layer connections
        dot.edge(curr_attn, f'comm_attn_{layer}')
        dot.edge(f'comm_attn_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'all2all_{layer}', style='dashed')
        
        # Connect experts
        for expert in range(16):
            dot.edge(f'all2all_{layer}', f'expert_{layer}_{expert}')
            dot.edge(f'expert_{layer}_{expert}', f'agg_{layer}')
    
    # Pipeline communication to stage 1
    dot.edge('agg_7', 'pipe_comm_0')
    dot.edge('pipe_comm_0', 'attn_8')
    
    # Connect layers within stage 1
    for layer in range(8, 16):
        if layer > 8:
            prev_agg = f'agg_{layer-1}'
            curr_attn = f'attn_{layer}'
            dot.edge(prev_agg, curr_attn)
        
        # Standard layer connections
        dot.edge(f'attn_{layer}', f'comm_attn_{layer}')
        dot.edge(f'comm_attn_{layer}', f'gate_{layer}')
        dot.edge(f'gate_{layer}', f'all2all_{layer}', style='dashed')
        
        # Connect experts
        for expert in range(16):
            dot.edge(f'all2all_{layer}', f'expert_{layer}_{expert}')
            dot.edge(f'expert_{layer}_{expert}', f'agg_{layer}')
    
    # Final output
    dot.edge('agg_15', 'output')
    
    return dot

if __name__ == "__main__":
    # Create the DAG
    dag = create_parallel_strategy_dag()
    
    # Save the DOT file
    dot_file_path = "./outputs/2025-12-26-17-44-50/parallel_strategy_dag.dot"
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    svg_file_path = "./outputs/2025-12-26-17-44-50/parallel_strategy_dag.svg"
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")