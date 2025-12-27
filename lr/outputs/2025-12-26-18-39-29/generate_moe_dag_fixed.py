#!/usr/bin/env python3

import graphviz

def create_moe_dag():
    """Create a complete MoE parallel strategy DAG with proper connectivity"""
    
    dot = graphviz.Digraph(comment='MoE Parallel Strategy DAG - Complete')
    dot.attr(dpi='300', rankdir='TB', size='30,30')
    dot.attr('node', fontsize='9', margin='0.03')
    
    # Input node
    dot.node('input', 
             'Input\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
             fillcolor='lightblue', shape='ellipse', style='filled')
    
    # Create 8 pipeline stages with 2 layers each
    for stage in range(8):
        with dot.subgraph(name=f'cluster_stage_{stage}') as c:
            c.attr(fillcolor='lightgray', label=f'Pipeline Stage {stage} (Layers {stage*2}-{stage*2+1})', style='rounded,filled')
            
            # Layer 0 in this stage
            layer0 = stage * 2
            create_layer_nodes(c, layer0, stage)
            
            # Layer 1 in this stage  
            layer1 = stage * 2 + 1
            create_layer_nodes(c, layer1, stage)
    
    # Output node
    dot.node('output', 
             'Output\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
             fillcolor='lightblue', shape='ellipse', style='filled')
    
    # Create all edges with proper connectivity
    create_all_edges(dot)
    
    return dot

def create_layer_nodes(subgraph, layer_num, stage):
    """Create all nodes for a single layer"""
    
    # Attention decomposition nodes
    q_proj = f'q_proj_{layer_num}'
    k_proj = f'k_proj_{layer_num}'
    v_proj = f'v_proj_{layer_num}'
    attn_scores = f'attn_scores_{layer_num}'
    softmax = f'softmax_{layer_num}'
    attn_out = f'attn_out_{layer_num}'
    
    # GPU assignment for attention (32 GPUs per attention operation)
    gpu_start = stage * 64
    gpu_range = f'{gpu_start}-{gpu_start+31}'
    
    subgraph.node(q_proj,
                  f'Q Projection L{layer_num}\nGPU: {gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(k_proj,
                  f'K Projection L{layer_num}\nGPU: {gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(v_proj,
                  f'V Projection L{layer_num}\nGPU: {gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(attn_scores,
                  f'Attention Scores L{layer_num}\nGPU: {gpu_range}\nInput: Q,K,V [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, seq_len=10240]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(softmax,
                  f'Softmax L{layer_num}\nGPU: {gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, seq_len=10240]\nOutput: [batch_size=128, seq_len=10240, heads=16, seq_len=10240]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(attn_out,
                  f'Attention Output L{layer_num}\nGPU: {gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=10240, heads=16, d_k=32]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    # Gate node
    gate = f'gate_{layer_num}'
    gate_gpu_range = f'{gpu_start}-{gpu_start+63}'  # 64 GPUs for gate
    subgraph.node(gate,
                  f'Gate L{layer_num}\nGPU: {gate_gpu_range}\nInput: [batch_size=128, seq_len=10240, heads=16, d_k=32]\nOutput: routing_decisions',
                  fillcolor='lightyellow', shape='parallelogram', style='filled')
    
    # Expert nodes (16 experts per layer)
    for expert in range(16):
        create_expert_nodes(subgraph, layer_num, expert, stage)

def create_expert_nodes(subgraph, layer_num, expert_num, stage):
    """Create nodes for a single expert"""
    
    # Route node
    route = f'route_{layer_num}_{expert_num}'
    
    # GPU assignment for experts
    if expert_num < 8:
        gpu_base = stage * 64
    else:
        gpu_base = stage * 64 + 32
    
    gpu_pair = f'{gpu_base + (expert_num % 8) * 2},{gpu_base + (expert_num % 8) * 2 + 1}'
    
    subgraph.node(route,
                  f'Route to Expert {expert_num}\nGPU: {gpu_pair}\nInput: [batch_size=128, seq_len=640, heads=16, d_k=32]\nOutput: [batch_size=128, seq_len=640, heads=16, d_k=32]',
                  fillcolor='lightblue', shape='ellipse', style='dashed,filled')
    
    # Expert TP nodes
    expert_0 = f'expert_{layer_num}_{expert_num}_0'
    expert_1 = f'expert_{layer_num}_{expert_num}_1'
    tp_reduce = f'tp_{layer_num}_{expert_num}'
    
    gpu_0 = gpu_base + (expert_num % 8) * 2
    gpu_1 = gpu_base + (expert_num % 8) * 2 + 1
    
    subgraph.node(expert_0,
                  f'Expert {expert_num} TP-0 L{layer_num}\nGPU: {gpu_0}\nInput: [batch_size=128, seq_len=640, heads=16, d_k=32]\nOutput: expert_out[hidden=512]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(expert_1,
                  f'Expert {expert_num} TP-1 L{layer_num}\nGPU: {gpu_1}\nInput: [batch_size=128, seq_len=640, heads=16, d_k=32]\nOutput: expert_out[hidden=512]',
                  fillcolor='lightgreen', shape='rectangle', style='filled')
    
    subgraph.node(tp_reduce,
                  f'TP All-Reduce Expert {expert_num} L{layer_num}\nGPU: {gpu_0}↔{gpu_1}\nInput: expert_out[hidden=512]\nOutput: expert_full[hidden=1024]',
                  fillcolor='lightblue', shape='ellipse', style='filled')

def create_all_edges(dot):
    """Create all edges with proper connectivity"""
    
    # Input to first layer
    dot.edge('input', 'q_proj_0')
    dot.edge('input', 'k_proj_0')
    dot.edge('input', 'v_proj_0')
    
    # Connect attention nodes within each layer
    for layer in range(16):
        # Attention flow
        dot.edge(f'q_proj_{layer}', f'attn_scores_{layer}')
        dot.edge(f'k_proj_{layer}', f'attn_scores_{layer}')
        dot.edge(f'v_proj_{layer}', f'attn_scores_{layer}')
        dot.edge(f'attn_scores_{layer}', f'softmax_{layer}')
        dot.edge(f'softmax_{layer}', f'attn_out_{layer}')
        dot.edge(f'attn_out_{layer}', f'gate_{layer}')
        
        # Connect gate to all expert routes
        for expert in range(16):
            dot.edge(f'gate_{layer}', f'route_{layer}_{expert}')
            dot.edge(f'route_{layer}_{expert}', f'expert_{layer}_{expert}_0')
            dot.edge(f'route_{layer}_{expert}', f'expert_{layer}_{expert}_1')
            dot.edge(f'expert_{layer}_{expert}_0', f'tp_{layer}_{expert}')
            dot.edge(f'expert_{layer}_{expert}_1', f'tp_{layer}_{expert}')
    
    # Connect layers sequentially
    for layer in range(15):
        # Connect TP outputs to next layer's attention inputs
        for expert in range(16):
            dot.edge(f'tp_{layer}_{expert}', f'q_proj_{layer+1}')
            dot.edge(f'tp_{layer}_{expert}', f'k_proj_{layer+1}')
            dot.edge(f'tp_{layer}_{expert}', f'v_proj_{layer+1}')
    
    # Final layer to output
    for expert in range(16):
        dot.edge(f'tp_15_{expert}', 'output')

if __name__ == '__main__':
    # Generate the DAG
    dag = create_moe_dag()
    
    # Save as DOT file
    dot_file = './outputs/2025-12-26-18-39-29/moe_parallel_strategy.dot'
    dag.save(dot_file)
    
    # Render as SVG
    svg_file = './outputs/2025-12-26-18-39-29/moe_parallel_strategy.svg'
    dag.render(svg_file, format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_file}")
    print(f"SVG rendered to: {svg_file}")