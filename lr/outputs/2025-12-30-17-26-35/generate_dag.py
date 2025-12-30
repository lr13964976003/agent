#!/usr/bin/env python3

import graphviz

def create_parallelism_dag():
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Qwen3-235B Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='rectangle', fillcolor='lightgray')
    
    # Pipeline Stage 0 (Layers 0-23)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0: Layers 0-23\\nGPUs: 0-31', style='rounded', fillcolor='lightcoral')
        
        # Token embedding and initial processing
        stage0.node('embed_0', 'Token Embedding\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle')
        
        # Gate selection (dashed line process)
        stage0.node('gate_0', 'Gate Selection\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Expert routing communication
        stage0.node('route_0', 'Expert Routing\\nGPU: 0-31\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        # Expert computation (experts 0-31 distributed across GPUs 0-31)
        for i in range(4):  # 4 TP groups per stage
            for j in range(8):  # 8 GPUs per TP group
                gpu_id = i * 8 + j
                stage0.node(f'expert_{gpu_id}', f'Expert {gpu_id} Computation\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]', 
                           shape='rectangle')
        
        # Attention with TP+SP
        stage0.node('attn_0', 'Attention (TP+SP)\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                   shape='rectangle')
        
        # Communication for attention
        stage0.node('attn_comm_0', 'Attention All-Reduce\\nGPU: 0-7\\nInput: [batch_size=16, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                     shape='ellipse', fillcolor='lightgreen')
        
        # Expert aggregation
        stage0.node('agg_0', 'Expert Aggregation\\nGPU: 0-7\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline Stage 1 (Layers 24-47)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1: Layers 24-47\\nGPUs: 32-63', style='rounded', fillcolor='lightsteelblue')
        
        stage1.node('embed_1', 'Token Embedding\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle')
        
        stage1.node('gate_1', 'Gate Selection\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage1.node('route_1', 'Expert Routing\\nGPU: 32-63\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        for i in range(4):
            for j in range(8):
                gpu_id = 32 + i * 8 + j
                stage1.node(f'expert_{gpu_id}', f'Expert {gpu_id} Computation\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]', 
                           shape='rectangle')
        
        stage1.node('attn_1', 'Attention (TP+SP)\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                   shape='rectangle')
        
        stage1.node('attn_comm_1', 'Attention All-Reduce\\nGPU: 32-39\\nInput: [batch_size=16, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                     shape='ellipse', fillcolor='lightgreen')
        
        stage1.node('agg_1', 'Expert Aggregation\\nGPU: 32-39\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline Stage 2 (Layers 48-71)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Pipeline Stage 2: Layers 48-71\\nGPUs: 64-95', style='rounded', fillcolor='lightseagreen')
        
        stage2.node('embed_2', 'Token Embedding\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle')
        
        stage2.node('gate_2', 'Gate Selection\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage2.node('route_2', 'Expert Routing\\nGPU: 64-95\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        for i in range(4):
            for j in range(8):
                gpu_id = 64 + i * 8 + j
                stage2.node(f'expert_{gpu_id}', f'Expert {gpu_id} Computation\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]', 
                           shape='rectangle')
        
        stage2.node('attn_2', 'Attention (TP+SP)\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                   shape='rectangle')
        
        stage2.node('attn_comm_2', 'Attention All-Reduce\\nGPU: 64-71\\nInput: [batch_size=16, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                     shape='ellipse', fillcolor='lightgreen')
        
        stage2.node('agg_2', 'Expert Aggregation\\nGPU: 64-71\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline Stage 3 (Layers 72-93)
    with dot.subgraph(name='cluster_stage3') as stage3:
        stage3.attr(label='Pipeline Stage 3: Layers 72-93\\nGPUs: 96-127', style='rounded', fillcolor='lightsalmon')
        
        stage3.node('embed_3', 'Token Embedding\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='rectangle')
        
        stage3.node('gate_3', 'Gate Selection\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage3.node('route_3', 'Expert Routing\\nGPU: 96-127\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        for i in range(4):
            for j in range(8):
                gpu_id = 96 + i * 8 + j
                stage3.node(f'expert_{gpu_id}', f'Expert {gpu_id} Computation\\nGPU: {gpu_id}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]', 
                           shape='rectangle')
        
        stage3.node('attn_3', 'Attention (TP+SP)\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                   shape='rectangle')
        
        stage3.node('attn_comm_3', 'Attention All-Reduce\\nGPU: 96-103\\nInput: [batch_size=16, seq_len=1024, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]', 
                     shape='ellipse', fillcolor='lightgreen')
        
        stage3.node('agg_3', 'Expert Aggregation\\nGPU: 96-103\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, vocab_size=128000]', 
             shape='rectangle', fillcolor='lightgray')
    
    # Pipeline communication between stages
    dot.node('pipe_comm_0_1', 'Pipeline Communication\\nGPU: 31->32\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='ellipse', fillcolor='lightgreen')
    
    dot.node('pipe_comm_1_2', 'Pipeline Communication\\nGPU: 63->64\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='ellipse', fillcolor='lightgreen')
    
    dot.node('pipe_comm_2_3', 'Pipeline Communication\\nGPU: 95->96\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Define edges (connections)
    # Input to first stage
    dot.edge('input', 'embed_0')
    dot.edge('embed_0', 'gate_0')
    dot.edge('gate_0', 'route_0', style='dashed')  # Gate selection with dashed line
    dot.edge('route_0', 'expert_0')
    dot.edge('route_0', 'expert_1')
    # ... (connect to all experts in stage 0)
    for i in range(32):
        dot.edge('route_0', f'expert_{i}')
        if i < 8:  # Connect to attention TP group
            dot.edge(f'expert_{i}', 'attn_0')
    
    dot.edge('attn_0', 'attn_comm_0')
    dot.edge('attn_comm_0', 'agg_0')
    
    # Connect experts to aggregation
    for i in range(8):
        dot.edge(f'expert_{i}', 'agg_0')
    
    # Pipeline stage connections
    dot.edge('agg_0', 'pipe_comm_0_1')
    dot.edge('pipe_comm_0_1', 'embed_1')
    
    # Stage 1
    dot.edge('embed_1', 'gate_1')
    dot.edge('gate_1', 'route_1', style='dashed')
    for i in range(32, 64):
        dot.edge('route_1', f'expert_{i}')
        if i < 40:
            dot.edge(f'expert_{i}', 'attn_1')
    
    dot.edge('attn_1', 'attn_comm_1')
    dot.edge('attn_comm_1', 'agg_1')
    
    for i in range(32, 40):
        dot.edge(f'expert_{i}', 'agg_1')
    
    dot.edge('agg_1', 'pipe_comm_1_2')
    dot.edge('pipe_comm_1_2', 'embed_2')
    
    # Stage 2
    dot.edge('embed_2', 'gate_2')
    dot.edge('gate_2', 'route_2', style='dashed')
    for i in range(64, 96):
        dot.edge('route_2', f'expert_{i}')
        if i < 72:
            dot.edge(f'expert_{i}', 'attn_2')
    
    dot.edge('attn_2', 'attn_comm_2')
    dot.edge('attn_comm_2', 'agg_2')
    
    for i in range(64, 72):
        dot.edge(f'expert_{i}', 'agg_2')
    
    dot.edge('agg_2', 'pipe_comm_2_3')
    dot.edge('pipe_comm_2_3', 'embed_3')
    
    # Stage 3
    dot.edge('embed_3', 'gate_3')
    dot.edge('gate_3', 'route_3', style='dashed')
    for i in range(96, 128):
        dot.edge('route_3', f'expert_{i}')
        if i < 104:
            dot.edge(f'expert_{i}', 'attn_3')
    
    dot.edge('attn_3', 'attn_comm_3')
    dot.edge('attn_comm_3', 'agg_3')
    
    for i in range(96, 104):
        dot.edge(f'expert_{i}', 'agg_3')
    
    dot.edge('agg_3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_parallelism_dag()
    
    # Save the DOT file
    dag.save('./outputs/2025-12-30-17-26-35/parallel_strategy_dag.dot')
    
    # Render to SVG
    dag.render('./outputs/2025-12-30-17-26-35/parallel_strategy_dag', format='svg', cleanup=True)
    
    print("DAG generated successfully!")
    print("Files saved:")
    print("- parallel_strategy_dag.dot")
    print("- parallel_strategy_dag.svg")