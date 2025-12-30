#!/usr/bin/env python3

import graphviz

def create_corrected_dag():
    # Create the DAG
    dot = graphviz.Digraph(comment='Qwen3-235B Parallel Strategy Deployment DAG - Corrected')
    
    # Set graph attributes
    dot.attr('graph', 
             comment='Qwen3-235B Parallel Strategy Deployment DAG - Corrected',
             dpi='300',
             rankdir='TB',
             size='30,40')
    
    # Set node attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define colors
    colors = {
        'stage0': 'lightcoral',
        'stage1': 'lightsteelblue', 
        'stage2': 'lightseagreen',
        'stage3': 'lightsalmon',
        'compute': 'lightblue',
        'comm': 'lightgreen',
        'route': 'lightyellow',
        'io': 'lightgray'
    }
    
    # Stage 0: Layers 0-23 (GPUs 0-31)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0: Layers 0-23\\nGPUs: 0-31', 
                   style='rounded', fillcolor=colors['stage0'])
        
        # Token Embedding
        stage0.node('embed_0', 
                   'Token Embedding\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Gate Selection
        stage0.node('gate_0', 
                   'Gate Selection\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
        
        # Expert Routing
        stage0.node('route_0', 
                   'Expert Routing\\nGPU: 0-31\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Expert Computations (8 experts per stage)
        for i in range(8):
            stage0.node(f'expert_{i}', 
                       f'Expert {i} Computation\\nGPU: {i}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                       shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Block - Broken down into submodules
        # QKV Projection with TP
        stage0.node('qkv_proj_0', 
                   'QKV Projection (TP=8)\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=16, seq_len=2048, heads=8, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # QKV Communication - AllGather for TP
        stage0.node('qkv_comm_0', 
                   'QKV AllGather (TP)\\nGPU: 0-7\\nInput: [batch_size=16, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Attention Score Computation with SP
        stage0.node('attn_scores_0', 
                   'Attention Score Computation (SP=2)\\nGPU: 0-7\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=64, seq_len=1024, seq_len=1024]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Weight Application with SP
        stage0.node('attn_apply_0', 
                   'Attention Weight Application (SP=2)\\nGPU: 0-7\\nInput: [batch_size=64, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=64, seq_len=1024, heads=64, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Output Communication - AllReduce for SP
        stage0.node('attn_sp_comm_0', 
                   'Attention SP AllReduce\\nGPU: 0-7\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Output Projection with TP
        stage0.node('out_proj_0', 
                   'Output Projection (TP=8)\\nGPU: 0-7\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Output Projection AllReduce for TP
        stage0.node('out_proj_comm_0', 
                   'Output AllReduce (TP)\\nGPU: 0-7\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Expert Aggregation
        stage0.node('agg_0', 
                   'Expert Aggregation\\nGPU: 0-7\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
    
    # Stage 1: Layers 24-47 (GPUs 32-63)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1: Layers 24-47\\nGPUs: 32-63', 
                   style='rounded', fillcolor=colors['stage1'])
        
        # Token Embedding
        stage1.node('embed_1', 
                   'Token Embedding\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Gate Selection
        stage1.node('gate_1', 
                   'Gate Selection\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
        
        # Expert Routing
        stage1.node('route_1', 
                   'Expert Routing\\nGPU: 32-63\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Expert Computations (8 experts per stage)
        for i in range(8):
            stage1.node(f'expert_{32+i}', 
                       f'Expert {32+i} Computation\\nGPU: {32+i}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                       shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Block - Broken down into submodules
        stage1.node('qkv_proj_1', 
                   'QKV Projection (TP=8)\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=16, seq_len=2048, heads=8, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage1.node('qkv_comm_1', 
                   'QKV AllGather (TP)\\nGPU: 32-39\\nInput: [batch_size=16, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage1.node('attn_scores_1', 
                   'Attention Score Computation (SP=2)\\nGPU: 32-39\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=64, seq_len=1024, seq_len=1024]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage1.node('attn_apply_1', 
                   'Attention Weight Application (SP=2)\\nGPU: 32-39\\nInput: [batch_size=64, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=64, seq_len=1024, heads=64, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage1.node('attn_sp_comm_1', 
                   'Attention SP AllReduce\\nGPU: 32-39\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage1.node('out_proj_1', 
                   'Output Projection (TP=8)\\nGPU: 32-39\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage1.node('out_proj_comm_1', 
                   'Output AllReduce (TP)\\nGPU: 32-39\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage1.node('agg_1', 
                   'Expert Aggregation\\nGPU: 32-39\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
    
    # Stage 2: Layers 48-71 (GPUs 64-95)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Pipeline Stage 2: Layers 48-71\\nGPUs: 64-95', 
                   style='rounded', fillcolor=colors['stage2'])
        
        # Token Embedding
        stage2.node('embed_2', 
                   'Token Embedding\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Gate Selection
        stage2.node('gate_2', 
                   'Gate Selection\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
        
        # Expert Routing
        stage2.node('route_2', 
                   'Expert Routing\\nGPU: 64-95\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Expert Computations (8 experts per stage)
        for i in range(8):
            stage2.node(f'expert_{64+i}', 
                       f'Expert {64+i} Computation\\nGPU: {64+i}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                       shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Block - Broken down into submodules
        stage2.node('qkv_proj_2', 
                   'QKV Projection (TP=8)\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=16, seq_len=2048, heads=8, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage2.node('qkv_comm_2', 
                   'QKV AllGather (TP)\\nGPU: 64-71\\nInput: [batch_size=16, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage2.node('attn_scores_2', 
                   'Attention Score Computation (SP=2)\\nGPU: 64-71\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=64, seq_len=1024, seq_len=1024]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage2.node('attn_apply_2', 
                   'Attention Weight Application (SP=2)\\nGPU: 64-71\\nInput: [batch_size=64, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=64, seq_len=1024, heads=64, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage2.node('attn_sp_comm_2', 
                   'Attention SP AllReduce\\nGPU: 64-71\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage2.node('out_proj_2', 
                   'Output Projection (TP=8)\\nGPU: 64-71\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage2.node('out_proj_comm_2', 
                   'Output AllReduce (TP)\\nGPU: 64-71\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage2.node('agg_2', 
                   'Expert Aggregation\\nGPU: 64-71\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
    
    # Stage 3: Layers 72-93 (GPUs 96-127)
    with dot.subgraph(name='cluster_stage3') as stage3:
        stage3.attr(label='Pipeline Stage 3: Layers 72-93\\nGPUs: 96-127', 
                   style='rounded', fillcolor=colors['stage3'])
        
        # Token Embedding
        stage3.node('embed_3', 
                   'Token Embedding\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Gate Selection
        stage3.node('gate_3', 
                   'Gate Selection\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, experts=32]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
        
        # Expert Routing
        stage3.node('route_3', 
                   'Expert Routing\\nGPU: 96-127\\nInput: [batch_size=128, seq_len=2048, experts=32]\\nOutput: [batch_size=128, seq_len=2048, experts=1]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        # Expert Computations (8 experts per stage)
        for i in range(8):
            stage3.node(f'expert_{96+i}', 
                       f'Expert {96+i} Computation\\nGPU: {96+i}\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                       shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        # Attention Block - Broken down into submodules
        stage3.node('qkv_proj_3', 
                   'QKV Projection (TP=8)\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=16, seq_len=2048, heads=8, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage3.node('qkv_comm_3', 
                   'QKV AllGather (TP)\\nGPU: 96-103\\nInput: [batch_size=16, seq_len=2048, heads=8, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage3.node('attn_scores_3', 
                   'Attention Score Computation (SP=2)\\nGPU: 96-103\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=64, seq_len=1024, seq_len=1024]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage3.node('attn_apply_3', 
                   'Attention Weight Application (SP=2)\\nGPU: 96-103\\nInput: [batch_size=64, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=64, seq_len=1024, heads=64, d_k=64]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage3.node('attn_sp_comm_3', 
                   'Attention SP AllReduce\\nGPU: 96-103\\nInput: [batch_size=64, seq_len=1024, heads=64, d_k=64]\\nOutput: [batch_size=128, seq_len=2048, heads=64, d_k=64]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage3.node('out_proj_3', 
                   'Output Projection (TP=8)\\nGPU: 96-103\\nInput: [batch_size=128, seq_len=2048, heads=64, d_k=64]\\nOutput: [batch_size=16, seq_len=2048, hidden=512]',
                   shape='rectangle', style='filled', fillcolor=colors['compute'])
        
        stage3.node('out_proj_comm_3', 
                   'Output AllReduce (TP)\\nGPU: 96-103\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='ellipse', style='filled', fillcolor=colors['comm'])
        
        stage3.node('agg_3', 
                   'Expert Aggregation\\nGPU: 96-103\\nInput: [batch_size=16, seq_len=2048, hidden=512]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
                   shape='parallelogram', style='filled', fillcolor=colors['route'])
    
    # Input and Output nodes
    dot.node('input', 
            'Input\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
            shape='rectangle', style='filled', fillcolor=colors['io'])
    
    dot.node('output', 
            'Output\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, vocab_size=128000]',
            shape='rectangle', style='filled', fillcolor=colors['io'])
    
    # Pipeline Communication nodes
    dot.node('pipe_comm_0_1', 
            'Pipeline Communication\\nGPU: 31->32\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
            shape='ellipse', style='filled', fillcolor=colors['comm'])
    
    dot.node('pipe_comm_1_2', 
            'Pipeline Communication\\nGPU: 63->64\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
            shape='ellipse', style='filled', fillcolor=colors['comm'])
    
    dot.node('pipe_comm_2_3', 
            'Pipeline Communication\\nGPU: 95->96\\nInput: [batch_size=128, seq_len=2048, hidden=4096]\\nOutput: [batch_size=128, seq_len=2048, hidden=4096]',
            shape='ellipse', style='filled', fillcolor=colors['comm'])
    
    # Define edges - Stage 0
    dot.edge('input', 'embed_0')
    dot.edge('embed_0', 'gate_0')
    dot.edge('gate_0', 'route_0', style='dashed')
    
    # Expert routing to individual experts
    for i in range(8):
        dot.edge('route_0', f'expert_{i}')
    
    # Expert outputs to attention components
    for i in range(8):
        dot.edge(f'expert_{i}', 'qkv_proj_0')
    
    # Attention computation flow
    dot.edge('qkv_proj_0', 'qkv_comm_0')
    dot.edge('qkv_comm_0', 'attn_scores_0')
    dot.edge('attn_scores_0', 'attn_apply_0')
    dot.edge('attn_apply_0', 'attn_sp_comm_0')
    dot.edge('attn_sp_comm_0', 'out_proj_0')
    dot.edge('out_proj_0', 'out_proj_comm_0')
    
    # Expert outputs to aggregation
    for i in range(8):
        dot.edge(f'expert_{i}', 'agg_0')
    dot.edge('out_proj_comm_0', 'agg_0')
    
    # Pipeline communication
    dot.edge('agg_0', 'pipe_comm_0_1')
    dot.edge('pipe_comm_0_1', 'embed_1')
    
    # Stage 1
    dot.edge('embed_1', 'gate_1')
    dot.edge('gate_1', 'route_1', style='dashed')
    
    for i in range(8):
        dot.edge('route_1', f'expert_{32+i}')
    
    for i in range(8):
        dot.edge(f'expert_{32+i}', 'qkv_proj_1')
    
    dot.edge('qkv_proj_1', 'qkv_comm_1')
    dot.edge('qkv_comm_1', 'attn_scores_1')
    dot.edge('attn_scores_1', 'attn_apply_1')
    dot.edge('attn_apply_1', 'attn_sp_comm_1')
    dot.edge('attn_sp_comm_1', 'out_proj_1')
    dot.edge('out_proj_1', 'out_proj_comm_1')
    
    for i in range(8):
        dot.edge(f'expert_{32+i}', 'agg_1')
    dot.edge('out_proj_comm_1', 'agg_1')
    
    dot.edge('agg_1', 'pipe_comm_1_2')
    dot.edge('pipe_comm_1_2', 'embed_2')
    
    # Stage 2
    dot.edge('embed_2', 'gate_2')
    dot.edge('gate_2', 'route_2', style='dashed')
    
    for i in range(8):
        dot.edge('route_2', f'expert_{64+i}')
    
    for i in range(8):
        dot.edge(f'expert_{64+i}', 'qkv_proj_2')
    
    dot.edge('qkv_proj_2', 'qkv_comm_2')
    dot.edge('qkv_comm_2', 'attn_scores_2')
    dot.edge('attn_scores_2', 'attn_apply_2')
    dot.edge('attn_apply_2', 'attn_sp_comm_2')
    dot.edge('attn_sp_comm_2', 'out_proj_2')
    dot.edge('out_proj_2', 'out_proj_comm_2')
    
    for i in range(8):
        dot.edge(f'expert_{64+i}', 'agg_2')
    dot.edge('out_proj_comm_2', 'agg_2')
    
    dot.edge('agg_2', 'pipe_comm_2_3')
    dot.edge('pipe_comm_2_3', 'embed_3')
    
    # Stage 3
    dot.edge('embed_3', 'gate_3')
    dot.edge('gate_3', 'route_3', style='dashed')
    
    for i in range(8):
        dot.edge('route_3', f'expert_{96+i}')
    
    for i in range(8):
        dot.edge(f'expert_{96+i}', 'qkv_proj_3')
    
    dot.edge('qkv_proj_3', 'qkv_comm_3')
    dot.edge('qkv_comm_3', 'attn_scores_3')
    dot.edge('attn_scores_3', 'attn_apply_3')
    dot.edge('attn_apply_3', 'attn_sp_comm_3')
    dot.edge('attn_sp_comm_3', 'out_proj_3')
    dot.edge('out_proj_3', 'out_proj_comm_3')
    
    for i in range(8):
        dot.edge(f'expert_{96+i}', 'agg_3')
    dot.edge('out_proj_comm_3', 'agg_3')
    
    dot.edge('agg_3', 'output')
    
    return dot

if __name__ == '__main__':
    # Create the corrected DAG
    dag = create_corrected_dag()
    
    # Save the DOT file
    dot_file_path = './outputs/2025-12-30-17-26-35/parallel_strategy_dag_fixed.dot'
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    dag.render('./outputs/2025-12-30-17-26-35/parallel_strategy_dag_fixed', format='svg', cleanup=True)
    
    print(f"Corrected DAG saved to: {dot_file_path}")
    print(f"SVG image saved to: ./outputs/2025-12-30-17-26-35/parallel_strategy_dag_fixed.svg")