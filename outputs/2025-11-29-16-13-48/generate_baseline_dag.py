#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """Create DAG for baseline TP=8, PP=2 deployment strategy"""
    
    dot = graphviz.Digraph(comment='Baseline Dense Model DAG (TP=8, PP=2)')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\n(batch_size=128, seq_len=10000, hidden_size=4096)', shape='box', fillcolor='lightcoral')
    
    # Stage 1: Pipeline Parallel Stage 1 (TP=8)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1 (TP=8)', style='rounded,dashed', bgcolor='lightblue', fontcolor='black')
        
        # Input split for tensor parallelism
        c.node('split_input_s1', 'Split Input\nColumn-wise', shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 1 computations across 8 GPUs
        for i in range(8):
            gpu_id = i
            c.node(f'layer1_gpu{gpu_id}', f'Layer 1\nGPU {gpu_id}\n(TP shard)', shape='box', fillcolor='lightgreen')
        
        # All-reduce for layer 1
        c.node('ar_layer1', 'All-Reduce\nLayer 1 Output', shape='ellipse', fillcolor='lightblue')
        
        # Layer 2 computations across 8 GPUs
        for i in range(8):
            gpu_id = i
            c.node(f'layer2_gpu{gpu_id}', f'Layer 2\nGPU {gpu_id}\n(TP shard)', shape='box', fillcolor='lightgreen')
        
        # All-reduce for layer 2
        c.node('ar_layer2', 'All-Reduce\nLayer 2 Output', shape='ellipse', fillcolor='lightblue')
        
        # Connect within stage 1
        c.edge('split_input_s1', 'layer1_gpu0', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu1', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu2', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu3', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu4', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu5', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu6', style='dashed')
        c.edge('split_input_s1', 'layer1_gpu7', style='dashed')
        
        for i in range(8):
            c.edge(f'layer1_gpu{i}', 'ar_layer1')
            c.edge('ar_layer1', f'layer2_gpu{i}')
            c.edge(f'layer2_gpu{i}', 'ar_layer2')
    
    # Stage 2: Pipeline Parallel Stage 2 (TP=8)
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Pipeline Stage 2 (TP=8)', style='rounded,dashed', bgcolor='lightpink', fontcolor='black')
        
        # Transfer from stage 1 to stage 2
        c.node('transfer_s1_s2', 'Transfer\nStage1→Stage2', shape='ellipse', fillcolor='lightblue')
        
        # Input split for tensor parallelism in stage 2
        c.node('split_input_s2', 'Split Input\nColumn-wise', shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 3 computations across 8 GPUs (8-15)
        for i in range(8):
            gpu_id = i + 8
            c.node(f'layer3_gpu{gpu_id}', f'Layer 3\nGPU {gpu_id}\n(TP shard)', shape='box', fillcolor='lightgreen')
        
        # All-reduce for layer 3
        c.node('ar_layer3', 'All-Reduce\nLayer 3 Output', shape='ellipse', fillcolor='lightblue')
        
        # Layer 4 computations across 8 GPUs
        for i in range(8):
            gpu_id = i + 8
            c.node(f'layer4_gpu{gpu_id}', f'Layer 4\nGPU {gpu_id}\n(TP shard)', shape='box', fillcolor='lightgreen')
        
        # All-reduce for layer 4
        c.node('ar_layer4', 'All-Reduce\nLayer 4 Output', shape='ellipse', fillcolor='lightblue')
        
        # Connect within stage 2
        c.edge('transfer_s1_s2', 'split_input_s2')
        c.edge('split_input_s2', 'layer3_gpu8', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu9', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu10', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu11', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu12', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu13', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu14', style='dashed')
        c.edge('split_input_s2', 'layer3_gpu15', style='dashed')
        
        for i in range(8):
            gpu_id = i + 8
            c.edge(f'layer3_gpu{gpu_id}', 'ar_layer3')
            c.edge('ar_layer3', f'layer4_gpu{gpu_id}')
            c.edge(f'layer4_gpu{gpu_id}', 'ar_layer4')
    
    # Connect stages
    dot.edge('input', 'split_input_s1')
    dot.edge('ar_layer2', 'transfer_s1_s2')
    
    # Output aggregation
    dot.node('output', 'Output\n(batch_size=128, seq_len=10000, hidden_size=4096)', shape='box', fillcolor='lightcoral')
    dot.edge('ar_layer4', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-11-29-16-13-48/baseline_model_dag.dot')
    
    # Save as SVG image
    dag.render('../outputs/2025-11-29-16-13-48/baseline_model_dag', format='svg', cleanup=True)
    
    print("Baseline DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-11-29-16-13-48/baseline_model_dag.dot")
    print(f"SVG file: ../outputs/2025-11-29-16-13-48/baseline_model_dag.svg")