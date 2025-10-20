#!/usr/bin/env python3
"""
Simplified but complete FA Pool DAG
Dynamic allocation: 8 base GPUs + 32 pool GPUs for 32K+ tokens
"""

import graphviz
import os

def create_fa_pool_dag():
    """Create simplified FA Pool DAG"""
    dot = graphviz.Digraph(comment='FA Pool Dynamic Allocation DAG')
    dot.attr(rankdir='TB', splines='polyline', ranksep='2', nodesep='1')
    
    # Configuration
    batch_size = 1024
    seq_len = 32768
    hidden = 4096
    pool_gpus = 32
    base_gpus = 8
    
    # Input
    dot.node('input', f'Total Input\\n[batch={batch_size}, seq={seq_len}, hidden={hidden}]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Base Layer - GPUs 0-7
    with dot.subgraph(name='cluster_base') as cbase:
        cbase.attr(label='Base Layer (GPUs 0-7)', style='rounded,filled', fillcolor='lightblue')
        
        cbase.node('embed', f'Embedding\\n[vocab→{hidden}]\\nGPU: 0-7', shape='rectangle')
        
        # FFN only layers (attention handled by pool)
        for i in range(4):
            cbase.node(f'layer{i}_ffn', f'L{i} FFN\\nTP=2 GPUs\\n[{batch_size}, {seq_len}, {hidden*4}]', 
                      shape='rectangle')
    
    # Attention Pool - GPUs 8-39
    with dot.subgraph(name='cluster_pool') as cpool:
        cpool.attr(label=f'Attention Pool (GPUs 8-{7+pool_gpus})', style='rounded,filled', fillcolor='lightcoral')
        
        # KV Cache replication
        cpool.node('kv_replicate', f'KV Cache Replication\\n[{batch_size}, {seq_len}, {hidden}]\\nAll {pool_gpus} GPUs', 
                  shape='parallelogram')
        
        # Parallel attention blocks
        for i in range(pool_gpus):
            block_size = seq_len // pool_gpus
            start = i * block_size
            end = (i + 1) * block_size if i < pool_gpus - 1 else seq_len
            actual_size = end - start
            
            cpool.node(f'attn_{i}', f'Flash Attention\\nGPU {8+i}\\n[{batch_size}, {actual_size}, {hidden}]', 
                      shape='rectangle')
        
        # Results aggregation
        cpool.node('agg', f'Results Aggregation\\n[{pool_gpus} blocks → {seq_len}]\\nGPU: 0-7', 
                  shape='parallelogram')
    
    # Output
    dot.node('output', f'Total Output\\n[{batch_size}, {seq_len}, 32000]\\nGPU: 0-7', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'embed')
    dot.edge('embed', 'kv_replicate')
    
    # Connect to all attention blocks
    for i in range(pool_gpus):
        dot.edge('embed', f'attn_{i}')
        dot.edge('kv_replicate', f'attn_{i}')
        dot.edge(f'attn_{i}', 'agg')
    
    # Layer connections through FFN
    dot.edge('agg', 'layer0_ffn')
    dot.edge('layer0_ffn', 'layer1_ffn')
    dot.edge('layer1_ffn', 'layer2_ffn')
    dot.edge('layer2_ffn', 'layer3_ffn')
    dot.edge('layer3_ffn', 'output')
    
    # Add resource manager
    dot.node('resource_mgr', f'Resource Manager\\nSequence Length > 4096\\nActivate {pool_gpus} GPUs\\nGPU: 0-7', 
             shape='diamond', fillcolor='yellow')
    dot.edge('input', 'resource_mgr')
    
    return dot

if __name__ == '__main__':
    dag = create_fa_pool_dag()
    
    # Save DOT file
    dot_path = os.path.join('../outputs/2025-10-19-22-51-33', 'fa_pool_dag.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG
    svg_path = os.path.join('../outputs/2025-10-19-22-51-33', 'fa_pool_dag.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"FA Pool DAG generated:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")