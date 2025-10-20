#!/usr/bin/env python3
"""
Simplified but complete baseline DAG (TP=8, PP=2)
Focus on parallel structure and GPU allocation
"""

import graphviz
import os

def create_baseline_dag():
    """Create simplified baseline DAG"""
    dot = graphviz.Digraph(comment='Baseline TP=8 PP=2 DAG')
    dot.attr(rankdir='TB', splines='polyline', ranksep='2', nodesep='1')
    
    # Model dimensions
    batch_size = 1024
    seq_len = 4096
    hidden = 4096
    heads = 32
    ffn = 16384
    
    # Input
    dot.node('input', f'Total Input\\n[batch={batch_size}, seq={seq_len}, hidden={hidden}]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0 - GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as c0:
        c0.attr(label='Stage 0 (GPUs 0-7)', style='rounded,filled', fillcolor='lightblue')
        
        c0.node('stage0_embed', f'Embedding\\n[vocab→{hidden}]\\nGPU: 0', 
               shape='rectangle')
        
        # Layer 0
        c0.node('layer0_attn', f'L0 Attention\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {hidden}]', 
               shape='rectangle')
        c0.node('layer0_ffn', f'L0 FFN\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {ffn}]', 
               shape='rectangle')
        
        # Layer 1
        c0.node('layer1_attn', f'L1 Attention\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {hidden}]', 
               shape='rectangle')
        c0.node('layer1_ffn', f'L1 FFN\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {ffn}]', 
               shape='rectangle')
    
    # Pipeline communication
    dot.node('pipeline_comm', 'Pipeline Send\\nGPU 7 → GPU 8', 
             shape='parallelogram', fillcolor='yellow')
    
    # Stage 1 - GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as c1:
        c1.attr(label='Stage 1 (GPUs 8-15)', style='rounded,filled', fillcolor='lightcoral')
        
        # Layer 2
        c1.node('layer2_attn', f'L2 Attention\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {hidden}]', 
               shape='rectangle')
        c1.node('layer2_ffn', f'L2 FFN\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {ffn}]', 
               shape='rectangle')
        
        # Layer 3
        c1.node('layer3_attn', f'L3 Attention\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {hidden}]', 
               shape='rectangle')
        c1.node('layer3_ffn', f'L3 FFN\\nTP=8 GPUs\\n[{batch_size}, {seq_len}, {ffn}]', 
               shape='rectangle')
        
        # Output
        c1.node('output', f'Total Output\\n[{batch_size}, {seq_len}, 32000]\\nGPU: 15', 
               shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'stage0_embed')
    dot.edge('stage0_embed', 'layer0_attn')
    dot.edge('layer0_attn', 'layer0_ffn')
    dot.edge('layer0_ffn', 'layer1_attn')
    dot.edge('layer1_attn', 'layer1_ffn')
    dot.edge('layer1_ffn', 'pipeline_comm')
    dot.edge('pipeline_comm', 'layer2_attn')
    dot.edge('layer2_attn', 'layer2_ffn')
    dot.edge('layer2_ffn', 'layer3_attn')
    dot.edge('layer3_attn', 'layer3_ffn')
    dot.edge('layer3_ffn', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    
    # Save DOT file
    dot_path = os.path.join('../outputs/2025-10-19-22-51-33', 'baseline_dag.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG
    svg_path = os.path.join('../outputs/2025-10-19-22-51-33', 'baseline_dag.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Baseline DAG generated:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")