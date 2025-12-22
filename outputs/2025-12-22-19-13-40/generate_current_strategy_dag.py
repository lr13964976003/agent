#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_current_strategy_dag():
    """
    Create DAG for current strategy: EP64-TP8-PP2-DP2
    Total GPUs: 2048 (64 × 8 × 2 × 2)
    """
    dot = Digraph(comment='Current Strategy: EP64-TP8-PP2-DP2')
    dot.attr(rankdir='TB', ranksep='1.0', nodesep='0.5')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='white')
    
    # Pipeline Stage 0 (Layers 0-7)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (Layers 0-7)\\nGPUs: 0-1023', style='rounded,filled', fillcolor='lightgray')
        
        # Layer 0
        stage0.node('layer0_attn_qkv_tp', 'Layer 0 Attention QKV\\nTP8 [GPUs: 0-7]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage0.node('layer0_attn_out_tp', 'Layer 0 Attention Output\\nTP8 [GPUs: 0-7]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        # Expert routing
        stage0.node('layer0_route', 'Layer 0 Expert Routing\\nEP64 [GPUs: 0-63]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Expert computation - 64 experts distributed
        for i in range(64):
            gpu_start = i * 16  # 16 GPUs per expert (8 TP × 2 for redundancy)
            stage0.node(f'layer0_expert_{i}', f'Expert {i}\\nEP64 [GPUs: {gpu_start}-{gpu_start+15}]\\nInput: [2,1024,1024]\\nOutput: [2,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        # Expert combine
        stage0.node('layer0_combine', 'Layer 0 Expert Combine\\nEP64 [GPUs: 0-63]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Communication nodes
        stage0.node('layer0_attn_allreduce', 'Attention All-Reduce\\nTP8 [GPUs: 0-7]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage0.node('layer0_moe_alltoall', 'MoE All-to-All\\nEP64 [GPUs: 0-63]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 1 (Layers 8-15)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (Layers 8-15)\\nGPUs: 1024-2047', style='rounded,filled', fillcolor='lightgray')
        
        # Similar structure for layer 8 (representative)
        stage1.node('layer8_attn_qkv_tp', 'Layer 8 Attention QKV\\nTP8 [GPUs: 1024-1031]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage1.node('layer8_attn_out_tp', 'Layer 8 Attention Output\\nTP8 [GPUs: 1024-1031]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        stage1.node('layer8_route', 'Layer 8 Expert Routing\\nEP64 [GPUs: 1024-1087]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        for i in range(64):
            gpu_start = 1024 + i * 16
            stage1.node(f'layer8_expert_{i}', f'Expert {i}\\nEP64 [GPUs: {gpu_start}-{gpu_start+15}]\\nInput: [2,1024,1024]\\nOutput: [2,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        stage1.node('layer8_combine', 'Layer 8 Expert Combine\\nEP64 [GPUs: 1024-1087]\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage1.node('layer8_attn_allreduce', 'Attention All-Reduce\\nTP8 [GPUs: 1024-1031]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage1.node('layer8_moe_alltoall', 'MoE All-to-All\\nEP64 [GPUs: 1024-1087]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='white')
    
    # Connections
    dot.edge('input', 'layer0_attn_qkv_tp')
    dot.edge('layer0_attn_qkv_tp', 'layer0_attn_allreduce')
    dot.edge('layer0_attn_allreduce', 'layer0_attn_out_tp')
    dot.edge('layer0_attn_out_tp', 'layer0_route')
    dot.edge('layer0_route', 'layer0_moe_alltoall')
    
    # Connect to experts
    for i in range(64):
        dot.edge('layer0_moe_alltoall', f'layer0_expert_{i}')
        dot.edge(f'layer0_expert_{i}', 'layer0_combine', style='dashed')
    
    dot.edge('layer0_combine', 'layer8_attn_qkv_tp', lhead='cluster_stage1')
    
    # Similar connections for stage 1
    dot.edge('layer8_attn_qkv_tp', 'layer8_attn_allreduce')
    dot.edge('layer8_attn_allreduce', 'layer8_attn_out_tp')
    dot.edge('layer8_attn_out_tp', 'layer8_route')
    dot.edge('layer8_route', 'layer8_moe_alltoall')
    
    for i in range(64):
        dot.edge('layer8_moe_alltoall', f'layer8_expert_{i}')
        dot.edge(f'layer8_expert_{i}', 'layer8_combine', style='dashed')
    
    dot.edge('layer8_combine', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_current_strategy_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-22-19-13-40/current_strategy_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-22-19-13-40/current_strategy_dag', format='svg', cleanup=True)
    
    print("Current strategy DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-22-19-13-40/current_strategy_dag.dot")
    print(f"SVG file: ../outputs/2025-12-22-19-13-40/current_strategy_dag.svg")