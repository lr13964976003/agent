#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_optimal_strategy_dag():
    """
    Create DAG for optimal strategy: EP32-TP4-PP4-DP8
    Total GPUs: 512 (32 × 4 × 4 × 8)
    """
    dot = Digraph(comment='Optimal Strategy: EP32-TP4-PP4-DP8')
    dot.attr(rankdir='TB', ranksep='1.0', nodesep='0.5')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')   # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\n[batch_size=128, seq_len=1024, hidden=1024]\\nDP8: 16 seqs per GPU', 
             shape='ellipse', fillcolor='white')
    
    # Pipeline Stage 0 (Layers 0-3)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (Layers 0-3)\\nGPUs: 0-127', style='rounded,filled', fillcolor='lightgray')
        
        # Layer 0 - Attention with TP4
        stage0.node('layer0_attn_qkv_tp', 'Layer 0 Attention QKV\\nTP4 [GPUs: 0-3]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage0.node('layer0_attn_score', 'Layer 0 Attention Score\\nTP4 [GPUs: 0-3]\\nInput: [16,16,1024,64]\\nOutput: [16,16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage0.node('layer0_attn_out_tp', 'Layer 0 Attention Output\\nTP4 [GPUs: 0-3]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        # Expert routing for EP32
        stage0.node('layer0_route', 'Layer 0 Expert Routing\\nEP32 [GPUs: 0-31]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Expert computation - 32 groups, 2 experts per GPU
        for i in range(32):
            gpu_start = i * 4  # 4 GPUs per expert group (TP4)
            stage0.node(f'layer0_expert_group_{i}', f'Expert Group {i}\\n(2 experts)\\nEP32 [GPUs: {gpu_start}-{gpu_start+3}]\\nInput: [1,1024,1024]\\nOutput: [1,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        # Expert combine
        stage0.node('layer0_combine', 'Layer 0 Expert Combine\\nEP32 [GPUs: 0-31]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Communication nodes
        stage0.node('layer0_attn_allreduce', 'Attention All-Reduce\\nTP4 [GPUs: 0-3]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage0.node('layer0_moe_alltoall', 'MoE All-to-All\\nEP32 [GPUs: 0-31]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 1 (Layers 4-7)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (Layers 4-7)\\nGPUs: 128-255', style='rounded,filled', fillcolor='lightgray')
        
        stage1.node('layer4_attn_qkv_tp', 'Layer 4 Attention QKV\\nTP4 [GPUs: 128-131]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage1.node('layer4_attn_score', 'Layer 4 Attention Score\\nTP4 [GPUs: 128-131]\\nInput: [16,16,1024,64]\\nOutput: [16,16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage1.node('layer4_attn_out_tp', 'Layer 4 Attention Output\\nTP4 [GPUs: 128-131]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        stage1.node('layer4_route', 'Layer 4 Expert Routing\\nEP32 [GPUs: 128-159]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        for i in range(32):
            gpu_start = 128 + i * 4
            stage1.node(f'layer4_expert_group_{i}', f'Expert Group {i}\\n(2 experts)\\nEP32 [GPUs: {gpu_start}-{gpu_start+3}]\\nInput: [1,1024,1024]\\nOutput: [1,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        stage1.node('layer4_combine', 'Layer 4 Expert Combine\\nEP32 [GPUs: 128-159]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage1.node('layer4_attn_allreduce', 'Attention All-Reduce\\nTP4 [GPUs: 128-131]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage1.node('layer4_moe_alltoall', 'MoE All-to-All\\nEP32 [GPUs: 128-159]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 2 (Layers 8-11)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Pipeline Stage 2 (Layers 8-11)\\nGPUs: 256-383', style='rounded,filled', fillcolor='lightgray')
        
        stage2.node('layer8_attn_qkv_tp', 'Layer 8 Attention QKV\\nTP4 [GPUs: 256-259]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage2.node('layer8_attn_score', 'Layer 8 Attention Score\\nTP4 [GPUs: 256-259]\\nInput: [16,16,1024,64]\\nOutput: [16,16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage2.node('layer8_attn_out_tp', 'Layer 8 Attention Output\\nTP4 [GPUs: 256-259]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        stage2.node('layer8_route', 'Layer 8 Expert Routing\\nEP32 [GPUs: 256-287]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        for i in range(32):
            gpu_start = 256 + i * 4
            stage2.node(f'layer8_expert_group_{i}', f'Expert Group {i}\\n(2 experts)\\nEP32 [GPUs: {gpu_start}-{gpu_start+3}]\\nInput: [1,1024,1024]\\nOutput: [1,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        stage2.node('layer8_combine', 'Layer 8 Expert Combine\\nEP32 [GPUs: 256-287]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage2.node('layer8_attn_allreduce', 'Attention All-Reduce\\nTP4 [GPUs: 256-259]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage2.node('layer8_moe_alltoall', 'MoE All-to-All\\nEP32 [GPUs: 256-287]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 3 (Layers 12-15)
    with dot.subgraph(name='cluster_stage3') as stage3:
        stage3.attr(label='Pipeline Stage 3 (Layers 12-15)\\nGPUs: 384-511', style='rounded,filled', fillcolor='lightgray')
        
        stage3.node('layer12_attn_qkv_tp', 'Layer 12 Attention QKV\\nTP4 [GPUs: 384-387]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage3.node('layer12_attn_score', 'Layer 12 Attention Score\\nTP4 [GPUs: 384-387]\\nInput: [16,16,1024,64]\\nOutput: [16,16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        stage3.node('layer12_attn_out_tp', 'Layer 12 Attention Output\\nTP4 [GPUs: 384-387]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='rectangle', fillcolor='lightblue')
        
        stage3.node('layer12_route', 'Layer 12 Expert Routing\\nEP32 [GPUs: 384-415]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        for i in range(32):
            gpu_start = 384 + i * 4
            stage3.node(f'layer12_expert_group_{i}', f'Expert Group {i}\\n(2 experts)\\nEP32 [GPUs: {gpu_start}-{gpu_start+3}]\\nInput: [1,1024,1024]\\nOutput: [1,1024,1024]', 
                       shape='rectangle', fillcolor='lightblue')
        
        stage3.node('layer12_combine', 'Layer 12 Expert Combine\\nEP32 [GPUs: 384-415]\\nInput: [16,1024,1024]\\nOutput: [16,1024,1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        stage3.node('layer12_attn_allreduce', 'Attention All-Reduce\\nTP4 [GPUs: 384-387]', 
                   shape='ellipse', fillcolor='lightgreen')
        stage3.node('layer12_moe_alltoall', 'MoE All-to-All\\nEP32 [GPUs: 384-415]', 
                   shape='ellipse', fillcolor='lightgreen')
    
    # Output
    dot.node('output', 'Output\\n[batch_size=128, seq_len=1024, hidden=1024]\\nDP8: Aggregated from 8 groups', 
             shape='ellipse', fillcolor='white')
    
    # Connections for first layer
    dot.edge('input', 'layer0_attn_qkv_tp')
    dot.edge('layer0_attn_qkv_tp', 'layer0_attn_score')
    dot.edge('layer0_attn_score', 'layer0_attn_allreduce')
    dot.edge('layer0_attn_allreduce', 'layer0_attn_out_tp')
    dot.edge('layer0_attn_out_tp', 'layer0_route')
    dot.edge('layer0_route', 'layer0_moe_alltoall')
    
    # Connect to expert groups
    for i in range(32):
        dot.edge('layer0_moe_alltoall', f'layer0_expert_group_{i}')
        dot.edge(f'layer0_expert_group_{i}', 'layer0_combine', style='dashed')
    
    # Connect to next pipeline stage
    dot.edge('layer0_combine', 'layer4_attn_qkv_tp', lhead='cluster_stage1')
    
    # Similar connections for stage 1
    dot.edge('layer4_attn_qkv_tp', 'layer4_attn_score')
    dot.edge('layer4_attn_score', 'layer4_attn_allreduce')
    dot.edge('layer4_attn_allreduce', 'layer4_attn_out_tp')
    dot.edge('layer4_attn_out_tp', 'layer4_route')
    dot.edge('layer4_route', 'layer4_moe_alltoall')
    
    for i in range(32):
        dot.edge('layer4_moe_alltoall', f'layer4_expert_group_{i}')
        dot.edge(f'layer4_expert_group_{i}', 'layer4_combine', style='dashed')
    
    dot.edge('layer4_combine', 'layer8_attn_qkv_tp', lhead='cluster_stage2')
    
    # Stage 2 connections
    dot.edge('layer8_attn_qkv_tp', 'layer8_attn_score')
    dot.edge('layer8_attn_score', 'layer8_attn_allreduce')
    dot.edge('layer8_attn_allreduce', 'layer8_attn_out_tp')
    dot.edge('layer8_attn_out_tp', 'layer8_route')
    dot.edge('layer8_route', 'layer8_moe_alltoall')
    
    for i in range(32):
        dot.edge('layer8_moe_alltoall', f'layer8_expert_group_{i}')
        dot.edge(f'layer8_expert_group_{i}', 'layer8_combine', style='dashed')
    
    dot.edge('layer8_combine', 'layer12_attn_qkv_tp', lhead='cluster_stage3')
    
    # Stage 3 connections
    dot.edge('layer12_attn_qkv_tp', 'layer12_attn_score')
    dot.edge('layer12_attn_score', 'layer12_attn_allreduce')
    dot.edge('layer12_attn_allreduce', 'layer12_attn_out_tp')
    dot.edge('layer12_attn_out_tp', 'layer12_route')
    dot.edge('layer12_route', 'layer12_moe_alltoall')
    
    for i in range(32):
        dot.edge('layer12_moe_alltoall', f'layer12_expert_group_{i}')
        dot.edge(f'layer12_expert_group_{i}', 'layer12_combine', style='dashed')
    
    dot.edge('layer12_combine', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_optimal_strategy_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-12-22-19-13-40/optimal_strategy_dag.dot')
    
    # Render as SVG
    dag.render('../outputs/2025-12-22-19-13-40/optimal_strategy_dag', format='svg', cleanup=True)
    
    print("Optimal strategy DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-12-22-19-13-40/optimal_strategy_dag.dot")
    print(f"SVG file: ../outputs/2025-12-22-19-13-40/optimal_strategy_dag.svg")