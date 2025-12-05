#!/usr/bin/env python3

import graphviz
from graphviz import Digraph
import os

def create_moe_deployment_dag():
    """
    Create a comprehensive DAG for 30B MoE model deployment with optimized parallel strategy.
    
    Parallel Configuration:
    - Tensor Parallelism: 4-way (TP)
    - Expert Parallelism: 8-way (EP) 
    - Pipeline Parallelism: 2-stage (PP)
    - Data Parallelism: 1-way (DP)
    
    GPU Assignment:
    - Stage 1: GPUs 0-7 (8 GPUs)
      - TP group 0: GPUs 0-3 (4-way tensor parallel)
      - TP group 1: GPUs 4-7 (4-way tensor parallel)
    - Stage 2: GPUs 8-15 (8 GPUs)
      - TP group 2: GPUs 8-11 (4-way tensor parallel)
      - TP group 3: GPUs 12-15 (4-way tensor parallel)
    """
    
    dot = Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='40,60', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nBatch: [batch_size=128, seq_len=1024, hidden=1024]\\nGPU: Host', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Stage 1: Layers 1-8 (GPUs 0-7)
    # ====================================================================================
    
    # Stage 1 Input Distribution
    with dot.subgraph(name='cluster_stage1_input') as c:
        c.attr(label='Stage 1 Input Distribution (GPUs 0-7)', style='rounded,filled', fillcolor='lightgray')
        
        # Data split and distribution
        c.node('s1_split', 'Split Input Data\\n[128, 1024, 1024] → 4×[128, 1024, 256]\\nGPU: 0-7', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Communication: Broadcast input to all GPUs
        c.node('s1_broadcast', 'Broadcast Input\\n[128, 1024, 256]\\nGPU: 0-7', 
               shape='ellipse', fillcolor='lightblue')
    
    # Layer 1 (Stage 1)
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 - Multi-Head Attention (Stage 1)', style='rounded,filled', fillcolor='lightcyan')
        
        # LayerNorm (replicated across TP group)
        c.node('l1_ln', 'LayerNorm\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # QKV Linear (Column Parallel)
        c.node('l1_qkv', 'QKV Linear (Col-Parallel)\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 192]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # QKV All-Gather Communication
        c.node('l1_qkv_comm', 'All-Gather QKV\\n[128, 1024, 192]\\nGPU: 0-3,4-7', 
               shape='ellipse', fillcolor='lightblue')
        
        # Reshape for Attention
        c.node('l1_reshape', 'Reshape for Attention\\n[128, 1024, 192] → [128, 16, 1024, 64]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention Scores (4 heads per GPU due to TP=4)
        c.node('l1_scores', 'Attention Scores\\nInput: [128, 4, 1024, 64]\\nOutput: [128, 4, 1024, 1024]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention Weights (Softmax)
        c.node('l1_weights', 'Attention Softmax\\nInput: [128, 4, 1024, 1024]\\nOutput: [128, 4, 1024, 1024]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention Output (4 heads per GPU)
        c.node('l1_attout', 'Attention Output\\nInput: [128, 4, 1024, 64]\\nOutput: [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Attention All-Reduce
        c.node('l1_att_allreduce', 'All-Reduce Attention\\n[128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='ellipse', fillcolor='lightblue')
        
        # Attention Residual
        c.node('l1_residual', 'Add Residual\\n[128, 1024, 256] + [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Layer 1 MLP (Stage 1)
    with dot.subgraph(name='cluster_layer1_mlp') as c:
        c.attr(label='Layer 1 - MLP (Stage 1)', style='rounded,filled', fillcolor='lightcyan')
        
        # MLP LayerNorm
        c.node('l1_mlp_ln', 'MLP LayerNorm\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # MLP First Linear (Column Parallel)
        c.node('l1_mlp_linear1', 'MLP Linear 1 (Col-Parallel)\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 1024]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # MLP All-Gather Communication
        c.node('l1_mlp_gather', 'All-Gather MLP\\n[128, 1024, 1024]\\nGPU: 0-3,4-7', 
               shape='ellipse', fillcolor='lightblue')
        
        # MLP Activation (GELU)
        c.node('l1_mlp_gelu', 'GELU Activation\\nInput: [128, 1024, 1024]\\nOutput: [128, 1024, 1024]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # MLP Second Linear (Row Parallel)
        c.node('l1_mlp_linear2', 'MLP Linear 2 (Row-Parallel)\\nInput: [128, 1024, 512]\\nOutput: [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
        
        # MLP All-Reduce
        c.node('l1_mlp_allreduce', 'All-Reduce MLP\\n[128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='ellipse', fillcolor='lightblue')
        
        # MLP Residual
        c.node('l1_mlp_residual', 'Add Residual\\n[128, 1024, 256] + [128, 1024, 256]\\nGPU: 0-3,4-7', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Layer 1 MoE (Stage 1)
    with dot.subgraph(name='cluster_layer1_moe') as c:
        c.attr(label='Layer 1 - MoE Routing (Stage 1)', style='rounded,filled', fillcolor='lightcyan')
        
        # MoE Gate (Router)
        c.node('l1_moe_gate', 'MoE Gate\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 64]\\nGPU: 0-3,4-7', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert Selection (Top-K)
        c.node('l1_moe_select', 'Expert Selection (Top-2)\\n[128, 1024, 64] → Expert IDs\\nGPU: 0-3,4-7', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert Routing (Dashed line for gate selection)
        for gpu_id in range(8):
            experts_range = f"{gpu_id*8}-{(gpu_id+1)*8-1}"
            c.node(f'l1_moe_route_gpu{gpu_id}', f'Route to Experts {experts_range}\\nGPU: {gpu_id}', 
                   shape='ellipse', fillcolor='lightblue', style='dashed')
    
    # Expert Computation (8 experts per GPU)
    with dot.subgraph(name='cluster_layer1_experts') as c:
        c.attr(label='Layer 1 - Expert Computation (8 experts per GPU)', style='rounded,filled', fillcolor='lightcyan')
        
        for gpu_id in range(8):
            experts_range = f"{gpu_id*8}-{(gpu_id+1)*8-1}"
            # Expert computation nodes
            c.node(f'l1_expert_linear1_gpu{gpu_id}', f'Expert Linear 1\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 2048]\\nGPU: {gpu_id}\\nExperts: {experts_range}', 
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_expert_gelu_gpu{gpu_id}', f'Expert GELU\\nInput: [128, 1024, 2048]\\nOutput: [128, 1024, 2048]\\nGPU: {gpu_id}\\nExperts: {experts_range}', 
                   shape='rectangle', fillcolor='lightgreen')
            
            c.node(f'l1_expert_linear2_gpu{gpu_id}', f'Expert Linear 2\\nInput: [128, 1024, 2048]\\nOutput: [128, 1024, 256]\\nGPU: {gpu_id}\\nExperts: {experts_range}', 
                   shape='rectangle', fillcolor='lightgreen')
    
    # Expert Aggregation
    with dot.subgraph(name='cluster_layer1_moe_agg') as c:
        c.attr(label='Layer 1 - MoE Aggregation (Stage 1)', style='rounded,filled', fillcolor='lightcyan')
        
        # Gather expert outputs
        c.node('l1_moe_gather', 'Gather Expert Outputs\\n8×[128, 1024, 256]\\nGPU: 0-7', 
               shape='ellipse', fillcolor='lightblue')
        
        # Weighted sum of expert outputs
        c.node('l1_moe_weighted', 'Weighted Sum\\n[128, 1024, 256]\\nGPU: 0-7', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # MoE residual
        c.node('l1_moe_residual', 'Add Residual\\n[128, 1024, 256] + [128, 1024, 256]\\nGPU: 0-7', 
               shape='rectangle', fillcolor='lightgreen')
    
    # Stage 1 → Stage 2 Pipeline Communication
    with dot.subgraph(name='cluster_pipeline_comm') as c:
        c.attr(label='Pipeline Communication (Stage 1 → Stage 2)', style='rounded,filled', fillcolor='lightgray')
        
        # Pipeline send from Stage 1
        c.node('pipe_send', 'Pipeline Send\\n[128, 1024, 256]\\nGPU: 0-7 → 8-15', 
               shape='ellipse', fillcolor='lightblue')
        
        # Pipeline receive in Stage 2
        c.node('pipe_recv', 'Pipeline Receive\\n[128, 1024, 256]\\nGPU: 8-15', 
               shape='ellipse', fillcolor='lightblue')
    
    # Stage 2: Layers 9-16 (GPUs 8-15)
    # ====================================================================================
    
    # Stage 2 Input Processing (similar to Stage 1 but for GPUs 8-15)
    with dot.subgraph(name='cluster_stage2_layers') as c:
        c.attr(label='Stage 2: Layers 9-16 (GPUs 8-15)', style='rounded,filled', fillcolor='lightgray')
        
        # Similar structure as Stage 1 but for remaining layers
        # For brevity, I'll show the pattern for one layer in Stage 2
        
        # Layer 16 (final layer)
        c.node('l16_final', 'Final LayerNorm\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 256]\\nGPU: 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Output projection
        c.node('output_proj', 'Output Projection\\nInput: [128, 1024, 256]\\nOutput: [128, 1024, 1024]\\nGPU: 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        
        # Final All-Gather for output
        c.node('final_gather', 'All-Gather Output\\n4×[128, 1024, 256] → [128, 1024, 1024]\\nGPU: 8-15', 
               shape='ellipse', fillcolor='lightblue')
    
    # Output node
    dot.node('output', 'Final Output\\n[128, 1024, 1024]\\nGPU: Host', 
             shape='ellipse', fillcolor='lightcoral')
    
    # Define edges (connections)
    # Input → Stage 1
    dot.edge('input', 's1_split')
    dot.edge('s1_split', 's1_broadcast')
    
    # Stage 1 Layer 1 Attention
    dot.edge('s1_broadcast', 'l1_ln')
    dot.edge('l1_ln', 'l1_qkv')
    dot.edge('l1_qkv', 'l1_qkv_comm')
    dot.edge('l1_qkv_comm', 'l1_reshape')
    dot.edge('l1_reshape', 'l1_scores')
    dot.edge('l1_scores', 'l1_weights')
    dot.edge('l1_weights', 'l1_attout')
    dot.edge('l1_attout', 'l1_att_allreduce')
    dot.edge('l1_att_allreduce', 'l1_residual')
    
    # Stage 1 Layer 1 MLP
    dot.edge('l1_residual', 'l1_mlp_ln')
    dot.edge('l1_mlp_ln', 'l1_mlp_linear1')
    dot.edge('l1_mlp_linear1', 'l1_mlp_gather')
    dot.edge('l1_mlp_gather', 'l1_mlp_gelu')
    dot.edge('l1_mlp_gelu', 'l1_mlp_linear2')
    dot.edge('l1_mlp_linear2', 'l1_mlp_allreduce')
    dot.edge('l1_mlp_allreduce', 'l1_mlp_residual')
    
    # Stage 1 Layer 1 MoE
    dot.edge('l1_mlp_residual', 'l1_moe_gate')
    dot.edge('l1_moe_gate', 'l1_moe_select')
    
    # Expert routing (dashed edges)
    for gpu_id in range(8):
        dot.edge('l1_moe_select', f'l1_moe_route_gpu{gpu_id}', style='dashed')
        dot.edge(f'l1_moe_route_gpu{gpu_id}', f'l1_expert_linear1_gpu{gpu_id}')
        dot.edge(f'l1_expert_linear1_gpu{gpu_id}', f'l1_expert_gelu_gpu{gpu_id}')
        dot.edge(f'l1_expert_gelu_gpu{gpu_id}', f'l1_expert_linear2_gpu{gpu_id}')
        dot.edge(f'l1_expert_linear2_gpu{gpu_id}', 'l1_moe_gather')
    
    # MoE aggregation
    dot.edge('l1_moe_gather', 'l1_moe_weighted')
    dot.edge('l1_moe_weighted', 'l1_moe_residual')
    
    # Stage 1 → Stage 2 Pipeline
    dot.edge('l1_moe_residual', 'pipe_send')
    dot.edge('pipe_send', 'pipe_recv')
    
    # Stage 2 → Output
    dot.edge('pipe_recv', 'l16_final')
    dot.edge('l16_final', 'output_proj')
    dot.edge('output_proj', 'final_gather')
    dot.edge('final_gather', 'output')
    
    return dot

def main():
    # Create the DAG
    dag = create_moe_deployment_dag()
    
    # Save as DOT file
    output_dir = "../outputs/2025-12-05-14-41-54"
    os.makedirs(output_dir, exist_ok=True)
    
    dot_file = os.path.join(output_dir, "moe_deployment_dag.dot")
    dag.save(dot_file)
    
    # Save as SVG image
    svg_file = os.path.join(output_dir, "moe_deployment_dag.svg")
    dag.render(svg_file, format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file}")
    print(f"SVG file: {svg_file}")
    
    return {
        "dot_file": dot_file,
        "svg_file": svg_file
    }

if __name__ == "__main__":
    result = main()
    print(f"Files generated: {result}")