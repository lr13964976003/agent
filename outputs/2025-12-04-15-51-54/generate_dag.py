#!/usr/bin/env python3

import os
from graphviz import Digraph

def create_llm_deployment_dag():
    """
    Create a detailed DAG for the 30B MoE model deployment across 128 GPUs
    with expert parallelism (64-way), tensor parallelism (2-way), and pipeline parallelism (4-way)
    """
    
    # Create the DAG
    dot = Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='100,100', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=10240, hidden=1024]\\nOutput: [batch_size=128, seq_len=10240, hidden=1024]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 0: Layers 0-3 (GPUs 0-31)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0: Layers 0-3\\nGPUs 0-31', style='rounded,filled', fillcolor='lightgray')
        
        # Layer 0 (Attention Layer)
        stage0.node('layer0_attn_norm', 'Layer 0 Attention Norm\\nGPU: 0-31\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        stage0.node('layer0_qkv_proj', 'Layer 0 QKV Projection\\nGPU: 0-31\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 3072]', 
                   shape='rectangle')
        
        stage0.node('layer0_attn_score', 'Layer 0 Attention Score\\nGPU: 0-31\\nInput: [128, 16, 10240, 64]\\nOutput: [128, 16, 10240, 10240]', 
                   shape='rectangle')
        
        stage0.node('layer0_attn_softmax', 'Layer 0 Attention Softmax\\nGPU: 0-31\\nInput: [128, 16, 10240, 10240]\\nOutput: [128, 16, 10240, 10240]', 
                   shape='rectangle')
        
        stage0.node('layer0_attn_output', 'Layer 0 Attention Output\\nGPU: 0-31\\nInput: [128, 16, 10240, 64]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        # Layer 0 MoE
        stage0.node('layer0_moe_norm', 'Layer 0 MoE Norm\\nGPU: 0-31\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        # Expert routing (gate)
        stage0.node('layer0_gate', 'Layer 0 Expert Gate\\nGPU: 0-31\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 64]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Expert 0 (GPUs 0,1)
        stage0.node('layer0_exp0_gpu0_col', 'Expert 0 Col Linear GPU0\\nGPU: 0\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle', fillcolor='lightcoral')
        stage0.node('layer0_exp0_gpu1_col', 'Expert 0 Col Linear GPU1\\nGPU: 1\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle', fillcolor='lightcoral')
        
        stage0.node('layer0_exp0_allreduce', 'Expert 0 All-Reduce\\nGPU: 0,1\\nInput: [128, 10240, 2048]\\nOutput: [128, 10240, 2048]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        stage0.node('layer0_exp0_gpu0_gelu', 'Expert 0 GELU GPU0\\nGPU: 0\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle', fillcolor='lightcoral')
        stage0.node('layer0_exp0_gpu1_gelu', 'Expert 0 GELU GPU1\\nGPU: 1\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle', fillcolor='lightcoral')
        
        stage0.node('layer0_exp0_gpu0_row', 'Expert 0 Row Linear GPU0\\nGPU: 0\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 512]', 
                   shape='rectangle', fillcolor='lightcoral')
        stage0.node('layer0_exp0_gpu1_row', 'Expert 0 Row Linear GPU1\\nGPU: 1\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 512]', 
                   shape='rectangle', fillcolor='lightcoral')
        
        stage0.node('layer0_exp0_allreduce2', 'Expert 0 Final All-Reduce\\nGPU: 0,1\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='ellipse', fillcolor='lightgreen')
        
        # Expert aggregation
        stage0.node('layer0_exp_agg', 'Layer 0 Expert Aggregation\\nGPU: 0-31\\nInput: [128, 10240, 1024, 64]\\nOutput: [128, 10240, 1024]', 
                   shape='parallelogram', fillcolor='lightyellow')
        
        # Layer 0 output
        stage0.node('layer0_output', 'Layer 0 Output\\nGPU: 0-31\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
    
    # Pipeline Stage 1: Layers 4-7 (GPUs 32-63)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1: Layers 4-7\\nGPUs 32-63', style='rounded,filled', fillcolor='lightgray')
        
        # Similar structure for Layer 4
        stage1.node('layer4_attn_norm', 'Layer 4 Attention Norm\\nGPU: 32-63\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        stage1.node('layer4_output', 'Layer 4 Output\\nGPU: 32-63\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
    
    # Pipeline Stage 2: Layers 8-11 (GPUs 64-95)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Pipeline Stage 2: Layers 8-11\\nGPUs 64-95', style='rounded,filled', fillcolor='lightgray')
        
        stage2.node('layer8_attn_norm', 'Layer 8 Attention Norm\\nGPU: 64-95\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        stage2.node('layer8_output', 'Layer 8 Output\\nGPU: 64-95\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
    
    # Pipeline Stage 3: Layers 12-15 (GPUs 96-127)
    with dot.subgraph(name='cluster_stage3') as stage3:
        stage3.attr(label='Pipeline Stage 3: Layers 12-15\\nGPUs 96-127', style='rounded,filled', fillcolor='lightgray')
        
        stage3.node('layer12_attn_norm', 'Layer 12 Attention Norm\\nGPU: 96-127\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
        
        stage3.node('layer12_output', 'Layer 12 Output\\nGPU: 96-127\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
                   shape='rectangle')
    
    # Output node
    dot.node('output', 'Final Output\\nInput: [128, 10240, 1024]\\nOutput: [128, 10240, 1024]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect nodes
    # Input to Stage 0
    dot.edge('input', 'layer0_attn_norm')
    
    # Stage 0 internal connections
    dot.edge('layer0_attn_norm', 'layer0_qkv_proj')
    dot.edge('layer0_qkv_proj', 'layer0_attn_score')
    dot.edge('layer0_attn_score', 'layer0_attn_softmax')
    dot.edge('layer0_attn_softmax', 'layer0_attn_output')
    dot.edge('layer0_attn_output', 'layer0_moe_norm')
    dot.edge('layer0_moe_norm', 'layer0_gate')
    
    # Expert 0 routing (dashed line for gate selection)
    dot.edge('layer0_gate', 'layer0_exp0_gpu0_col', style='dashed', label='expert_select')
    dot.edge('layer0_gate', 'layer0_exp0_gpu1_col', style='dashed', label='expert_select')
    
    # Expert 0 computation
    dot.edge('layer0_exp0_gpu0_col', 'layer0_exp0_allreduce')
    dot.edge('layer0_exp0_gpu1_col', 'layer0_exp0_allreduce')
    dot.edge('layer0_exp0_allreduce', 'layer0_exp0_gpu0_gelu')
    dot.edge('layer0_exp0_allreduce', 'layer0_exp0_gpu1_gelu')
    dot.edge('layer0_exp0_gpu0_gelu', 'layer0_exp0_gpu0_row')
    dot.edge('layer0_exp0_gpu1_gelu', 'layer0_exp0_gpu1_row')
    dot.edge('layer0_exp0_gpu0_row', 'layer0_exp0_allreduce2')
    dot.edge('layer0_exp0_gpu1_row', 'layer0_exp0_allreduce2')
    
    # Expert aggregation
    dot.edge('layer0_exp0_allreduce2', 'layer0_exp_agg')
    dot.edge('layer0_exp_agg', 'layer0_output')
    
    # Pipeline communication (between stages)
    dot.edge('layer0_output', 'layer4_attn_norm', style='dashed', label='pipeline_send')
    dot.edge('layer4_output', 'layer8_attn_norm', style='dashed', label='pipeline_send')
    dot.edge('layer8_output', 'layer12_attn_norm', style='dashed', label='pipeline_send')
    dot.edge('layer12_output', 'output')
    
    return dot

def main():
    # Create output directory if it doesn't exist
    os.makedirs('../outputs/2025-12-04-15-51-54', exist_ok=True)
    
    # Generate the DAG
    dag = create_llm_deployment_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-04-15-51-54/llm_deployment_dag.dot'
    with open(dot_file_path, 'w') as f:
        f.write(dag.source)
    
    # Save as SVG
    svg_file_path = '../outputs/2025-12-04-15-51-54/llm_deployment_dag.svg'
    dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")
    
    return {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path
    }

if __name__ == "__main__":
    main()