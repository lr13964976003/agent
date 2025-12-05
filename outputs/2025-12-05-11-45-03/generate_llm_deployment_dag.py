#!/usr/bin/env python3

import graphviz
import os

def create_llm_deployment_dag():
    """
    Create a comprehensive DAG for 30B MoE model deployment with:
    - PP 2 × TP 2 × DP 2 = 8 GPUs
    - 16 layers total, 8 per pipeline stage
    - 64 experts per layer, 16 per GPU (4-way EP)
    - 2-way tensor parallelism for attention
    """
    
    dot = graphviz.Digraph(comment='30B MoE Model Deployment DAG')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node shapes
    # Ellipses for communication
    # Rectangles for computation  
    # Parallelograms for routing/aggregation
    
    # Input node
    dot.node('input', 'Input\\nBatch: [128, 1024, 1024]\\nSeq_len=1024, Hidden=1024', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Data Parallel split
    dot.node('dp_split', 'DP Split\\n2-way Data Parallel', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    # Stage 0: Layers 0-7 (GPUs 0,1,4,5)
    # DP Group 0: GPUs 0,1
    # DP Group 1: GPUs 4,5
    
    # Create nodes for each layer in Stage 0
    for layer in range(8):
        # Attention computation nodes for GPU 0 (TP group 0)
        dot.node(f'attn_qkv_s0_l{layer}_gpu0', 
                f'Attention QKV\\nGPU-0\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_score_s0_l{layer}_gpu0', 
                f'Attention Score\\nGPU-0\\nInput: [64,8,1024,64]\\nOutput: [64,8,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
                
        dot.node(f'attn_softmax_s0_l{layer}_gpu0', 
                f'Attention Softmax\\nGPU-0\\nInput: [64,8,1024,1024]\\nOutput: [64,8,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
                
        dot.node(f'attn_out_s0_l{layer}_gpu0', 
                f'Attention Output\\nGPU-0\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention computation nodes for GPU 1 (TP group 1)
        dot.node(f'attn_qkv_s0_l{layer}_gpu1', 
                f'Attention QKV\\nGPU-1\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_score_s0_l{layer}_gpu1', 
                f'Attention Score\\nGPU-1\\nInput: [64,8,1024,64]\\nOutput: [64,8,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
                
        dot.node(f'attn_softmax_s0_l{layer}_gpu1', 
                f'Attention Softmax\\nGPU-1\\nInput: [64,8,1024,1024]\\nOutput: [64,8,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
                
        dot.node(f'attn_out_s0_l{layer}_gpu1', 
                f'Attention Output\\nGPU-1\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention all-reduce communication
        dot.node(f'attn_allreduce_s0_l{layer}', 
                f'All-Reduce\\nTP Group 0,1\\n[64,1024,1024]',
                shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # MoE Gate (routing) - dashed line for gate selection
        dot.node(f'gate_s0_l{layer}_gpu0', 
                f'MoE Gate\\nGPU-0\\nInput: [64,1024,1024]\\nOutput: [64,1024,4]\\nTop-1 routing',
                shape='parallelogram', style='filled, dashed', fillcolor='orange')
                
        dot.node(f'gate_s0_l{layer}_gpu1', 
                f'MoE Gate\\nGPU-1\\nInput: [64,1024,1024]\\nOutput: [64,1024,4]\\nTop-1 routing',
                shape='parallelogram', style='filled, dashed', fillcolor='orange')
        
        # Expert computation (local experts per GPU)
        dot.node(f'expert_s0_l{layer}_gpu0', 
                f'MoE Experts 0-15\\nGPU-0\\nInput: [64,1024,1024]\\nOutput: [64,1024,2048]\\n16 experts',
                shape='rectangle', style='filled', fillcolor='lightblue')
                
        dot.node(f'expert_s0_l{layer}_gpu1', 
                f'MoE Experts 16-31\\nGPU-1\\nInput: [64,1024,1024]\\nOutput: [64,1024,2048]\\n16 experts',
                shape='rectangle', style='filled', fillcolor='lightblue')
        
        # All-to-all communication for expert parallel
        dot.node(f'all2all_s0_l{layer}_gpu01', 
                f'All-to-All\\nEP GPUs 0,1\\nToken routing',
                shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # MoE output aggregation
        dot.node(f'moe_agg_s0_l{layer}_gpu0', 
                f'MoE Aggregate\\nGPU-0\\nInput: [64,1024,2048]\\nOutput: [64,1024,1024]',
                shape='parallelogram', style='filled', fillcolor='orange')
                
        dot.node(f'moe_agg_s0_l{layer}_gpu1', 
                f'MoE Aggregate\\nGPU-1\\nInput: [64,1024,2048]\\nOutput: [64,1024,1024]',
                shape='parallelogram', style='filled', fillcolor='orange')
    
    # Stage 1: Layers 8-15 (GPUs 2,3,6,7)
    # Similar structure for stage 1
    for layer in range(8):
        # GPU 2: TP group 0, DP group 0
        dot.node(f'attn_qkv_s1_l{layer}_gpu2', 
                f'Attention QKV\\nGPU-2\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_out_s1_l{layer}_gpu2', 
                f'Attention Output\\nGPU-2\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # GPU 3: TP group 1, DP group 0
        dot.node(f'attn_qkv_s1_l{layer}_gpu3', 
                f'Attention QKV\\nGPU-3\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_out_s1_l{layer}_gpu3', 
                f'Attention Output\\nGPU-3\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention all-reduce for stage 1
        dot.node(f'attn_allreduce_s1_l{layer}', 
                f'All-Reduce\\nTP Group 2,3\\n[64,1024,1024]',
                shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # MoE for stage 1
        dot.node(f'gate_s1_l{layer}_gpu2', 
                f'MoE Gate\\nGPU-2\\nInput: [64,1024,1024]\\nOutput: [64,1024,4]\\nTop-1 routing',
                shape='parallelogram', style='filled, dashed', fillcolor='orange')
                
        dot.node(f'expert_s1_l{layer}_gpu2', 
                f'MoE Experts 0-15\\nGPU-2\\nInput: [64,1024,1024]\\nOutput: [64,1024,2048]\\n16 experts',
                shape='rectangle', style='filled', fillcolor='lightblue')
                
        dot.node(f'moe_agg_s1_l{layer}_gpu2', 
                f'MoE Aggregate\\nGPU-2\\nInput: [64,1024,2048]\\nOutput: [64,1024,1024]',
                shape='parallelogram', style='filled', fillcolor='orange')
        
        # GPU 6,7 for DP group 1 in stage 1
        dot.node(f'attn_qkv_s1_l{layer}_gpu6', 
                f'Attention QKV\\nGPU-6\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_out_s1_l{layer}_gpu6', 
                f'Attention Output\\nGPU-6\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_qkv_s1_l{layer}_gpu7', 
                f'Attention QKV\\nGPU-7\\nInput: [64,1024,1024]\\nOutput: [64,1024,512]\\n8 heads, 64d each',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        dot.node(f'attn_out_s1_l{layer}_gpu7', 
                f'Attention Output\\nGPU-7\\nInput: [64,1024,512]\\nOutput: [64,1024,1024]',
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Attention all-reduce for stage 1, DP group 1
        dot.node(f'attn_allreduce_s1_l{layer}_dp1', 
                f'All-Reduce\\nTP Group 6,7\\n[64,1024,1024]',
                shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # MoE for stage 1, DP group 1
        dot.node(f'gate_s1_l{layer}_gpu6', 
                f'MoE Gate\\nGPU-6\\nInput: [64,1024,1024]\\nOutput: [64,1024,4]\\nTop-1 routing',
                shape='parallelogram', style='filled, dashed', fillcolor='orange')
                
        dot.node(f'expert_s1_l{layer}_gpu6', 
                f'MoE Experts 32-47\\nGPU-6\\nInput: [64,1024,1024]\\nOutput: [64,1024,2048]\\n16 experts',
                shape='rectangle', style='filled', fillcolor='lightblue')
                
        dot.node(f'moe_agg_s1_l{layer}_gpu6', 
                f'MoE Aggregate\\nGPU-6\\nInput: [64,1024,2048]\\nOutput: [64,1024,1024]',
                shape='parallelogram', style='filled', fillcolor='orange')
    
    # Pipeline communication between stages
    dot.node(f'pipe_comm_s0_s1', 
            f'Pipeline Comm\\nStage0→Stage1\\nMicro-batch transfer',
            shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Data parallel reduction
    dot.node(f'dp_reduce', 
            f'Data Parallel Reduce\\n2-way DP\\nGradient sync',
            shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Output node
    dot.node('output', 'Output\\nBatch: [128, 1024, 1024]\\nSeq_len=1024, Hidden=1024', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create edges - representative flow through the system
    # Input flow
    dot.edge('input', 'dp_split')
    
    # Layer 0, Stage 0, DP Group 0 flow
    dot.edge('dp_split', 'attn_qkv_s0_l0_gpu0')
    dot.edge('dp_split', 'attn_qkv_s0_l0_gpu1')
    
    # Attention flow for layer 0
    dot.edge('attn_qkv_s0_l0_gpu0', 'attn_score_s0_l0_gpu0')
    dot.edge('attn_score_s0_l0_gpu0', 'attn_softmax_s0_l0_gpu0')
    dot.edge('attn_softmax_s0_l0_gpu0', 'attn_out_s0_l0_gpu0')
    
    dot.edge('attn_qkv_s0_l0_gpu1', 'attn_score_s0_l0_gpu1')
    dot.edge('attn_score_s0_l0_gpu1', 'attn_softmax_s0_l0_gpu1')
    dot.edge('attn_softmax_s0_l0_gpu1', 'attn_out_s0_l0_gpu1')
    
    # All-reduce communication
    dot.edge('attn_out_s0_l0_gpu0', 'attn_allreduce_s0_l0')
    dot.edge('attn_out_s0_l0_gpu1', 'attn_allreduce_s0_l0')
    
    # MoE flow for layer 0
    dot.edge('attn_allreduce_s0_l0', 'gate_s0_l0_gpu0')
    dot.edge('attn_allreduce_s0_l0', 'gate_s0_l0_gpu1')
    dot.edge('attn_allreduce_s0_l0', 'expert_s0_l0_gpu0')
    dot.edge('attn_allreduce_s0_l0', 'expert_s0_l0_gpu1')
    
    # Expert computation and communication
    dot.edge('gate_s0_l0_gpu0', 'all2all_s0_l0_gpu01')
    dot.edge('gate_s0_l0_gpu1', 'all2all_s0_l0_gpu01')
    dot.edge('expert_s0_l0_gpu0', 'all2all_s0_l0_gpu01')
    dot.edge('expert_s0_l0_gpu1', 'all2all_s0_l0_gpu01')
    dot.edge('all2all_s0_l0_gpu01', 'moe_agg_s0_l0_gpu0')
    dot.edge('all2all_s0_l0_gpu01', 'moe_agg_s0_l0_gpu1')
    
    # Pipeline communication to stage 1
    dot.edge('moe_agg_s0_l7_gpu0', 'pipe_comm_s0_s1')
    dot.edge('moe_agg_s0_l7_gpu1', 'pipe_comm_s0_s1')
    dot.edge('pipe_comm_s0_s1', 'attn_qkv_s1_l0_gpu2')
    dot.edge('pipe_comm_s0_s1', 'attn_qkv_s1_l0_gpu3')
    
    # Final output flow
    dot.edge('moe_agg_s1_l7_gpu2', 'dp_reduce')
    dot.edge('moe_agg_s1_l7_gpu3', 'dp_reduce')
    dot.edge('moe_agg_s1_l7_gpu6', 'dp_reduce')
    dot.edge('moe_agg_s1_l7_gpu7', 'dp_reduce')
    dot.edge('dp_reduce', 'output')
    
    # Save the DAG
    output_path = '../outputs/2025-12-05-11-45-03/llm_deployment_dag'
    dot.render(output_path, format='dot', cleanup=False)
    dot.render(output_path, format='svg', cleanup=False)
    
    return output_path + '.dot', output_path + '.svg'

if __name__ == '__main__':
    dot_path, svg_path = create_llm_deployment_dag()
    print(f"Generated DAG files:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")