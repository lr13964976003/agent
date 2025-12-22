#!/usr/bin/env python3

import graphviz
import os

def generate_llm_dag():
    """Generate comprehensive DAG for LLM EP64-TP8-PP2-DP2 deployment"""
    
    # Create main graph
    dot = graphviz.Digraph(comment='LLM EP64-TP8-PP2-DP2 Deployment DAG')
    dot.attr(rankdir='TB', size='100,200')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute nodes
    
    # GPU assignments
    stage1_gpus = list(range(0, 1024))  # GPUs 0-1023
    stage2_gpus = list(range(1024, 2048))  # GPUs 1024-2047
    
    # Model configuration
    batch_size = 128
    seq_len = 1024
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    num_experts = 64
    expert_hidden = 2048
    
    # ============ STAGE 1 (GPUs 0-1023) ============
    
    # Input node
    dot.node('input', f'Input\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # LayerNorm 1 (Stage 1)
    dot.node('layernorm1_s1', f'LayerNorm 1\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # QKV projection with TP8 - requires All-Reduce
    dot.node('qkv_proj_s1', f'QKV Projection (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for QKV TP8
    dot.node('allreduce_qkv_s1', f'All-Reduce QKV\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # Self-Attention
    dot.node('attention_s1', f'Self-Attention\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # Attention output projection with TP8 - requires All-Reduce
    dot.node('attn_out_proj_s1', f'Attention Output Proj (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for Attention Output TP8
    dot.node('allreduce_attn_out_s1', f'All-Reduce Attention Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # MoE Routing (Gate)
    dot.node('moe_gate_s1', f'MoE Gate (Router)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='parallelogram', fillcolor='orange')
    
    # Expert dispatch (All-to-All)
    dot.node('expert_dispatch_s1', f'Expert Dispatch (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # Expert computations (64 experts distributed across Stage 1 GPUs)
    for expert_id in range(32):  # First 32 experts in Stage 1
        gpu_start = expert_id * 32
        gpu_end = (expert_id + 1) * 32 - 1
        dot.node(f'expert_{expert_id}_s1', f'Expert {expert_id}\\nInput: [batch={batch_size//64}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//64}, seq={seq_len}, hidden={expert_hidden}]\\nGPUs: {gpu_start}-{gpu_end}', 
                 fillcolor='lightblue')
    
    # Expert combine (All-to-All)
    dot.node('expert_combine_s1', f'Expert Combine (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={expert_hidden}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # MoE output projection with TP8 - requires All-Reduce
    dot.node('moe_out_proj_s1', f'MoE Output Proj (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for MoE Output TP8
    dot.node('allreduce_moe_out_s1', f'All-Reduce MoE Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # LayerNorm 2 (Stage 1)
    dot.node('layernorm2_s1', f'LayerNorm 2\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[0]}-{stage1_gpus[-1]}', 
             fillcolor='lightblue')
    
    # ============ STAGE 2 (GPUs 1024-2047) ============
    
    # Pipeline transfer from Stage 1 to Stage 2
    dot.node('pp_transfer_s1_s2', f'Pipeline Transfer S1→S2\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage1_gpus[-1]} → {stage2_gpus[0]}', 
             shape='ellipse', fillcolor='red')
    
    # LayerNorm 1 (Stage 2)
    dot.node('layernorm1_s2', f'LayerNorm 1\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # QKV projection with TP8 - requires All-Reduce
    dot.node('qkv_proj_s2', f'QKV Projection (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for QKV TP8
    dot.node('allreduce_qkv_s2', f'All-Reduce QKV\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # Self-Attention
    dot.node('attention_s2', f'Self-Attention\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # Attention output projection with TP8 - requires All-Reduce
    dot.node('attn_out_proj_s2', f'Attention Output Proj (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for Attention Output TP8
    dot.node('allreduce_attn_out_s2', f'All-Reduce Attention Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # MoE Routing (Gate)
    dot.node('moe_gate_s2', f'MoE Gate (Router)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='parallelogram', fillcolor='orange')
    
    # Expert dispatch (All-to-All)
    dot.node('expert_dispatch_s2', f'Expert Dispatch (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # Expert computations (64 experts distributed across Stage 2 GPUs)
    for expert_id in range(32, 64):  # Last 32 experts in Stage 2
        gpu_start = 1024 + (expert_id - 32) * 32
        gpu_end = 1024 + ((expert_id - 32) + 1) * 32 - 1
        dot.node(f'expert_{expert_id}_s2', f'Expert {expert_id}\\nInput: [batch={batch_size//64}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size//64}, seq={seq_len}, hidden={expert_hidden}]\\nGPUs: {gpu_start}-{gpu_end}', 
                 fillcolor='lightblue')
    
    # Expert combine (All-to-All)
    dot.node('expert_combine_s2', f'Expert Combine (All-to-All)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={expert_hidden}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # MoE output projection with TP8 - requires All-Reduce
    dot.node('moe_out_proj_s2', f'MoE Output Proj (TP8)\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # All-Reduce for MoE Output TP8
    dot.node('allreduce_moe_out_s2', f'All-Reduce MoE Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             shape='ellipse', fillcolor='yellow')
    
    # LayerNorm 2 (Stage 2)
    dot.node('layernorm2_s2', f'LayerNorm 2\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: {stage2_gpus[0]}-{stage2_gpus[-1]}', 
             fillcolor='lightblue')
    
    # Output node
    dot.node('output', f'Output\\nInput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nOutput: [batch={batch_size}, seq={seq_len}, hidden={hidden_size}]\\nGPUs: All', 
             shape='ellipse', fillcolor='lightgreen')
    
    # ============ EDGES (DEPENDENCIES) ============
    
    # Stage 1 flow
    dot.edge('input', 'layernorm1_s1')
    dot.edge('layernorm1_s1', 'qkv_proj_s1')
    dot.edge('qkv_proj_s1', 'allreduce_qkv_s1')
    dot.edge('allreduce_qkv_s1', 'attention_s1')
    dot.edge('attention_s1', 'attn_out_proj_s1')
    dot.edge('attn_out_proj_s1', 'allreduce_attn_out_s1')
    dot.edge('allreduce_attn_out_s1', 'moe_gate_s1')
    dot.edge('moe_gate_s1', 'expert_dispatch_s1', style='dashed')  # Gate selection with dashed line
    dot.edge('expert_dispatch_s1', 'expert_0_s1')
    
    # Connect experts in Stage 1
    for expert_id in range(1, 32):
        dot.edge('expert_dispatch_s1', f'expert_{expert_id}_s1')
    
    # Connect experts to combine
    for expert_id in range(32):
        dot.edge(f'expert_{expert_id}_s1', 'expert_combine_s1')
    
    dot.edge('expert_combine_s1', 'moe_out_proj_s1')
    dot.edge('moe_out_proj_s1', 'allreduce_moe_out_s1')
    dot.edge('allreduce_moe_out_s1', 'layernorm2_s1')
    dot.edge('layernorm2_s1', 'pp_transfer_s1_s2')
    
    # Stage 2 flow
    dot.edge('pp_transfer_s1_s2', 'layernorm1_s2')
    dot.edge('layernorm1_s2', 'qkv_proj_s2')
    dot.edge('qkv_proj_s2', 'allreduce_qkv_s2')
    dot.edge('allreduce_qkv_s2', 'attention_s2')
    dot.edge('attention_s2', 'attn_out_proj_s2')
    dot.edge('attn_out_proj_s2', 'allreduce_attn_out_s2')
    dot.edge('allreduce_attn_out_s2', 'moe_gate_s2')
    dot.edge('moe_gate_s2', 'expert_dispatch_s2', style='dashed')  # Gate selection with dashed line
    dot.edge('expert_dispatch_s2', 'expert_32_s2')
    
    # Connect experts in Stage 2
    for expert_id in range(33, 64):
        dot.edge('expert_dispatch_s2', f'expert_{expert_id}_s2')
    
    # Connect experts to combine
    for expert_id in range(32, 64):
        dot.edge(f'expert_{expert_id}_s2', 'expert_combine_s2')
    
    dot.edge('expert_combine_s2', 'moe_out_proj_s2')
    dot.edge('moe_out_proj_s2', 'allreduce_moe_out_s2')
    dot.edge('allreduce_moe_out_s2', 'layernorm2_s2')
    dot.edge('layernorm2_s2', 'output')
    
    return dot

def main():
    # Generate the comprehensive DAG
    dag = generate_llm_dag()
    
    # Save DOT file
    dot_path = '../outputs/2025-12-22-14-17-26/llm_comprehensive_dag.dot'
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Render to SVG
    svg_path = '../outputs/2025-12-22-14-17-26/llm_comprehensive_dag.svg'
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Generated comprehensive DAG:")
    print(f"DOT file: {dot_path}")
    print(f"SVG file: {svg_path}")
    
    # Count operations for verification
    dot_content = dag.source
    allreduce_count = dot_content.count('All-Reduce')
    alltoall_count = dot_content.count('All-to-All')
    
    print(f"\nOperation counts:")
    print(f"All-Reduce operations: {allreduce_count}")
    print(f"All-to-All operations: {alltoall_count}")
    
    return {
        "dot_path": dot_path,
        "svg_path": svg_path,
        "allreduce_count": allreduce_count,
        "alltoall_count": alltoall_count
    }

if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")