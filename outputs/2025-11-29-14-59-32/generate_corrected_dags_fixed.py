#!/usr/bin/env python3
"""
Generate corrected DAGs for the LLM deployment project.
This script creates proper tensor parallelism and pipeline parallelism DAGs
that fix the critical connection errors identified in previous versions.
"""

import os

def generate_baseline_tensor_pipeline_dag():
    """Generate corrected baseline tensor pipeline DAG with proper connections"""
    
    dot_content = """// Corrected Baseline Tensor Pipeline Parallelism DAG (Fixed Connections)
digraph baseline_tensor_pipeline_fixed {
    bgcolor=white fontname=Arial rankdir=TB
    node [fontname=Arial fontsize=10]
    node [fillcolor=lightblue shape=rectangle style=filled]
    
    // Input node
    input [label="Input\\nBatch: 128\\nSeq: 10000\\nHidden: 4096" fillcolor=lightgreen shape=parallelogram]
    output [label="Output\\nBatch: 128\\nSeq: 10000\\nHidden: 4096" fillcolor=lightgreen shape=parallelogram]
    
"""
    
    # Generate nodes for all 16 layers
    for layer in range(16):
        # Determine GPU range based on pipeline stage
        if layer < 8:
            gpu_start, gpu_end = 0, 8  # Pipeline stage 0: GPUs 0-7
            pipeline_stage = 0
        else:
            gpu_start, gpu_end = 8, 16  # Pipeline stage 1: GPUs 8-15
            pipeline_stage = 1
        
        # Layer input distribution
        dot_content += f'    layer_{layer}_input_dist [label="Layer {layer}\\nInput Distribution" fillcolor=lightyellow shape=parallelogram]\n'
        
        # QKV projections for each GPU with tensor parallelism
        for gpu in range(gpu_start, gpu_end):
            tp_rank = gpu % 8
            dot_content += f'    layer_{layer}_qkv_gpu{gpu} [label="Layer {layer} QKV Proj\\nGPU {gpu}\\nTP Rank {tp_rank}\\n[4096→12288]" fillcolor=lightblue]\n'
        
        # AllGather for QKV
        dot_content += f'    layer_{layer}_allgather_qkv [label="Layer {layer}\\nAllGather QKV\\nTP=8" fillcolor=orange shape=ellipse]\n'
        
        # Attention computation
        dot_content += f'    layer_{layer}_attention [label="Layer {layer}\\nAttention\\n32×128 dim\\nSoftmax" fillcolor=lightblue]\n'
        
        # Attention output projections for each GPU
        for gpu in range(gpu_start, gpu_end):
            tp_rank = gpu % 8
            dot_content += f'    layer_{layer}_attn_out_gpu{gpu} [label="Layer {layer} Attn Out\\nGPU {gpu}\\nTP Rank {tp_rank}\\n[4096→4096]" fillcolor=lightblue]\n'
        
        # AllReduce for attention
        dot_content += f'    layer_{layer}_allreduce_attn [label="Layer {layer}\\nAllReduce Attn\\nTP=8" fillcolor=orange shape=ellipse]\n'
        
        # Residual and LayerNorm
        dot_content += f'    layer_{layer}_residual1 [label="Layer {layer}\\nResidual Add" fillcolor=lightyellow shape=parallelogram]\n'
        dot_content += f'    layer_{layer}_ln1 [label="Layer {layer}\\nLayerNorm 1\\nγ,β: 4096" fillcolor=lightblue]\n'
        
        # MLP FC1 projections for each GPU
        for gpu in range(gpu_start, gpu_end):
            tp_rank = gpu % 8
            dot_content += f'    layer_{layer}_mlp_fc1_gpu{gpu} [label="Layer {layer} MLP FC1\\nGPU {gpu}\\nTP Rank {tp_rank}\\n[4096→16384]" fillcolor=lightblue]\n'
        
        # AllGather for MLP
        dot_content += f'    layer_{layer}_allgather_mlp [label="Layer {layer}\\nAllGather MLP\\nTP=8" fillcolor=orange shape=ellipse]\n'
        
        # GELU activation
        dot_content += f'    layer_{layer}_gelu [label="Layer {layer}\\nGELU Activation" fillcolor=lightblue]\n'
        
        # MLP FC2 projections for each GPU
        for gpu in range(gpu_start, gpu_end):
            tp_rank = gpu % 8
            dot_content += f'    layer_{layer}_mlp_fc2_gpu{gpu} [label="Layer {layer} MLP FC2\\nGPU {gpu}\\nTP Rank {tp_rank}\\n[16384→4096]" fillcolor=lightblue]\n'
        
        # AllReduce for MLP
        dot_content += f'    layer_{layer}_allreduce_mlp [label="Layer {layer}\\nAllReduce MLP\\nTP=8" fillcolor=orange shape=ellipse]\n'
        
        # Final residual and LayerNorm
        dot_content += f'    layer_{layer}_residual2 [label="Layer {layer}\\nResidual Add" fillcolor=lightyellow shape=parallelogram]\n'
        dot_content += f'    layer_{layer}_ln2 [label="Layer {layer}\\nLayerNorm 2\\nγ,β: 4096" fillcolor=lightblue]\n'
        
        # Output distribution
        if layer < 15:
            dot_content += f'    layer_{layer}_output_dist [label="Layer {layer}\\nOutput Distribution" fillcolor=lightyellow shape=parallelogram]\n'
    
    # Now generate the connections - this is where the critical fixes are
    dot_content += "\n    // Connections - Fixed topology for proper tensor parallelism\n"
    
    # Input connections
    dot_content += "    input -> layer_0_input_dist\n"
    
    for layer in range(16):
        # Determine GPU range
        if layer < 8:
            gpu_start, gpu_end = 0, 8
        else:
            gpu_start, gpu_end = 8, 16
        
        # Input distribution to QKV projections
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_input_dist -> layer_{layer}_qkv_gpu{gpu}\n'
        
        # QKV projections to AllGather
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_qkv_gpu{gpu} -> layer_{layer}_allgather_qkv\n'
        
        # AllGather to attention
        dot_content += f'    layer_{layer}_allgather_qkv -> layer_{layer}_attention\n'
        
        # CRITICAL FIX: Attention to ALL attention output projections (not just GPU7)
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_attention -> layer_{layer}_attn_out_gpu{gpu}\n'
        
        # Attention outputs to AllReduce
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_attn_out_gpu{gpu} -> layer_{layer}_allreduce_attn\n'
        
        # AllReduce to residual
        dot_content += f'    layer_{layer}_allreduce_attn -> layer_{layer}_residual1\n'
        dot_content += f'    layer_{layer}_input_dist -> layer_{layer}_residual1\n'
        
        # Residual to LayerNorm
        dot_content += f'    layer_{layer}_residual1 -> layer_{layer}_ln1\n'
        
        # LayerNorm to MLP FC1 projections
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_ln1 -> layer_{layer}_mlp_fc1_gpu{gpu}\n'
        
        # MLP FC1 to AllGather
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_mlp_fc1_gpu{gpu} -> layer_{layer}_allgather_mlp\n'
        
        # AllGather to GELU
        dot_content += f'    layer_{layer}_allgather_mlp -> layer_{layer}_gelu\n'
        
        # CRITICAL FIX: GELU to ALL MLP FC2 projections (not just GPU7)
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_gelu -> layer_{layer}_mlp_fc2_gpu{gpu}\n'
        
        # MLP FC2 to AllReduce
        for gpu in range(gpu_start, gpu_end):
            dot_content += f'    layer_{layer}_mlp_fc2_gpu{gpu} -> layer_{layer}_allreduce_mlp\n'
        
        # AllReduce to final residual
        dot_content += f'    layer_{layer}_allreduce_mlp -> layer_{layer}_residual2\n'
        dot_content += f'    layer_{layer}_residual1 -> layer_{layer}_residual2\n'
        
        # Final residual to LayerNorm
        dot_content += f'    layer_{layer}_residual2 -> layer_{layer}_ln2\n'
        
        # Connect to next layer or output
        if layer < 15:
            dot_content += f'    layer_{layer}_ln2 -> layer_{layer}_output_dist\n'
            dot_content += f'    layer_{layer}_output_dist -> layer_{layer+1}_input_dist\n'
        else:
            dot_content += f'    layer_{layer}_ln2 -> output\n'
    
    dot_content += "}\n"
    
    return dot_content

def generate_proposed_layer_wise_dag():
    """Generate corrected proposed layer-wise DAG with proper connections"""
    
    dot_content = """// Corrected Proposed Layer-wise DAG (Fixed Connections)
digraph proposed_layer_wise_fixed {
    bgcolor=white fontname=Arial rankdir=TB
    node [fontname=Arial fontsize=10]
    node [fillcolor=lightblue shape=rectangle style=filled]
    
    // Input node
    input [label="Input\\nBatch: 128\\nSeq: 10000\\nHidden: 4096" fillcolor=lightgreen shape=parallelogram]
    output [label="Output\\nBatch: 128\\nSeq: 10000\\nHidden: 4096" fillcolor=lightgreen shape=parallelogram]
    
"""
    
    # Generate nodes for all 16 layers across 8 GPUs (2 layers per GPU)
    for layer in range(16):
        gpu_id = layer // 2  # 2 layers per GPU
        
        # Layer input distribution
        dot_content += f'    layer_{layer}_input_dist [label="Layer {layer}\\nInput Distribution\\nGPU {gpu_id}" fillcolor=lightyellow shape=parallelogram]\n'
        
        # QKV projections (single GPU per layer in layer-wise partitioning)
        dot_content += f'    layer_{layer}_qkv_gpu{gpu_id} [label="Layer {layer} QKV Proj\\nGPU {gpu_id}\\n[4096→12288]" fillcolor=lightblue]\n'
        
        # Attention computation
        dot_content += f'    layer_{layer}_attention [label="Layer {layer}\\nAttention\\n32×128 dim\\nSoftmax" fillcolor=lightblue]\n'
        
        # Attention output projection
        dot_content += f'    layer_{layer}_attn_out_gpu{gpu_id} [label="Layer {layer} Attn Out\\nGPU {gpu_id}\\n[4096→4096]" fillcolor=lightblue]\n'
        
        # Residual and LayerNorm
        dot_content += f'    layer_{layer}_residual1 [label="Layer {layer}\\nResidual Add" fillcolor=lightyellow shape=parallelogram]\n'
        dot_content += f'    layer_{layer}_ln1 [label="Layer {layer}\\nLayerNorm 1\\nγ,β: 4096" fillcolor=lightblue]\n'
        
        # MLP FC1
        dot_content += f'    layer_{layer}_mlp_fc1_gpu{gpu_id} [label="Layer {layer} MLP FC1\\nGPU {gpu_id}\\n[4096→16384]" fillcolor=lightblue]\n'
        
        # GELU activation
        dot_content += f'    layer_{layer}_gelu [label="Layer {layer}\\nGELU Activation" fillcolor=lightblue]\n'
        
        # MLP FC2
        dot_content += f'    layer_{layer}_mlp_fc2_gpu{gpu_id} [label="Layer {layer} MLP FC2\\nGPU {gpu_id}\\n[16384→4096]" fillcolor=lightblue]\n'
        
        # Final residual and LayerNorm
        dot_content += f'    layer_{layer}_residual2 [label="Layer {layer}\\nResidual Add" fillcolor=lightyellow shape=parallelogram]\n'
        dot_content += f'    layer_{layer}_ln2 [label="Layer {layer}\\nLayerNorm 2\\nγ,β: 4096" fillcolor=lightblue]\n'
        
        # Output distribution (except last layer)
        if layer < 15:
            next_gpu_id = (layer + 1) // 2
            dot_content += f'    layer_{layer}_output_dist [label="Layer {layer}\\nOutput Distribution\\nGPU {gpu_id}→{next_gpu_id}" fillcolor=lightyellow shape=parallelogram]\n'
            dot_content += f'    layer_{layer}_pipeline_send [label="Layer {layer}→{layer+1}\\nPipeline Send\\nGPU {gpu_id}→{next_gpu_id}" fillcolor=red shape=ellipse]\n'
            dot_content += f'    layer_{layer+1}_pipeline_recv [label="Layer {layer+1}\\nPipeline Recv\\nGPU {next_gpu_id}" fillcolor=red shape=ellipse]\n'
    
    # Now generate the connections
    dot_content += "\n    // Connections - Fixed for proper layer-wise execution\n"
    
    # Input connections
    dot_content += "    input -> layer_0_input_dist\n"
    
    for layer in range(16):
        gpu_id = layer // 2
        
        # Input distribution to QKV
        dot_content += f'    layer_{layer}_input_dist -> layer_{layer}_qkv_gpu{gpu_id}\n'
        
        # QKV to attention
        dot_content += f'    layer_{layer}_qkv_gpu{gpu_id} -> layer_{layer}_attention\n'
        
        # Attention to attention output
        dot_content += f'    layer_{layer}_attention -> layer_{layer}_attn_out_gpu{gpu_id}\n'
        
        # Attention output to residual
        dot_content += f'    layer_{layer}_attn_out_gpu{gpu_id} -> layer_{layer}_residual1\n'
        dot_content += f'    layer_{layer}_input_dist -> layer_{layer}_residual1\n'
        
        # Residual to LayerNorm
        dot_content += f'    layer_{layer}_residual1 -> layer_{layer}_ln1\n'
        
        # LayerNorm to MLP FC1
        dot_content += f'    layer_{layer}_ln1 -> layer_{layer}_mlp_fc1_gpu{gpu_id}\n'
        
        # MLP FC1 to GELU
        dot_content += f'    layer_{layer}_mlp_fc1_gpu{gpu_id} -> layer_{layer}_gelu\n'
        
        # GELU to MLP FC2
        dot_content += f'    layer_{layer}_gelu -> layer_{layer}_mlp_fc2_gpu{gpu_id}\n'
        
        # MLP FC2 to final residual
        dot_content += f'    layer_{layer}_mlp_fc2_gpu{gpu_id} -> layer_{layer}_residual2\n'
        dot_content += f'    layer_{layer}_residual1 -> layer_{layer}_residual2\n'
        
        # Final residual to LayerNorm (FIXED: removed self-loop)
        dot_content += f'    layer_{layer}_residual2 -> layer_{layer}_ln2\n'
        
        # Connect to next layer or output
        if layer < 15:
            # Pipeline communication
            dot_content += f'    layer_{layer}_ln2 -> layer_{layer}_output_dist\n'
            dot_content += f'    layer_{layer}_output_dist -> layer_{layer}_pipeline_send\n'
            dot_content += f'    layer_{layer}_pipeline_send -> layer_{layer+1}_pipeline_recv\n'
            dot_content += f'    layer_{layer+1}_pipeline_recv -> layer_{layer+1}_input_dist\n'
        else:
            dot_content += f'    layer_{layer}_ln2 -> output\n'
    
    dot_content += "}\n"
    
    return dot_content

def main():
    """Generate both corrected DAGs"""
    
    print("Generating corrected baseline tensor pipeline DAG...")
    baseline_content = generate_baseline_tensor_pipeline_dag()
    
    baseline_path = "../outputs/2025-11-29-14-59-32/final_baseline_tensor_pipeline_dag.dot"
    with open(baseline_path, 'w') as f:
        f.write(baseline_content)
    print(f"Baseline DAG written to: {baseline_path}")
    
    print("\nGenerating corrected proposed layer-wise DAG...")
    proposed_content = generate_proposed_layer_wise_dag()
    
    proposed_path = "../outputs/2025-11-29-14-59-32/final_proposed_layer_wise_dag.dot"
    with open(proposed_path, 'w') as f:
        f.write(proposed_content)
    print(f"Proposed DAG written to: {proposed_path}")
    
    # Generate SVG images using Graphviz
    print("\nGenerating SVG images...")
    
    # Baseline SVG
    baseline_svg = "../outputs/2025-11-29-14-59-32/final_baseline_tensor_pipeline_dag.svg"
    os.system(f'dot -Tsvg {baseline_path} -o {baseline_svg}')
    print(f"Baseline SVG generated: {baseline_svg}")
    
    # Proposed SVG  
    proposed_svg = "../outputs/2025-11-29-14-59-32/final_proposed_layer_wise_dag.svg"
    os.system(f'dot -Tsvg {proposed_path} -o {proposed_svg}')
    print(f"Proposed SVG generated: {proposed_svg}")
    
    print("\nAll DAGs generated successfully!")

if __name__ == "__main__":
    main()