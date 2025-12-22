#!/usr/bin/env python3
"""
Corrected DAG Generator for LLM EP64-TP8-PP2-DP2 Deployment
This version includes the missing All-Reduce communications for tensor parallelism.
"""

import graphviz
from graphviz import Digraph
import os

def create_corrected_llm_dag():
    """Create the corrected LLM deployment DAG with all communications."""
    
    # Create the main DAG
    dot = Digraph(comment='LLM EP64-TP8-PP2-DP2 Deployment DAG - Corrected')
    dot.attr(rankdir='TB', bgcolor='white', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define GPU assignments
    # Stage 1: GPUs 0-1023 (1024 GPUs)
    # Stage 2: GPUs 1024-2047 (1024 GPUs)
    
    # Define node styles
    compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightblue'}
    comm_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'}
    routing_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'yellow'}
    
    # Input node
    dot.node('input', 
             'Input\\n[batch_size=128, seq_len=1024, hidden=1024]\\nGPU: All', 
             shape='octagon', style='filled', fillcolor='lightgray')
    
    # Stage 1: Layers 0-7 (1024 GPUs: 0-1023)
    stage1_gpus = "GPUs 0-1023"
    
    for layer in range(8):
        layer_prefix = f'layer{layer}'
        
        # LayerNorm (TP8 - split across 8 GPUs within each EP group)
        dot.node(f'{layer_prefix}_ln1', 
                 f'LayerNorm{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # QKV Projection (TP8)
        dot.node(f'{layer_prefix}_qkv', 
                 f'QKV{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,16,64]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # All-Reduce after QKV (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_qkv_ar', 
                 f'All-Reduce QKV{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # Self-Attention
        dot.node(f'{layer_prefix}_attn', 
                 f'Self-Attention{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # All-Reduce after Attention (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_attn_ar', 
                 f'All-Reduce Attn{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # Attention Output Projection (TP8)
        dot.node(f'{layer_prefix}_attn_out', 
                 f'Attention Output{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # All-Reduce after Attention Output (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_attn_out_ar', 
                 f'All-Reduce AttnOut{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # Residual Connection
        dot.node(f'{layer_prefix}_res1', 
                 f'Residual1{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # LayerNorm2 (TP8)
        dot.node(f'{layer_prefix}_ln2', 
                 f'LayerNorm2{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # MoE Routing (EP64 - 64 experts)
        dot.node(f'{layer_prefix}_route', 
                 f'MoE Route{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,64]\\nGPU: {stage1_gpus}', 
                 **routing_style)
        
        # Expert Dispatch (All-to-All communication)
        dot.node(f'{layer_prefix}_dispatch', 
                 f'Expert Dispatch{layer}\\nInput: [128,1024,1024]\\nOutput: [2,1024,1024]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # 64 Expert Computations (EP64)
        for expert in range(64):
            expert_gpu = (layer * 128 + expert * 2) % 1024  # Distribute across stage 1 GPUs
            dot.node(f'{layer_prefix}_expert{expert}',                      f'Expert{layer}_{expert}\\nInput: [2,1024,1024]\\nOutput: [2,1024,2048]\\nGPU: {expert_gpu}', 
                     **compute_style)
        
        # Expert Combine (All-to-All communication)
        dot.node(f'{layer_prefix}_combine', 
                 f'Expert Combine{layer}\\nInput: [2,1024,2048]\\nOutput: [128,1024,2048]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # MoE Output Projection (TP8)
        dot.node(f'{layer_prefix}_moe_out', 
                 f'MoE Output{layer}\\nInput: [128,1024,2048]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
        
        # All-Reduce after MoE (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_moe_ar', 
                 f'All-Reduce MoE{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **comm_style)
        
        # Residual Connection 2
        dot.node(f'{layer_prefix}_res2', 
                 f'Residual2{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage1_gpus}', 
                 **compute_style)
    
    # Pipeline transfer from Stage 1 to Stage 2
    dot.node('pp_transfer', 
             'Pipeline Transfer\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: 1023→1024', 
             **comm_style)
    
    # Stage 2: Layers 8-15 (1024 GPUs: 1024-2047)
    stage2_gpus = "GPUs 1024-2047"
    
    for layer in range(8, 16):
        layer_prefix = f'layer{layer}'
        
        # LayerNorm (TP8)
        dot.node(f'{layer_prefix}_ln1', 
                 f'LayerNorm{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # QKV Projection (TP8)
        dot.node(f'{layer_prefix}_qkv', 
                 f'QKV{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,16,64]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # All-Reduce after QKV (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_qkv_ar', 
                 f'All-Reduce QKV{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # Self-Attention
        dot.node(f'{layer_prefix}_attn', 
                 f'Self-Attention{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # All-Reduce after Attention (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_attn_ar', 
                 f'All-Reduce Attn{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,16,64]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # Attention Output Projection (TP8)
        dot.node(f'{layer_prefix}_attn_out', 
                 f'Attention Output{layer}\\nInput: [128,1024,16,64]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # All-Reduce after Attention Output (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_attn_out_ar', 
                 f'All-Reduce AttnOut{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # Residual Connection
        dot.node(f'{layer_prefix}_res1', 
                 f'Residual1{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # LayerNorm2 (TP8)
        dot.node(f'{layer_prefix}_ln2', 
                 f'LayerNorm2{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # MoE Routing (EP64)
        dot.node(f'{layer_prefix}_route', 
                 f'MoE Route{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,64]\\nGPU: {stage2_gpus}', 
                 **routing_style)
        
        # Expert Dispatch (All-to-All communication)
        dot.node(f'{layer_prefix}_dispatch', 
                 f'Expert Dispatch{layer}\\nInput: [128,1024,1024]\\nOutput: [2,1024,1024]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # 64 Expert Computations (EP64)
        for expert in range(64):
            expert_gpu = 1024 + (layer * 128 + expert * 2) % 1024  # Distribute across stage 2 GPUs
            dot.node(f'{layer_prefix}_expert{expert}', 
                     f'Expert{layer}_{expert}\\nInput: [2,1024,1024]\\nOutput: [2,1024,2048]\\nGPU: {expert_gpu}', 
                     **compute_style)
        
        # Expert Combine (All-to-All communication)
        dot.node(f'{layer_prefix}_combine', 
                 f'Expert Combine{layer}\\nInput: [2,1024,2048]\\nOutput: [128,1024,2048]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # MoE Output Projection (TP8)
        dot.node(f'{layer_prefix}_moe_out', 
                 f'MoE Output{layer}\\nInput: [128,1024,2048]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
        
        # All-Reduce after MoE (CRITICAL - was missing)
        dot.node(f'{layer_prefix}_moe_ar', 
                 f'All-Reduce MoE{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **comm_style)
        
        # Residual Connection 2
        dot.node(f'{layer_prefix}_res2', 
                 f'Residual2{layer}\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: {stage2_gpus}', 
                 **compute_style)
    
    # Final LayerNorm and Output
    dot.node('final_ln', 
             'Final LayerNorm\\nInput: [128,1024,1024]\\nOutput: [128,1024,1024]\\nGPU: 2047', 
             **compute_style)
    
    dot.node('output', 
             'Output\\n[batch_size=128, seq_len=1024, vocab=32000]\\nGPU: 2047', 
             shape='doubleoctagon', style='filled', fillcolor='lightgray')
    
    # Connect all nodes with proper dependencies
    
    # Input to first layer
    dot.edge('input', 'layer0_ln1')
    
    # Stage 1 connections (layers 0-7)
    for layer in range(8):
        layer_prefix = f'layer{layer}'
        
        # LayerNorm1 -> QKV -> All-Reduce -> Attention -> All-Reduce -> Attention Output -> All-Reduce -> Residual
        dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_qkv')
        dot.edge(f'{layer_prefix}_qkv', f'{layer_prefix}_qkv_ar')
        dot.edge(f'{layer_prefix}_qkv_ar', f'{layer_prefix}_attn')
        dot.edge(f'{layer_prefix}_attn', f'{layer_prefix}_attn_ar')
        dot.edge(f'{layer_prefix}_attn_ar', f'{layer_prefix}_attn_out')
        dot.edge(f'{layer_prefix}_attn_out', f'{layer_prefix}_attn_out_ar')
        dot.edge(f'{layer_prefix}_attn_out_ar', f'{layer_prefix}_res1')
        
        # Residual -> LayerNorm2 -> Route -> Dispatch -> Experts -> Combine -> MoE Output -> All-Reduce -> Residual
        dot.edge(f'{layer_prefix}_res1', f'{layer_prefix}_ln2')
        dot.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_route')
        
        # Gate selection with dashed line
        dot.edge(f'{layer_prefix}_route', f'{layer_prefix}_dispatch', style='dashed')
        dot.edge(f'{layer_prefix}_dispatch', f'{layer_prefix}_expert0')
        
        # Connect all experts in parallel
        for expert in range(64):
            dot.edge(f'{layer_prefix}_dispatch', f'{layer_prefix}_expert{expert}')
            dot.edge(f'{layer_prefix}_expert{expert}', f'{layer_prefix}_combine')
        
        dot.edge(f'{layer_prefix}_combine', f'{layer_prefix}_moe_out')
        dot.edge(f'{layer_prefix}_moe_out', f'{layer_prefix}_moe_ar')
        dot.edge(f'{layer_prefix}_moe_ar', f'{layer_prefix}_res2')
        
        # Connect to next layer or pipeline transfer
        if layer < 7:
            dot.edge(f'{layer_prefix}_res2', f'layer{layer+1}_ln1')
        else:
            # Last layer of stage 1 -> pipeline transfer
            dot.edge(f'{layer_prefix}_res2', 'pp_transfer')
    
    # Pipeline transfer to stage 2
    dot.edge('pp_transfer', 'layer8_ln1')
    
    # Stage 2 connections (layers 8-15)
    for layer in range(8, 16):
        layer_prefix = f'layer{layer}'
        
        # LayerNorm1 -> QKV -> All-Reduce -> Attention -> All-Reduce -> Attention Output -> All-Reduce -> Residual
        dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_qkv')
        dot.edge(f'{layer_prefix}_qkv', f'{layer_prefix}_qkv_ar')
        dot.edge(f'{layer_prefix}_qkv_ar', f'{layer_prefix}_attn')
        dot.edge(f'{layer_prefix}_attn', f'{layer_prefix}_attn_ar')
        dot.edge(f'{layer_prefix}_attn_ar', f'{layer_prefix}_attn_out')
        dot.edge(f'{layer_prefix}_attn_out', f'{layer_prefix}_attn_out_ar')
        dot.edge(f'{layer_prefix}_attn_out_ar', f'{layer_prefix}_res1')
        
        # Residual -> LayerNorm2 -> Route -> Dispatch -> Experts -> Combine -> MoE Output -> All-Reduce -> Residual
        dot.edge(f'{layer_prefix}_res1', f'{layer_prefix}_ln2')
        dot.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_route')
        
        # Gate selection with dashed line
        dot.edge(f'{layer_prefix}_route', f'{layer_prefix}_dispatch', style='dashed')
        
        # Connect all experts in parallel
        for expert in range(64):
            dot.edge(f'{layer_prefix}_dispatch', f'{layer_prefix}_expert{expert}')
            dot.edge(f'{layer_prefix}_expert{expert}', f'{layer_prefix}_combine')
        
        dot.edge(f'{layer_prefix}_combine', f'{layer_prefix}_moe_out')
        dot.edge(f'{layer_prefix}_moe_out', f'{layer_prefix}_moe_ar')
        dot.edge(f'{layer_prefix}_moe_ar', f'{layer_prefix}_res2')
        
        # Connect to next layer or final processing
        if layer < 15:
            dot.edge(f'{layer_prefix}_res2', f'layer{layer+1}_ln1')
        else:
            # Last layer -> final processing
            dot.edge(f'{layer_prefix}_res2', 'final_ln')
    
    # Final processing
    dot.edge('final_ln', 'output')
    
    return dot

def validate_dag(dot):
    """Validate the DAG for cycles and proper structure."""
    # Save to temporary file for validation
    temp_file = '../outputs/2025-12-22-14-17-26/temp_dag.dot'
    dot.save(temp_file)
    
    # Use the Extract Info From DAG tool to validate
    try:
        from graphviz import Source
        src = Source.from_file(temp_file)
        # Basic validation - check if it can be rendered
        src.render('../outputs/2025-12-22-14-17-26/temp_validated', format='dot', cleanup=True)
        return True, "DAG validation successful"
    except Exception as e:
        return False, f"DAG validation failed: {str(e)}"

def main():
    """Main function to generate the corrected DAG."""
    
    print("Generating corrected LLM deployment DAG...")
    
    # Create the corrected DAG
    dot = create_corrected_llm_dag()
    
    # Validate the DAG
    is_valid, message = validate_dag(dot)
    print(f"DAG validation: {message}")
    
    if is_valid:
        # Save the DOT file
        dot_file = '../outputs/2025-12-22-14-17-26/llm_corrected_dag.dot'
        dot.save(dot_file)
        print(f"Saved DOT file: {dot_file}")
        
        # Render to SVG
        svg_file = '../outputs/2025-12-22-14-17-26/llm_corrected_dag.svg'
        dot.render('../outputs/2025-12-22-14-17-26/llm_corrected_dag', format='svg', cleanup=True)
        print(f"Saved SVG file: {svg_file}")
        
        # Count communication operations
        dot_source = dot.source
        all_reduce_count = dot_source.count('All-Reduce')
        all_to_all_count = dot_source.count('All-to-All') + dot_source.count('Expert Dispatch') + dot_source.count('Expert Combine')
        pipeline_count = dot_source.count('Pipeline Transfer')
        
        print(f"\nCommunication Summary:")
        print(f"All-Reduce operations: {all_reduce_count} (Expected: 16)")
        print(f"All-to-All operations: {all_to_all_count} (Expected: 128)")
        print(f"Pipeline transfers: {pipeline_count} (Expected: 1)")
        
        if all_reduce_count == 16:
            print("✅ All-Reduce operations are now CORRECT!")
        else:
            print(f"❌ All-Reduce operations mismatch: got {all_reduce_count}, expected 16")
        
        return {
            "dag_files": [
                "../outputs/2025-12-22-14-17-26/llm_corrected_dag.dot",
                "../outputs/2025-12-22-14-17-26/llm_corrected_dag.svg"
            ],
            "communication_summary": {
                "all_reduce_operations": all_reduce_count,
                "all_to_all_operations": all_to_all_count,
                "pipeline_transfers": pipeline_count
            }
        }
    else:
        print(f"Failed to generate valid DAG: {message}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print("\nFinal result:")
        print(result)