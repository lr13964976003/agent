#!/usr/bin/env python3
"""
DAG Generator for 30B-MoE Model Deployment
Generates Graphviz code for the complete deployment DAG
"""

import os
from graphviz import Digraph

def create_moe_deployment_dag():
    """Create complete DAG for 30B-MoE model deployment"""
    
    # Create directed graph
    dot = Digraph(comment='30B-MoE Model Deployment DAG')
    dot.attr(rankdir='TB', size='30,40', fontsize='10')
    
    # Global settings
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Data Parallel Split
    dot.node('dp_split', 'DP Split\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024] x 8', 
             shape='parallelogram', fillcolor='yellow')
    dot.edge('input', 'dp_split')
    
    # Pipeline Stage 0 (Layers 0-3) - GPU 0-127
    create_pipeline_stage(dot, 0, 0, 127)
    
    # Pipeline Stage 1 (Layers 4-7) - GPU 128-255
    create_pipeline_stage(dot, 1, 128, 255)
    
    # Pipeline Stage 2 (Layers 8-11) - GPU 256-383
    create_pipeline_stage(dot, 2, 256, 383)
    
    # Pipeline Stage 3 (Layers 12-15) - GPU 384-511
    create_pipeline_stage(dot, 3, 384, 511)
    
    # Connect pipeline stages
    for dp in range(8):
        dot.edge(f'stage2_pipe_{dp}', f'stage3_pipe_{dp}')
    
    # Final output nodes for each DP
    for dp in range(8):
        dot.node(f'stage3_final_{dp}', f'Final Output DP{dp}\\nGPU {(384 + dp * 16) // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
                 shape='ellipse', fillcolor='orange', style='dashed')
        dot.edge(f'stage3_layer15_dp{dp}_output', f'stage3_final_{dp}')
    
    # Data Parallel Merge
    dot.node('dp_merge', 'DP Merge\\nInput: [batch_size=16, seq_len=1024, hidden=1024] x 8\\nOutput: [batch_size=128, seq_len=1024, hidden=1024]', 
             shape='parallelogram', fillcolor='yellow')
    
    for dp in range(8):
        dot.edge(f'stage3_final_{dp}', 'dp_merge')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=128, seq_len=1024, hidden=1024]\\nOutput: [batch_size=128, seq_len=1024, vocab=32000]', 
             shape='ellipse', fillcolor='lightgreen')
    dot.edge('dp_merge', 'output')
    
    return dot

def create_pipeline_stage(dot, stage_id, gpu_start, gpu_end):
    """Create a pipeline stage with 4 layers"""
    
    layer_start = stage_id * 4
    layer_end = layer_start + 4
    
    for dp in range(8):
        # Stage input
        if stage_id == 0:
            dot.edge('dp_split', f'stage{stage_id}_layer{layer_start}_dp{dp}')
        else:
            # Connect from previous stage
            prev_stage = stage_id - 1
            dot.edge(f'stage{prev_stage}_pipe_{dp}', f'stage{stage_id}_layer{layer_start}_dp{dp}')
        
        for layer in range(layer_start, layer_end):
            create_transformer_layer(dot, stage_id, layer, dp, gpu_start, gpu_end)
        
        # Mark pipeline output
        dot.node(f'stage{stage_id}_pipe_{dp}', f'Stage {stage_id} Output DP{dp}\\nGPU {(gpu_start + dp * 16) // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
                 shape='ellipse', fillcolor='orange', style='dashed')
        dot.edge(f'stage{stage_id}_layer{layer_end-1}_dp{dp}_output', f'stage{stage_id}_pipe_{dp}')

def create_transformer_layer(dot, stage_id, layer_id, dp_id, gpu_start, gpu_end):
    """Create a complete transformer layer with attention and MoE"""
    
    gpu_base = gpu_start + dp_id * 16  # 16 GPUs per DP group
    
    # Layer input
    if layer_id == stage_id * 4:
        # First layer in stage
        input_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}'
        dot.node(input_node, f'Layer {layer_id} Input DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
                shape='ellipse', fillcolor='lightgreen')
    else:
        # Connect from previous layer
        prev_layer = layer_id - 1
        dot.edge(f'stage{stage_id}_layer{prev_layer}_dp{dp_id}_output', f'stage{stage_id}_layer{layer_id}_dp{dp_id}')
        input_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}'
    
    # Layer Norm 1
    ln1_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_ln1'
    dot.node(ln1_node, f'Layer Norm 1\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(input_node, ln1_node)
    
    # Attention - Multi-head self attention
    create_attention_block(dot, stage_id, layer_id, dp_id, gpu_base, ln1_node)
    
    # Add & Norm
    add_norm_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_add_norm'
    dot.node(add_norm_node, f'Add & Norm\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(f'stage{stage_id}_layer{layer_id}_dp{dp_id}_attn_output', add_norm_node)
    dot.edge(input_node, add_norm_node, style='dashed')  # Residual connection
    
    # Layer Norm 2
    ln2_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_ln2'
    dot.node(ln2_node, f'Layer Norm 2\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(add_norm_node, ln2_node)
    
    # MoE Block
    create_moe_block(dot, stage_id, layer_id, dp_id, gpu_base, ln2_node)
    
    # Final Add & Output
    final_add = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_final_add'
    dot.node(final_add, f'Final Add\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(f'stage{stage_id}_layer{layer_id}_dp{dp_id}_moe_output', final_add)
    dot.edge(add_norm_node, final_add, style='dashed')  # Residual connection
    
    # Layer output
    output_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_output'
    dot.node(output_node, f'Layer {layer_id} Output DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='orange')
    dot.edge(final_add, output_node)

def create_attention_block(dot, stage_id, layer_id, dp_id, gpu_base, input_node):
    """Create multi-head self-attention block with operator granularity"""
    
    # QKV Linear
    qkv_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_qkv'
    dot.node(qkv_node, f'QKV Linear\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=3072]')
    dot.edge(input_node, qkv_node)
    
    # Split QKV
    split_qkv = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_split_qkv'
    dot.node(split_qkv, f'Split QKV\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=3072]\\nOutput: Q[16,1024,1024], K[16,1024,1024], V[16,1024,1024]', 
             shape='parallelogram', fillcolor='yellow')
    dot.edge(qkv_node, split_qkv)
    
    # Reshape for multi-head
    reshape_q = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_reshape_q'
    dot.node(reshape_q, f'Reshape Q\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, heads=16, d_k=64]')
    dot.edge(split_qkv, reshape_q)
    
    reshape_k = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_reshape_k'
    dot.node(reshape_k, f'Reshape K\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, heads=16, d_k=64]')
    dot.edge(split_qkv, reshape_k)
    
    reshape_v = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_reshape_v'
    dot.node(reshape_v, f'Reshape V\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, heads=16, d_k=64]')
    dot.edge(split_qkv, reshape_v)
    
    # Attention scores
    attn_scores = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_attn_scores'
    dot.node(attn_scores, f'Attention Scores\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: Q[16,1024,16,64], K[16,1024,16,64]\\nOutput: [batch_size=16, heads=16, seq_len=1024, seq_len=1024]')
    dot.edge(reshape_q, attn_scores)
    dot.edge(reshape_k, attn_scores)
    
    # Scale
    scale_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_scale'
    dot.node(scale_node, f'Scale (1/√d_k)\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, heads=16, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=16, heads=16, seq_len=1024, seq_len=1024]')
    dot.edge(attn_scores, scale_node)
    
    # Softmax
    softmax_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_softmax'
    dot.node(softmax_node, f'Softmax\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, heads=16, seq_len=1024, seq_len=1024]\\nOutput: [batch_size=16, heads=16, seq_len=1024, seq_len=1024]')
    dot.edge(scale_node, softmax_node)
    
    # Attention weights application
    attn_weights = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_attn_weights'
    dot.node(attn_weights, f'Apply Attention Weights\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: Weights[16,16,1024,1024], V[16,1024,16,64]\\nOutput: [batch_size=16, heads=16, seq_len=1024, d_k=64]')
    dot.edge(softmax_node, attn_weights)
    dot.edge(reshape_v, attn_weights)
    
    # Concat heads
    concat_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_concat'
    dot.node(concat_node, f'Concat Heads\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, heads=16, seq_len=1024, d_k=64]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(attn_weights, concat_node)
    
    # Output linear
    out_linear = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_attn_out'
    dot.node(out_linear, f'Attention Output Linear\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
    dot.edge(concat_node, out_linear)
    
    # Final attention output
    attn_output = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_attn_output'
    dot.node(attn_output, f'Attention Output\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='orange')
    dot.edge(out_linear, attn_output)

def create_moe_block(dot, stage_id, layer_id, dp_id, gpu_base, input_node):
    """Create MoE block with gate and expert routing"""
    
    # Gate
    gate_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_gate'
    dot.node(gate_node, f'MoE Gate\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, num_experts=64]')
    dot.edge(input_node, gate_node)
    
    # Expert routing (dashed line for gate selection)
    for expert_id in range(4):  # 4 experts per GPU
        local_expert = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_expert{expert_id}'
        expert_gpu = gpu_base + (expert_id % 16)  # Distribute across EP group
        dot.node(local_expert, f'Expert {expert_id}\\nL{layer_id} DP{dp_id}\\nGPU {expert_gpu // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]')
        dot.edge(input_node, local_expert)
        dot.edge(gate_node, local_expert, style='dashed', label=f'select expert {expert_id}')
    
    # Expert aggregation
    agg_node = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_expert_agg'
    dot.node(agg_node, f'Expert Aggregation\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024] x 4\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
             shape='parallelogram', fillcolor='yellow')
    
    for expert_id in range(4):
        dot.edge(f'stage{stage_id}_layer{layer_id}_dp{dp_id}_expert{expert_id}', agg_node)
    
    # MoE output
    moe_output = f'stage{stage_id}_layer{layer_id}_dp{dp_id}_moe_output'
    dot.node(moe_output, f'MoE Output\\nL{layer_id} DP{dp_id}\\nGPU {gpu_base // 16}\\nInput: [batch_size=16, seq_len=1024, hidden=1024]\\nOutput: [batch_size=16, seq_len=1024, hidden=1024]', 
             shape='ellipse', fillcolor='orange')
    dot.edge(agg_node, moe_output)

if __name__ == "__main__":
    # Create DAG
    dag = create_moe_deployment_dag()
    
    # Save DOT file
    dot_file = "../outputs/2025-12-05-15-55-09/moe_deployment_dag.dot"
    dag.save(dot_file)
    
    # Save SVG image
    svg_file = "../outputs/2025-12-05-15-55-09/moe_deployment_dag.svg"
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"DAG saved to: {dot_file}")
    print(f"SVG saved to: {svg_file}")