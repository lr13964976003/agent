#!/usr/bin/env python3
"""
Generate baseline DAG (TP=8, PP=2) for 4-layer dense model
Based on deployment_config.json baseline configuration
"""

import graphviz
import os

def create_baseline_dag():
    """Create baseline TP=8, PP=2 DAG"""
    dot = graphviz.Digraph(comment='Baseline TP=8 PP=2 DAG')
    dot.attr(rankdir='TB', splines='ortho', ranksep='1.5', nodesep='0.8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Global attributes
    batch_size = 'batch_size=1024'
    seq_len = 'seq_len=4096'  # Default sequence length
    
    # Input node
    dot.node('input', 
             f'Total Input\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0 (GPUs 0-7)
    dot.attr('node', fillcolor='lightblue')
    
    # Embedding (on GPU 0 of stage 0)
    dot.node('embed_0', 
             'Embedding Layer\\nInput: [{batch_size}, {seq_len}]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0', 
             shape='rectangle')
    
    # Positional Encoding
    dot.node('pos_enc_0', 
             'Positional Encoding\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='rectangle')
    
    # Layer 0 components (TP=8 across stage 0 GPUs)
    
    # Layer 0 - Attention components
    dot.node('l0_q_proj', 
             'Layer 0 Q Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_k_proj', 
             'Layer 0 K Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_v_proj', 
             'Layer 0 V Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_attention', 
             'Layer 0 Multi-Head Attention\\nQ/K/V Input: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_dropout', 
             'Layer 0 Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='rectangle')
    
    dot.node('l0_add_norm1', 
             'Layer 0 Add & Norm 1\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='hexagon')
    
    # Layer 0 - FFN components
    dot.node('l0_ffn_up', 
             'Layer 0 FFN Up\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_ffn_gate', 
             'Layer 0 FFN Gate\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_ffn_down', 
             'Layer 0 FFN Down\\nInput: [{batch_size}, {seq_len}, ffn=16384]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l0_ffn_dropout', 
             'Layer 0 FFN Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='rectangle')
    
    dot.node('l0_add_norm2', 
             'Layer 0 Add & Norm 2\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='hexagon')
    
    # Pipeline communication between stages
    dot.node('pipeline_send_0', 
             'Pipeline Send\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nFrom: stage_0\\nTo: stage_1', 
             shape='parallelogram', fillcolor='yellow')
    
    # Stage 1 (GPUs 8-15)
    
    # Layer 1 (on stage 0)
    dot.node('l1_q_proj', 
             'Layer 1 Q Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_k_proj', 
             'Layer 1 K Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_v_proj', 
             'Layer 1 V Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_attention', 
             'Layer 1 Multi-Head Attention\\nQ/K/V Input: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_dropout', 
             'Layer 1 Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='rectangle')
    
    dot.node('l1_add_norm1', 
             'Layer 1 Add & Norm 1\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='hexagon')
    
    dot.node('l1_ffn_up', 
             'Layer 1 FFN Up\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_ffn_gate', 
             'Layer 1 FFN Gate\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_ffn_down', 
             'Layer 1 FFN Down\\nInput: [{batch_size}, {seq_len}, ffn=16384]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7 (TP=8)', 
             shape='rectangle')
    
    dot.node('l1_ffn_dropout', 
             'Layer 1 FFN Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='rectangle')
    
    dot.node('l1_add_norm2', 
             'Layer 1 Add & Norm 2\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_0-7', 
             shape='hexagon')
    
    # Layer 2 (on stage 1)
    dot.node('l2_q_proj', 
             'Layer 2 Q Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_k_proj', 
             'Layer 2 K Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_v_proj', 
             'Layer 2 V Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_attention', 
             'Layer 2 Multi-Head Attention\\nQ/K/V Input: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_dropout', 
             'Layer 2 Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_add_norm1', 
             'Layer 2 Add & Norm 1\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='hexagon', fillcolor='lightcoral')
    
    dot.node('l2_ffn_up', 
             'Layer 2 FFN Up\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_ffn_gate', 
             'Layer 2 FFN Gate\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_ffn_down', 
             'Layer 2 FFN Down\\nInput: [{batch_size}, {seq_len}, ffn=16384]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_ffn_dropout', 
             'Layer 2 FFN Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l2_add_norm2', 
             'Layer 2 Add & Norm 2\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='hexagon', fillcolor='lightcoral')
    
    # Layer 3 (on stage 1)
    dot.node('l3_q_proj', 
             'Layer 3 Q Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_k_proj', 
             'Layer 3 K Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_v_proj', 
             'Layer 3 V Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_attention', 
             'Layer 3 Multi-Head Attention\\nQ/K/V Input: [{batch_size}, {seq_len}, heads=32, d_k=128]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_dropout', 
             'Layer 3 Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_add_norm1', 
             'Layer 3 Add & Norm 1\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='hexagon', fillcolor='lightcoral')
    
    dot.node('l3_ffn_up', 
             'Layer 3 FFN Up\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_ffn_gate', 
             'Layer 3 FFN Gate\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, ffn=16384]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_ffn_down', 
             'Layer 3 FFN Down\\nInput: [{batch_size}, {seq_len}, ffn=16384]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15 (TP=8)', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_ffn_dropout', 
             'Layer 3 FFN Dropout\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='rectangle', fillcolor='lightcoral')
    
    dot.node('l3_add_norm2', 
             'Layer 3 Add & Norm 2\\nInput1: [{batch_size}, {seq_len}, hidden=4096]\\nInput2: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_8-15', 
             shape='hexagon', fillcolor='lightcoral')
    
    # Final output layer
    dot.node('output_norm', 
             'Final Layer Norm\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, hidden=4096]\\nGPU: gpu_15', 
             shape='rectangle', fillcolor='lightgreen')
    
    dot.node('output_proj', 
             'Output Projection\\nInput: [{batch_size}, {seq_len}, hidden=4096]\\nOutput: [{batch_size}, {seq_len}, vocab=32000]\\nGPU: gpu_15', 
             shape='rectangle', fillcolor='lightgreen')
    
    dot.node('output', 
             'Total Output\\nInput: [{batch_size}, {seq_len}, vocab=32000]\\nGPU: gpu_15', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create edges
    dot.edge('input', 'embed_0')
    dot.edge('embed_0', 'pos_enc_0')
    
    # Layer 0
    dot.edge('pos_enc_0', 'l0_q_proj')
    dot.edge('pos_enc_0', 'l0_k_proj')
    dot.edge('pos_enc_0', 'l0_v_proj')
    dot.edge('l0_q_proj', 'l0_attention')
    dot.edge('l0_k_proj', 'l0_attention')
    dot.edge('l0_v_proj', 'l0_attention')
    dot.edge('l0_attention', 'l0_dropout')
    dot.edge('l0_dropout', 'l0_add_norm1')
    dot.edge('pos_enc_0', 'l0_add_norm1')
    dot.edge('l0_add_norm1', 'l0_ffn_up')
    dot.edge('l0_add_norm1', 'l0_ffn_gate')
    dot.edge('l0_ffn_up', 'l0_ffn_down')
    dot.edge('l0_ffn_gate', 'l0_ffn_down')
    dot.edge('l0_ffn_down', 'l0_ffn_dropout')
    dot.edge('l0_ffn_dropout', 'l0_add_norm2')
    dot.edge('l0_add_norm1', 'l0_add_norm2')
    
    # Layer 1
    dot.edge('l0_add_norm2', 'l1_q_proj')
    dot.edge('l0_add_norm2', 'l1_k_proj')
    dot.edge('l0_add_norm2', 'l1_v_proj')
    dot.edge('l1_q_proj', 'l1_attention')
    dot.edge('l1_k_proj', 'l1_attention')
    dot.edge('l1_v_proj', 'l1_attention')
    dot.edge('l1_attention', 'l1_dropout')
    dot.edge('l1_dropout', 'l1_add_norm1')
    dot.edge('l0_add_norm2', 'l1_add_norm1')
    dot.edge('l1_add_norm1', 'l1_ffn_up')
    dot.edge('l1_add_norm1', 'l1_ffn_gate')
    dot.edge('l1_ffn_up', 'l1_ffn_down')
    dot.edge('l1_ffn_gate', 'l1_ffn_down')
    dot.edge('l1_ffn_down', 'l1_ffn_dropout')
    dot.edge('l1_ffn_dropout', 'l1_add_norm2')
    dot.edge('l1_add_norm1', 'l1_add_norm2')
    
    # Pipeline communication
    dot.edge('l1_add_norm2', 'pipeline_send_0')
    dot.edge('pipeline_send_0', 'l2_q_proj')
    
    # Layer 2
    dot.edge('l2_q_proj', 'l2_attention')
    dot.edge('l2_k_proj', 'l2_attention')
    dot.edge('l2_v_proj', 'l2_attention')
    dot.edge('l2_attention', 'l2_dropout')
    dot.edge('l2_dropout', 'l2_add_norm1')
    dot.edge('pipeline_send_0', 'l2_add_norm1')
    dot.edge('l2_add_norm1', 'l2_ffn_up')
    dot.edge('l2_add_norm1', 'l2_ffn_gate')
    dot.edge('l2_ffn_up', 'l2_ffn_down')
    dot.edge('l2_ffn_gate', 'l2_ffn_down')
    dot.edge('l2_ffn_down', 'l2_ffn_dropout')
    dot.edge('l2_ffn_dropout', 'l2_add_norm2')
    dot.edge('l2_add_norm1', 'l2_add_norm2')
    
    # Layer 3
    dot.edge('l2_add_norm2', 'l3_q_proj')
    dot.edge('l2_add_norm2', 'l3_k_proj')
    dot.edge('l2_add_norm2', 'l3_v_proj')
    dot.edge('l3_q_proj', 'l3_attention')
    dot.edge('l3_k_proj', 'l3_attention')
    dot.edge('l3_v_proj', 'l3_attention')
    dot.edge('l3_attention', 'l3_dropout')
    dot.edge('l3_dropout', 'l3_add_norm1')
    dot.edge('l2_add_norm2', 'l3_add_norm1')
    dot.edge('l3_add_norm1', 'l3_ffn_up')
    dot.edge('l3_add_norm1', 'l3_ffn_gate')
    dot.edge('l3_ffn_up', 'l3_ffn_down')
    dot.edge('l3_ffn_gate', 'l3_ffn_down')
    dot.edge('l3_ffn_down', 'l3_ffn_dropout')
    dot.edge('l3_ffn_dropout', 'l3_add_norm2')
    dot.edge('l3_add_norm1', 'l3_add_norm2')
    
    # Output
    dot.edge('l3_add_norm2', 'output_norm')
    dot.edge('output_norm', 'output_proj')
    dot.edge('output_proj', 'output')
    
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