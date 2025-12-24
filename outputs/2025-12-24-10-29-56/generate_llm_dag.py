#!/usr/bin/env python3
"""
Generate DAG for LLM TP4xPP2 Hybrid Parallel Strategy
"""

import graphviz
from graphviz import Digraph
import os

def create_llm_tp4_pp2_dag():
    """Create a detailed DAG for TP4xPP2 hybrid configuration"""
    
    # Create the directed graph
    dot = Digraph(comment='LLM TP4xPP2 Hybrid Parallel Strategy DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='dashed', label='Input', color='black')
        c.node('input', 'Input Layer\nInput: [batch_size=32, seq_len=2048, hidden=8192]\nOutput: [batch_size=32, seq_len=2048, hidden=8192]', 
               shape='box', style='rounded,filled', fillcolor='lightgray')
    
    # Stage 0: Layers 0-39 on GPUs 0-3 (TP Group 0)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(style='filled', color='lightblue', fillcolor='lightblue', label='PP Stage 0: GPUs 0-3 (TP4)')
        
        # Embed + Pos Encoding (Layer 0)
        c.node('embed_0', 'Embedding + Pos Encoding (GPU0-3)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
               shape='rectangle')
        
        # Layer 1-39 (we'll show a few representative layers)
        for layer_id in range(1, 4):  # Show first few layers as examples
            # RMSNorm
            c.node(f'rmsnorm_{layer_id}_s0', f'Layer {layer_id}: RMSNorm (GPU0-3)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                   shape='rectangle')
            
            # Attention - detailed breakdown
            with dot.subgraph(name=f'cluster_attn_{layer_id}_s0') as attn_c:
                attn_c.attr(style='filled', color='lightgray', fillcolor='lightgray', label=f'Layer {layer_id}: Multi-Head Attention (TP4)')
                
                # QKV Projection
                attn_c.node(f'qkv_proj_{layer_id}_s0', f'QKV Projection (GPU0-3)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: Q,K,V [batch=32, seq=2048, head=64, d_k=128]', 
                           shape='rectangle')
                
                # Attention Score Computation
                attn_c.node(f'attn_scores_{layer_id}_s0', f'Attention Scores (GPU0-3)\nInput: Q,K,V [batch=32, seq=2048, head=64, d_k=128]\nOutput: Scores [batch=32, head=64, seq=2048, seq=2048]', 
                           shape='rectangle')
                
                # Attention Output
                attn_c.node(f'attn_out_{layer_id}_s0', f'Attention Output (GPU0-3)\nInput: Scores [batch=32, head=64, seq=2048, seq=2048]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                           shape='rectangle')
                
                # Attention All-Reduce
                attn_c.node(f'attn_ar_{layer_id}_s0', f'Attention All-Reduce (GPU0-3)\nTP4 Collective Operation', 
                           shape='ellipse', fillcolor='lightgreen')
            
            # FFN - detailed breakdown
            with dot.subgraph(name=f'cluster_ffn_{layer_id}_s0') as ffn_c:
                ffn_c.attr(style='filled', color='lightcoral', fillcolor='lightcoral', label=f'Layer {layer_id}: FFN (TP4)')
                
                # FFN Gate + Up Projection
                ffn_c.node(f'ffn_gate_up_{layer_id}_s0', f'FFN Gate+Up Projection (GPU0-3)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, ffn_hidden=28672]', 
                          shape='rectangle')
                
                # SiLU Activation
                ffn_c.node(f'ffn_silu_{layer_id}_s0', f'SiLU Activation (GPU0-3)\nInput: [batch=32, seq=2048, ffn_hidden=28672]\nOutput: [batch=32, seq=2048, ffn_hidden=28672]', 
                          shape='rectangle')
                
                # FFN Down Projection
                ffn_c.node(f'ffn_down_{layer_id}_s0', f'FFN Down Projection (GPU0-3)\nInput: [batch=32, seq=2048, ffn_hidden=28672]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                          shape='rectangle')
                
                # FFN All-Reduce
                ffn_c.node(f'ffn_ar_{layer_id}_s0', f'FFN All-Reduce (GPU0-3)\nTP4 Collective Operation', 
                          shape='ellipse', fillcolor='lightgreen')
        
        # Final layer of stage 0
        c.node('stage0_final', 'Stage 0 Final Layer (GPU0-3)\nLayer 39: RMSNorm + Attention + FFN\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
               shape='rectangle', fillcolor='lightblue')
    
    # Communication between Stage 0 and Stage 1
    dot.node('pp_comm_s0_s1', 'Pipeline Communication\nStage 0 → Stage 1\nGPU0-3 → GPU4-7\nMessage: [batch=32, seq=2048, hidden=8192]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 1: Layers 40-79 on GPUs 4-7 (TP Group 1)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(style='filled', color='lightpink', fillcolor='lightpink', label='PP Stage 1: GPUs 4-7 (TP4)')
        
        # First layer of stage 1
        c.node('stage1_first', 'Stage 1 First Layer (GPU4-7)\nLayer 40: RMSNorm + Attention + FFN\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
               shape='rectangle', fillcolor='lightpink')
        
        # Representative layers from stage 1
        for layer_id in range(41, 44):  # Show a few layers as examples
            # RMSNorm
            c.node(f'rmsnorm_{layer_id}_s1', f'Layer {layer_id}: RMSNorm (GPU4-7)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                   shape='rectangle')
            
            # Attention - detailed breakdown
            with dot.subgraph(name=f'cluster_attn_{layer_id}_s1') as attn_c:
                attn_c.attr(style='filled', color='lightgray', fillcolor='lightgray', label=f'Layer {layer_id}: Multi-Head Attention (TP4)')
                
                # QKV Projection
                attn_c.node(f'qkv_proj_{layer_id}_s1', f'QKV Projection (GPU4-7)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: Q,K,V [batch=32, seq=2048, head=64, d_k=128]', 
                           shape='rectangle')
                
                # Attention Score Computation
                attn_c.node(f'attn_scores_{layer_id}_s1', f'Attention Scores (GPU4-7)\nInput: Q,K,V [batch=32, seq=2048, head=64, d_k=128]\nOutput: Scores [batch=32, head=64, seq=2048, seq=2048]', 
                           shape='rectangle')
                
                # Attention Output
                attn_c.node(f'attn_out_{layer_id}_s1', f'Attention Output (GPU4-7)\nInput: Scores [batch=32, head=64, seq=2048, seq=2048]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                           shape='rectangle')
                
                # Attention All-Reduce
                attn_c.node(f'attn_ar_{layer_id}_s1', f'Attention All-Reduce (GPU4-7)\nTP4 Collective Operation', 
                           shape='ellipse', fillcolor='lightgreen')
            
            # FFN - detailed breakdown
            with dot.subgraph(name=f'cluster_ffn_{layer_id}_s1') as ffn_c:
                ffn_c.attr(style='filled', color='lightcoral', fillcolor='lightcoral', label=f'Layer {layer_id}: FFN (TP4)')
                
                # FFN Gate + Up Projection
                ffn_c.node(f'ffn_gate_up_{layer_id}_s1', f'FFN Gate+Up Projection (GPU4-7)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, ffn_hidden=28672]', 
                          shape='rectangle')
                
                # SiLU Activation
                ffn_c.node(f'ffn_silu_{layer_id}_s1', f'SiLU Activation (GPU4-7)\nInput: [batch=32, seq=2048, ffn_hidden=28672]\nOutput: [batch=32, seq=2048, ffn_hidden=28672]', 
                          shape='rectangle')
                
                # FFN Down Projection
                ffn_c.node(f'ffn_down_{layer_id}_s1', f'FFN Down Projection (GPU4-7)\nInput: [batch=32, seq=2048, ffn_hidden=28672]\nOutput: [batch=32, seq=2048, hidden=8192]', 
                          shape='rectangle')
                
                # FFN All-Reduce
                ffn_c.node(f'ffn_ar_{layer_id}_s1', f'FFN All-Reduce (GPU4-7)\nTP4 Collective Operation', 
                          shape='ellipse', fillcolor='lightgreen')
        
        # Final layer processing
        c.node('stage1_final', 'Stage 1 Final Layer (GPU4-7)\nLayer 79: RMSNorm\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, hidden=8192]', 
               shape='rectangle', fillcolor='lightpink')
    
    # Output processing
    with dot.subgraph(name='cluster_output') as c:
        c.attr(style='dashed', label='Output', color='black')
        c.node('lm_head', 'LM Head (GPU4-7)\nInput: [batch=32, seq=2048, hidden=8192]\nOutput: [batch=32, seq=2048, vocab=128256]', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('output', 'Output Layer\nInput: [batch=32, seq=2048, vocab=128256]\nOutput: [batch=32, seq=2048, vocab=128256]', 
               shape='box', style='rounded,filled', fillcolor='lightgray')
    
    # Create edges (dependencies)
    # Input to Stage 0
    dot.edge('input', 'embed_0')
    
    # Stage 0 internal connections (representative)
    layer_connections = [
        ('embed_0', 'rmsnorm_1_s0'),
        ('rmsnorm_1_s0', 'qkv_proj_1_s0'),
        ('qkv_proj_1_s0', 'attn_scores_1_s0'),
        ('attn_scores_1_s0', 'attn_out_1_s0'),
        ('attn_out_1_s0', 'attn_ar_1_s0'),
        ('attn_ar_1_s0', 'ffn_gate_up_1_s0'),
        ('ffn_gate_up_1_s0', 'ffn_silu_1_s0'),
        ('ffn_silu_1_s0', 'ffn_down_1_s0'),
        ('ffn_down_1_s0', 'ffn_ar_1_s0'),
    ]
    
    for src, dst in layer_connections:
        dot.edge(src, dst)
    
    # Connect to final stage 0 layer
    dot.edge('ffn_ar_3_s0', 'stage0_final')
    
    # Stage 0 to Stage 1 communication
    dot.edge('stage0_final', 'pp_comm_s0_s1')
    dot.edge('pp_comm_s0_s1', 'stage1_first')
    
    # Stage 1 internal connections (representative)
    stage1_connections = [
        ('stage1_first', 'rmsnorm_41_s1'),
        ('rmsnorm_41_s1', 'qkv_proj_41_s1'),
        ('qkv_proj_41_s1', 'attn_scores_41_s1'),
        ('attn_scores_41_s1', 'attn_out_41_s1'),
        ('attn_out_41_s1', 'attn_ar_41_s1'),
        ('attn_ar_41_s1', 'ffn_gate_up_41_s1'),
        ('ffn_gate_up_41_s1', 'ffn_silu_41_s1'),
        ('ffn_silu_41_s1', 'ffn_down_41_s1'),
        ('ffn_down_41_s1', 'ffn_ar_41_s1'),
    ]
    
    for src, dst in stage1_connections:
        dot.edge(src, dst)
    
    # Connect to final stage 1 layer
    dot.edge('ffn_ar_43_s1', 'stage1_final')
    
    # Stage 1 to output
    dot.edge('stage1_final', 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

def create_simplified_llm_dag():
    """Create a simplified but complete DAG showing the key components"""
    
    dot = Digraph(comment='LLM TP4xPP2 Simplified DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Communication
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing
    
    # Input
    dot.node('input', 'Input\n[batch=32, seq=2048, hidden=8192]', 
             shape='box', style='rounded,filled', fillcolor='lightgray')
    
    # Stage 0: Layers 0-39 (GPUs 0-3)
    dot.node('stage0_embed', 'Embedding + Pos Enc (GPU0-3)\nTP4 All-Reduce', 
             shape='rectangle')
    
    # Representative layers for Stage 0
    for i in range(1, 5):  # Show 4 representative layers
        layer_num = i * 10  # Every 10th layer
        dot.node(f'stage0_layer{layer_num}', f'Layer {layer_num} (GPU0-3)\nRMSNorm + MHA + FFN\nTP4 All-Reduce', 
                 shape='rectangle')
        
        if i > 1:
            prev_layer = (i-1) * 10
            dot.edge(f'stage0_layer{prev_layer}', f'stage0_layer{layer_num}')
    
    # Final layer of Stage 0
    dot.node('stage0_final', 'Layer 39 (GPU0-3)\nFinal RMSNorm', 
             shape='rectangle')
    
    # Pipeline communication
    dot.node('pp_comm', 'Pipeline Communication\nStage 0 → Stage 1\nGPU0-3 → GPU4-7', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 1: Layers 40-79 (GPUs 4-7)
    dot.node('stage1_first', 'Layer 40 (GPU4-7)\nFirst Layer Processing\nTP4 All-Reduce', 
             shape='rectangle', fillcolor='lightpink')
    
    # Representative layers for Stage 1
    for i in range(1, 5):  # Show 4 representative layers
        layer_num = 40 + i * 10  # Every 10th layer from layer 40
        dot.node(f'stage1_layer{layer_num}', f'Layer {layer_num} (GPU4-7)\nRMSNorm + MHA + FFN\nTP4 All-Reduce', 
                 shape='rectangle', fillcolor='lightpink')
        
        if i > 1:
            prev_layer = 40 + (i-1) * 10
            dot.edge(f'stage1_layer{prev_layer}', f'stage1_layer{layer_num}')
    
    # Final layer of Stage 1
    dot.node('stage1_final', 'Layer 79 (GPU4-7)\nFinal RMSNorm', 
             shape='rectangle', fillcolor='lightpink')
    
    # Output
    dot.node('lm_head', 'LM Head (GPU4-7)\n[hidden=8192 → vocab=128256]', 
             shape='parallelogram', fillcolor='lightyellow')
    dot.node('output', 'Output\n[batch=32, seq=2048, vocab=128256]', 
             shape='box', style='rounded,filled', fillcolor='lightgray')
    
    # Create edges
    dot.edge('input', 'stage0_embed')
    dot.edge('stage0_embed', 'stage0_layer10')
    dot.edge('stage0_layer30', 'stage0_final')
    dot.edge('stage0_final', 'pp_comm')
    dot.edge('pp_comm', 'stage1_first')
    dot.edge('stage1_first', 'stage1_layer50')
    dot.edge('stage1_layer70', 'stage1_final')
    dot.edge('stage1_final', 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

def main():
    """Generate both detailed and simplified DAGs"""
    
    output_dir = '../outputs/2025-12-24-10-29-56'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating LLM TP4xPP2 Hybrid DAG...")
    
    # Generate detailed DAG
    detailed_dag = create_llm_tp4_pp2_dag()
    
    # Save as DOT file
    detailed_dot_path = os.path.join(output_dir, 'llm_tp4_pp2_detailed.dot')
    with open(detailed_dot_path, 'w') as f:
        f.write(detailed_dag.source)
    print(f"Detailed DAG DOT saved to: {detailed_dot_path}")
    
    # Generate SVG (this might take a while for detailed version)
    try:
        detailed_svg_path = os.path.join(output_dir, 'llm_tp4_pp2_detailed.svg')
        detailed_dag.render(detailed_svg_path.replace('.svg', ''), format='svg', cleanup=True)
        print(f"Detailed DAG SVG saved to: {detailed_svg_path}")
    except Exception as e:
        print(f"Warning: Could not generate detailed SVG: {e}")
    
    # Generate simplified DAG
    simplified_dag = create_simplified_llm_dag()
    
    # Save as DOT file
    simplified_dot_path = os.path.join(output_dir, 'llm_tp4_pp2_simplified.dot')
    with open(simplified_dot_path, 'w') as f:
        f.write(simplified_dag.source)
    print(f"Simplified DAG DOT saved to: {simplified_dot_path}")
    
    # Generate SVG for simplified version
    try:
        simplified_svg_path = os.path.join(output_dir, 'llm_tp4_pp2_simplified.svg')
        simplified_dag.render(simplified_svg_path.replace('.svg', ''), format='svg', cleanup=True)
        print(f"Simplified DAG SVG saved to: {simplified_svg_path}")
    except Exception as e:
        print(f"Warning: Could not generate simplified SVG: {e}")
    
    print("DAG generation complete!")
    
    # Return paths for verification
    return {
        'detailed_dot': detailed_dot_path,
        'detailed_svg': detailed_svg_path if os.path.exists(detailed_svg_path) else None,
        'simplified_dot': simplified_dot_path,
        'simplified_svg': simplified_svg_path if os.path.exists(simplified_svg_path) else None
    }

if __name__ == '__main__':
    paths = main()
    print("\nGenerated files:")
    for key, path in paths.items():
        if path:
            print(f"  {key}: {path}")