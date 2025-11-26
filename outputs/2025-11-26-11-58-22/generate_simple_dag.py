#!/usr/bin/env python3
"""
Simplified Large-Scale Cross-Node Expert Parallelism DAG Generator
"""

import graphviz
import json
import os

def generate_complete_dag():
    """Generate complete DAG for the entire model"""
    dot = graphviz.Digraph(comment='Large-Scale Cross-Node Expert Parallelism MoE')
    dot.attr(rankdir='TB', size='200,300')
    
    # Model specs
    batch_size = 32
    seq_len = 2048
    hidden_size = 7168
    heads = 128
    d_k = 128
    ffn_hidden = 2048
    
    # Input
    dot.node('input', f'INPUT\\nbatch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: All', 
             shape='ellipse', fillcolor='lightgreen', style='filled')
    
    # Dense layers (0-2)
    for layer in range(3):
        gpu_id = 3712 + layer
        
        # LayerNorm 1
        dot.node(f'd{layer}_ln1', f'LayerNorm\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightyellow', style='filled')
        
        # QKV Projection
        dot.node(f'd{layer}_qkv', f'QKV Projection\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}\\nGPU: {gpu_id}', 
                 fillcolor='lightblue', style='filled')
        
        # MHA Attention
        dot.node(f'd{layer}_mha', f'MHA Attention\\nInput: batch_size={batch_size}, seq_len={seq_len}, heads={heads}, d_k={d_k}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightblue', style='filled')
        
        # Output Projection
        dot.node(f'd{layer}_out', f'Output Projection\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightblue', style='filled')
        
        # Residual 1
        dot.node(f'd{layer}_res1', f'Residual Add\\nInput1: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nInput2: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightpink', style='filled')
        
        # LayerNorm 2
        dot.node(f'd{layer}_ln2', f'LayerNorm\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightyellow', style='filled')
        
        # FFN Linear 1
        dot.node(f'd{layer}_ffn1', f'FFN Linear1\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}\\nGPU: {gpu_id}', 
                 fillcolor='lightblue', style='filled')
        
        # GELU
        dot.node(f'd{layer}_gelu', f'GELU\\nInput: batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}\\nGPU: {gpu_id}', 
                 fillcolor='lightgreen', style='filled')
        
        # FFN Linear 2
        dot.node(f'd{layer}_ffn2', f'FFN Linear2\\nInput: batch_size={batch_size}, seq_len={seq_len}, ffn_hidden={ffn_hidden}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightblue', style='filled')
        
        # Final Residual
        dot.node(f'd{layer}_final', f'Residual Add\\nInput1: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nInput2: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nOutput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}', 
                 fillcolor='lightpink', style='filled')
    
    # Connect dense layers
    dot.edge('input', 'd0_ln1')
    dot.edge('d0_ln1', 'd0_qkv')
    dot.edge('d0_qkv', 'd0_mha')
    dot.edge('d0_mha', 'd0_out')
    dot.edge('d0_out', 'd0_res1')
    dot.edge('input', 'd0_res1')
    dot.edge('d0_res1', 'd0_ln2')
    dot.edge('d0_ln2', 'd0_ffn1')
    dot.edge('d0_ffn1', 'd0_gelu')
    dot.edge('d0_gelu', 'd0_ffn2')
    dot.edge('d0_ffn2', 'd0_final')
    dot.edge('d0_res1', 'd0_final')
    
    dot.edge('d0_final', 'd1_ln1')
    # ... (similar connections for d1)
    
    # MoE layers (3-60)
    for layer in range(3, 61):
        layer_gpu = 3712 + layer
        expert_start = (layer - 3) * 64
        
        # MHA part
        dot.node(f'm{layer}_ln1', f'LayerNorm\\nGPU: {layer_gpu}', 
                 fillcolor='lightyellow', style='filled')
        dot.node(f'm{layer}_qkv', f'QKV Projection\\nGPU: {layer_gpu}', 
                 fillcolor='lightblue', style='filled')
        dot.node(f'm{layer}_mha', f'MHA Attention\\nGPU: {layer_gpu}', 
                 fillcolor='lightblue', style='filled')
        dot.node(f'm{layer}_out', f'Output Projection\\nGPU: {layer_gpu}', 
                 fillcolor='lightblue', style='filled')
        dot.node(f'm{layer}_res1', f'Residual Add\\nGPU: {layer_gpu}', 
                 fillcolor='lightpink', style='filled')
        
        # Expert routing
        dot.node(f'm{layer}_gate', f'Expert Gate\\nTop-2 selection\\nGPU: {layer_gpu}', 
                 shape='parallelogram', fillcolor='orange', style='filled')
        dot.node(f'm{layer}_route', f'Expert Router\\nDistribute tokens\\nGPU: All', 
                 shape='ellipse', fillcolor='lightgray', style='filled')
        
        # Expert computation (show first few and last few)
        expert_nodes = []
        for expert in range(3):  # Show first 3 experts
            gpu = expert_start + expert
            node = gpu // 8
            expert_node = f'm{layer}_expert{expert}'
            dot.node(expert_node, 
                     f'Expert {expert}\\nNode {node} GPU {gpu % 8}\\nInput: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}', 
                     fillcolor='lightblue', style='filled')
            expert_nodes.append(expert_node)
        
        # Aggregation
        dot.node(f'm{layer}_agg', f'Expert Aggregation\\nGPU: {layer_gpu}', 
                 shape='parallelogram', fillcolor='purple', style='filled')
        
        # Final residual
        dot.node(f'm{layer}_final', f'Final Residual Add\\nGPU: {layer_gpu}', 
                 fillcolor='lightpink', style='filled')
        
        # Connect MoE layer
        if layer == 3:
            dot.edge('d2_final', f'm{layer}_ln1')
        dot.edge(f'm{layer}_ln1', f'm{layer}_qkv')
        dot.edge(f'm{layer}_qkv', f'm{layer}_mha')
        dot.edge(f'm{layer}_mha', f'm{layer}_out')
        dot.edge(f'm{layer}_out', f'm{layer}_res1')
        dot.edge('d2_final' if layer == 3 else f'm{layer-1}_final', f'm{layer}_res1')
        dot.edge(f'm{layer}_res1', f'm{layer}_gate')
        dot.edge(f'm{layer}_gate', f'm{layer}_route', style='dashed')
        dot.edge(f'm{layer}_res1', f'm{layer}_route')
        
        for expert in expert_nodes:
            dot.edge(f'm{layer}_route', expert)
            dot.edge(expert, f'm{layer}_agg')
        
        dot.edge(f'm{layer}_agg', f'm{layer}_final')
        dot.edge(f'm{layer}_res1', f'm{layer}_final')
    
    # Output
    dot.node('output', f'OUTPUT\\nbatch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: All', 
             shape='ellipse', fillcolor='lightcoral', style='filled')
    dot.edge('m60_final', 'output')
    
    return dot.source

def generate_layer_dag(layer_idx):
    """Generate DAG for specific layer"""
    dot = graphviz.Digraph(comment=f'Layer {layer_idx}')
    dot.attr(rankdir='TB', size='50,100')
    
    batch_size = 32
    seq_len = 2048
    hidden_size = 7168
    
    if layer_idx < 3:
        # Dense layer
        gpu_id = 3712 + layer_idx
        dot.node('input', f'Input\\nbatch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}')
        dot.node('ln1', f'LayerNorm\\nGPU: {gpu_id}')
        dot.node('qkv', f'QKV Projection\\nGPU: {gpu_id}')
        dot.node('attn', f'MHA Attention\\nGPU: {gpu_id}')
        dot.node('out', f'Output Projection\\nGPU: {gpu_id}')
        dot.node('res1', f'Residual Add\\nGPU: {gpu_id}')
        dot.node('ln2', f'LayerNorm\\nGPU: {gpu_id}')
        dot.node('ffn1', f'FFN Linear1\\nGPU: {gpu_id}')
        dot.node('gelu', f'GELU\\nGPU: {gpu_id}')
        dot.node('ffn2', f'FFN Linear2\\nGPU: {gpu_id}')
        dot.node('final', f'Residual Add\\nGPU: {gpu_id}')
        
        dot.edge('input', 'ln1')
        dot.edge('ln1', 'qkv')
        dot.edge('qkv', 'attn')
        dot.edge('attn', 'out')
        dot.edge('out', 'res1')
        dot.edge('input', 'res1')
        dot.edge('res1', 'ln2')
        dot.edge('ln2', 'ffn1')
        dot.edge('ffn1', 'gelu')
        dot.edge('gelu', 'ffn2')
        dot.edge('ffn2', 'final')
        dot.edge('res1', 'final')
    else:
        # MoE layer
        gpu_id = 3712
        expert_start = (layer_idx - 3) * 64
        
        dot.node('input', f'Input\\nbatch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}\\nGPU: {gpu_id}')
        dot.node('ln1', f'LayerNorm\\nGPU: {gpu_id}')
        dot.node('qkv', f'QKV Projection\\nGPU: {gpu_id}')
        dot.node('attn', f'MHA Attention\\nGPU: {gpu_id}')
        dot.node('out', f'Output Projection\\nGPU: {gpu_id}')
        dot.node('res1', f'Residual Add\\nGPU: {gpu_id}')
        dot.node('gate', f'Expert Gate\\nGPU: {gpu_id}', shape='parallelogram')
        dot.node('route', f'Expert Router\\nGPU: All', shape='ellipse')
        
        # Add experts
        experts = []
        for i in range(64):
            gpu = expert_start + i
            expert = f'expert{i}'
            dot.node(expert, f'Expert {i}\\nGPU: {gpu}')
            experts.append(expert)
        
        dot.node('agg', f'Expert Aggregation\\nGPU: {gpu_id}', shape='parallelogram')
        dot.node('final', f'Final Residual Add\\nGPU: {gpu_id}')
        
        dot.edge('input', 'ln1')
        dot.edge('ln1', 'qkv')
        dot.edge('qkv', 'attn')
        dot.edge('attn', 'out')
        dot.edge('out', 'res1')
        dot.edge('input', 'res1')
        dot.edge('res1', 'gate')
        dot.edge('gate', 'route', style='dashed')
        dot.edge('res1', 'route')
        
        for expert in experts:
            dot.edge('route', expert)
            dot.edge(expert, 'agg')
        
        dot.edge('agg', 'final')
        dot.edge('res1', 'final')
    
    return dot.source

def generate_expert_parallelism_overview():
    """Generate expert parallelism overview"""
    dot = graphviz.Digraph(comment='Expert Parallelism')
    dot.attr(rankdir='LR', size='150,50')
    
    # Show expert distribution across nodes
    for layer in [3, 4, 5]:
        expert_start = (layer - 3) * 64
        
        dot.node(f'layer{layer}_in', f'Layer {layer}\\nInput\\nGPU: 3712')
        
        # Show 8 nodes with 8 GPUs each
        for node in range(8):
            start_gpu = expert_start + node * 8
            node_experts = []
            
            for gpu in range(8):
                expert = f'layer{layer}_node{node}_gpu{gpu}'
                dot.node(expert, f'Expert {start_gpu + gpu}\\nNode {node}\\nGPU: {gpu}')
                dot.edge(f'layer{layer}_in', expert)
                node_experts.append(expert)
        
        dot.node(f'layer{layer}_out', f'Layer {layer}\\nOutput\\nGPU: 3713')
        
        # Connect all experts to output (simplified)
        for expert in [f'layer{layer}_node{node}_gpu{gpu}' for node in range(8) for gpu in range(8)]:
            dot.edge(expert, f'layer{layer}_out')
    
    return dot.source

def main():
    output_dir = "../outputs/2025-11-26-11-58-22"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all DAGs
    dags = {
        "complete_model_dag": generate_complete_dag(),
        "layer_0_dag": generate_layer_dag(0),  # Dense layer
        "layer_3_dag": generate_layer_dag(3),  # First MoE layer
        "layer_10_dag": generate_layer_dag(10),  # Middle MoE layer
        "expert_parallelism_dag": generate_expert_parallelism_overview()
    }
    
    # Save DOT files
    for name, content in dags.items():
        with open(f"{output_dir}/{name}.dot", "w") as f:
            f.write(content)
    
    # Generate SVGs if graphviz available
    try:
        import subprocess
        for name in dags.keys():
            subprocess.run(["dot", "-Tsvg", f"{output_dir}/{name}.dot", "-o", f"{output_dir}/{name}.svg"], check=True)
    except:
        pass
    
    # Create paths file
    paths = {f"{name}": f"{output_dir}/{name}.dot" for name in dags.keys()}
    paths.update({f"{name}_svg": f"{output_dir}/{name}.svg" for name in dags.keys()})
    
    with open(f"{output_dir}/generated_dags.json", "w") as f:
        json.dump(paths, f, indent=2)
    
    return paths

if __name__ == "__main__":
    paths = main()
    print(json.dumps(paths, indent=2))