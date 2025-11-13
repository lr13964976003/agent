#!/usr/bin/env python3
"""
Generate baseline DAGs for comparison
"""

import graphviz
from typing import Dict, List

def create_baseline_sequential_dag(model_name: str, num_layers: int, hidden_size: int, 
                                 ffn_hidden_size: int, num_heads: int, vocab_size: int):
    """Create baseline sequential DAG"""
    dot = graphviz.Digraph(comment=f'{model_name} Baseline Sequential', format='svg')
    dot.attr(rankdir='TB', splines='ortho')
    
    batch_size = 1
    seq_len = 2048
    head_dim = hidden_size // num_heads
    
    # Define node shapes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    
    # Global input
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}]\\nGPU: all GPUs (sequential)', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Embedding layer
    dot.node('embedding', f'Embedding\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs', 
             shape='rectangle', fillcolor='lightblue')
    dot.edge('input', 'embedding')
    
    prev_node = 'embedding'
    
    # Create all layers sequentially
    for layer_idx in range(1, num_layers + 1):
        layer_name = f'layer_{layer_idx}'
        
        # Create nodes for multi-head attention
        attn_qkv = f'{layer_name}_attn_qkv'
        attn_score = f'{layer_name}_attn_score'
        attn_softmax = f'{layer_name}_attn_softmax'
        attn_dropout = f'{layer_name}_attn_dropout'
        attn_out = f'{layer_name}_attn_out'
        attn_residual = f'{layer_name}_attn_residual'
        attn_norm = f'{layer_name}_attn_norm'
        
        # Create nodes for FFN
        ffn_up = f'{layer_name}_ffn_up'
        ffn_act = f'{layer_name}_ffn_act'
        ffn_down = f'{layer_name}_ffn_down'
        ffn_residual = f'{layer_name}_ffn_residual'
        ffn_norm = f'{layer_name}_ffn_norm'
        
        # Attention computation
        dot.node(attn_qkv, f'Layer {layer_idx}\\nAttention QKV\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: all GPUs')
        dot.node(attn_score, f'Layer {layer_idx}\\nAttention Score\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: all GPUs')
        dot.node(attn_softmax, f'Layer {layer_idx}\\nAttention Softmax\\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: all GPUs')
        dot.node(attn_dropout, f'Layer {layer_idx}\\nAttention Dropout\\nInput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len}]\\nGPU: all GPUs')
        dot.node(attn_out, f'Layer {layer_idx}\\nAttention Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        dot.node(attn_residual, f'Layer {layer_idx}\\nAttention Residual\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        dot.node(attn_norm, f'Layer {layer_idx}\\nAttention Norm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        
        # FFN computation
        dot.node(ffn_up, f'Layer {layer_idx}\\nFFN Up\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nGPU: all GPUs')
        dot.node(ffn_act, f'Layer {layer_idx}\\nFFN Activation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nGPU: all GPUs')
        dot.node(ffn_down, f'Layer {layer_idx}\\nFFN Down\\nInput: [batch_size={batch_size}, seq_len={seq_len}, ffn_hidden_size={ffn_hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        dot.node(ffn_residual, f'Layer {layer_idx}\\nFFN Residual\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        dot.node(ffn_norm, f'Layer {layer_idx}\\nFFN Norm\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: all GPUs')
        
        # Connect operations
        dot.edge(prev_node, attn_qkv)
        dot.edge(attn_qkv, attn_score)
        dot.edge(attn_score, attn_softmax)
        dot.edge(attn_softmax, attn_dropout)
        dot.edge(attn_dropout, attn_out)
        dot.edge(attn_out, attn_residual)
        dot.edge(attn_residual, attn_norm)
        
        dot.edge(attn_norm, ffn_up)
        dot.edge(ffn_up, ffn_act)
        dot.edge(ffn_act, ffn_down)
        dot.edge(ffn_down, ffn_residual)
        dot.edge(ffn_residual, ffn_norm)
        
        prev_node = ffn_norm
    
    # Final output layer
    dot.node('lm_head', f'LM Head\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: all GPUs', 
             shape='rectangle', fillcolor='lightblue')
    dot.node('output', f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    dot.edge(prev_node, 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

def create_gpipe_dag(model_name: str, num_layers: int, hidden_size: int, 
                    ffn_hidden_size: int, num_heads: int, vocab_size: int):
    """Create GPipe DAG with batch dimension pipeline"""
    dot = graphviz.Digraph(comment=f'{model_name} GPipe Batch Pipeline', format='svg')
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    
    batch_size = 12  # For GPT3, 6 for LLaMA
    micro_batch_size = 1
    seq_len = 2048
    head_dim = hidden_size // num_heads
    
    # Adjust batch size based on model
    if model_name == "LLaMA-7B":
        batch_size = 6
    
    # Define node shapes
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightgreen')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Communication
    
    # Global input
    dot.node('input', f'Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}]\\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Split input into micro-batches
    dot.node('split_batches', f'Split Batches\\nInput: [batch_size={batch_size}, seq_len={seq_len}]\\nOutput: [micro_batch_size={micro_batch_size}, seq_len={seq_len}]\\nGPU: all GPUs', 
             shape='parallelogram', fillcolor='lightyellow')
    dot.edge('input', 'split_batches')
    
    # Calculate layers per stage for 6 stages
    layers_per_stage = num_layers // 6
    devices = ['P@1', 'P@2', 'P@3', 'P@4', 'R@1', 'R@2']
    
    prev_node = 'split_batches'
    
    for stage_idx, device in enumerate(devices):
        start_layer = stage_idx * layers_per_stage + 1
        end_layer = (stage_idx + 1) * layers_per_stage if stage_idx < 5 else num_layers
        
        with dot.subgraph(name=f'cluster_stage_{stage_idx}') as stage:
            stage.attr(label=f'Stage {stage_idx+1}\\n{device}\\nLayers {start_layer}-{end_layer}', style='dashed')
            
            # Process each micro-batch
            for micro_batch in range(batch_size):
                mb_prefix = f'mb_{micro_batch}_'
                
                if stage_idx == 0:
                    # First stage includes embedding
                    embedding = f'{mb_prefix}embedding'
                    stage.node(embedding, f'Micro-Batch {micro_batch}\\nEmbedding\\nInput: [batch_size=1, seq_len={seq_len}]\\nOutput: [batch_size=1, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: {device}')
                    dot.edge(prev_node, embedding)
                    prev_layer_node = embedding
                else:
                    # Communication from previous stage
                    comm_node = f'{mb_prefix}comm_stage_{stage_idx-1}'
                    stage.node(comm_node, f'Micro-Batch {micro_batch}\\nBatch Transfer\\nInput: [batch_size=1, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size=1, seq_len={seq_len}, hidden_size={hidden_size}]\\nFrom: {devices[stage_idx-1]} To: {device}', 
                             shape='parallelogram', fillcolor='lightyellow')
                    dot.edge(f'{mb_prefix}stage_{stage_idx-1}_output', comm_node)
                    prev_layer_node = comm_node
                
                # Process each layer in this stage
                for layer_idx in range(start_layer, end_layer + 1):
                    layer_base = f'{mb_prefix}layer_{layer_idx}'
                    
                    # Attention computation
                    attn_norm = f'{layer_base}_attn_norm'
                    attn_qkv = f'{layer_base}_attn_qkv'
                    attn_out = f'{layer_base}_attn_out'
                    attn_residual = f'{layer_base}_attn_residual'
                    
                    # FFN computation
                    ffn_norm = f'{layer_base}_ffn_norm'
                    ffn_up = f'{layer_base}_ffn_up'
                    ffn_down = f'{layer_base}_ffn_down'
                    ffn_residual = f'{layer_base}_ffn_residual'
                    
                    stage.node(attn_norm, f'MB{micro_batch} L{layer_idx}\\nAttention Norm\\nGPU: {device}')
                    stage.node(attn_qkv, f'MB{micro_batch} L{layer_idx}\\nAttention QKV\\nGPU: {device}')
                    stage.node(attn_out, f'MB{micro_batch} L{layer_idx}\\nAttention Out\\nGPU: {device}')
                    stage.node(attn_residual, f'MB{micro_batch} L{layer_idx}\\nAttn Residual\\nGPU: {device}')
                    
                    stage.node(ffn_norm, f'MB{micro_batch} L{layer_idx}\\nFFN Norm\\nGPU: {device}')
                    stage.node(ffn_up, f'MB{micro_batch} L{layer_idx}\\nFFN Up\\nGPU: {device}')
                    stage.node(ffn_down, f'MB{micro_batch} L{layer_idx}\\nFFN Down\\nGPU: {device}')
                    stage.node(ffn_residual, f'MB{micro_batch} L{layer_idx}\\nFFN Residual\\nGPU: {device}')
                    
                    if layer_idx == start_layer:
                        stage.edge(prev_layer_node, attn_norm)
                    else:
                        stage.edge(f'{mb_prefix}layer_{layer_idx-1}_ffn_residual', attn_norm)
                    
                    stage.edge(attn_norm, attn_qkv)
                    stage.edge(attn_qkv, attn_out)
                    stage.edge(attn_out, attn_residual)
                    stage.edge(attn_residual, ffn_norm)
                    stage.edge(ffn_norm, ffn_up)
                    stage.edge(ffn_up, ffn_down)
                    stage.edge(ffn_down, ffn_residual)
                    
                    if layer_idx == end_layer:
                        # Output for this stage
                        stage_output = f'{mb_prefix}stage_{stage_idx}_output'
                        stage.node(stage_output, f'MB{micro_batch}\\nStage {stage_idx+1}\\nOutput\\nGPU: {device}')
                        stage.edge(ffn_residual, stage_output)
    
    # Final output layer
    final_comm = 'final_comm'
    dot.node(final_comm, f'Gather Batches\\nInput: [micro_batch_size={micro_batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nGPU: R@2', 
             shape='parallelogram', fillcolor='lightyellow')
    
    lm_head = 'lm_head'
    dot.node(lm_head, f'LM Head\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: R@2', 
             shape='rectangle', fillcolor='lightblue')
    
    output = 'output'
    dot.node(output, f'Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]\\nGPU: all GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect final stages
    for micro_batch in range(batch_size):
        dot.edge(f'mb_{micro_batch}_stage_5_output', final_comm)
    
    dot.edge(final_comm, lm_head)
    dot.edge(lm_head, output)
    
    return dot

if __name__ == '__main__':
    # Create baseline DAGs
    llama_baseline = create_baseline_sequential_dag("LLaMA-7B", 32, 4096, 11008, 32, 32000)
    llama_baseline.save('../outputs/2025-10-31-11-22-09/llama_7b_baseline')
    
    gpt3_baseline = create_baseline_sequential_dag("GPT3-2B", 24, 2560, 10240, 32, 50257)
    gpt3_baseline.save('../outputs/2025-10-31-11-22-09/gpt3_2b_baseline')
    
    llama_gpipe = create_gpipe_dag("LLaMA-7B", 32, 4096, 11008, 32, 32000)
    llama_gpipe.save('../outputs/2025-10-31-11-22-09/llama_7b_gpipe')
    
    gpt3_gpipe = create_gpipe_dag("GPT3-2B", 24, 2560, 10240, 32, 50257)
    gpt3_gpipe.save('../outputs/2025-10-31-11-22-09/gpt3_2b_gpipe')
    
    print("All baseline DAGs saved")