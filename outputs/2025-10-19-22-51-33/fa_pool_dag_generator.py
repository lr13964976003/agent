#!/usr/bin/env python3
"""
Generate FA Pool DAG for dynamic GPU allocation based on sequence length
Handles 32K+ token scenario with 32 pool GPUs + 8 base GPUs = 40 total GPUs
"""

import graphviz
import os
import math

def create_fa_pool_dag():
    """Create FA Pool DAG with dynamic allocation"""
    dot = graphviz.Digraph(comment='FA Pool Dynamic Allocation DAG')
    dot.attr(rankdir='TB', splines='ortho', ranksep='1.2', nodesep='0.6')
    
    # Configuration for 32K+ tokens
    sequence_length = 32768
    batch_size = 1024
    num_pool_gpus = 32
    num_base_gpus = 8
    total_gpus = 40
    
    # Calculate block size for pool
    block_size = math.ceil(sequence_length / num_pool_gpus)
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Global input
    dot.node('input', 
             f'Total Input\\nInput: [batch_size=1024, seq_len={sequence_length}]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightgreen')
    
    # === BASE LAYER (GPUs 0-7) ===
    dot.attr('node', fillcolor='lightblue')
    
    # Base layer components
    dot.node('embed', 
             f'Embedding Layer\\nInput: [batch_size=1024, seq_len={sequence_length}]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('pos_enc', 
             f'Positional Encoding\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    # === ATTENTION POOL (GPUs 8-39) ===
    
    # KV Cache replication across all pool GPUs
    dot.node('kv_cache_replicate', 
             f'KV Cache Replication\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nReplicated to: pool_gpu_8-39\\nGPU: all pool GPUs', 
             shape='parallelogram', fillcolor='yellow')
    
    # Attention computation blocks across pool GPUs
    attention_nodes = []
    for i in range(num_pool_gpus):
        start_pos = i * block_size
        end_pos = min((i + 1) * block_size, sequence_length)
        actual_block_size = end_pos - start_pos
        
        # Q projection for this block (distributed from base)
        dot.node(f'q_proj_{i}', 
                 f'Q Projection GPU {8+i}\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={actual_block_size}, heads=32, d_k=128]\\nGPU: pool_gpu_{8+i}', 
                 shape='rectangle', fillcolor='lightcoral')
        
        # K/V access from replicated cache
        dot.node(f'k_access_{i}', 
                 f'K Access GPU {8+i}\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, heads=32, d_k=128]\\nGPU: pool_gpu_{8+i}', 
                 shape='rectangle', fillcolor='lightcoral')
        
        dot.node(f'v_access_{i}', 
                 f'V Access GPU {8+i}\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, heads=32, d_k=128]\\nGPU: pool_gpu_{8+i}', 
                 shape='rectangle', fillcolor='lightcoral')
        
        # Flash attention computation
        dot.node(f'flash_attn_{i}', 
                 f'Flash Attention GPU {8+i}\\nQ: [batch_size=1024, seq_len={actual_block_size}, heads=32, d_k=128]\\nK/V: [batch_size=1024, seq_len={sequence_length}, heads=32, d_k=128]\\nOutput: [batch_size=1024, seq_len={actual_block_size}, hidden=4096]\\nGPU: pool_gpu_{8+i}', 
                 shape='rectangle', fillcolor='lightcoral')
        
        attention_nodes.append(f'flash_attn_{i}')
    
    # Attention results aggregation (tree-based reduction)
    dot.node('attention_agg', 
             f'Attention Results Aggregation\\nInput: [{num_pool_gpus} blocks × [batch_size=1024, seq_len={block_size}, hidden=4096]]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='parallelogram', fillcolor='yellow')
    
    # === MODEL LAYERS ===
    
    # Layer 0 - Attention path
    dot.node('layer0_attn_dropout', 
             f'Layer 0 Attention Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer0_add_norm1', 
             f'Layer 0 Add & Norm 1\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 0 - FFN path (base layer only)
    dot.node('layer0_ffn_up', 
             f'Layer 0 FFN Up\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer0_ffn_gate', 
             f'Layer 0 FFN Gate\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer0_ffn_down', 
             f'Layer 0 FFN Down\\nInput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer0_ffn_dropout', 
             f'Layer 0 FFN Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer0_add_norm2', 
             f'Layer 0 Add & Norm 2\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 1 - Attention path
    dot.node('layer1_attn_dropout', 
             f'Layer 1 Attention Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer1_add_norm1', 
             f'Layer 1 Add & Norm 1\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 1 - FFN path
    dot.node('layer1_ffn_up', 
             f'Layer 1 FFN Up\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer1_ffn_gate', 
             f'Layer 1 FFN Gate\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer1_ffn_down', 
             f'Layer 1 FFN Down\\nInput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer1_ffn_dropout', 
             f'Layer 1 FFN Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer1_add_norm2', 
             f'Layer 1 Add & Norm 2\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 2 - Attention path
    dot.node('layer2_attn_dropout', 
             f'Layer 2 Attention Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer2_add_norm1', 
             f'Layer 2 Add & Norm 1\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 2 - FFN path
    dot.node('layer2_ffn_up', 
             f'Layer 2 FFN Up\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer2_ffn_gate', 
             f'Layer 2 FFN Gate\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer2_ffn_down', 
             f'Layer 2 FFN Down\\nInput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer2_ffn_dropout', 
             f'Layer 2 FFN Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer2_add_norm2', 
             f'Layer 2 Add & Norm 2\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 3 - Attention path
    dot.node('layer3_attn_dropout', 
             f'Layer 3 Attention Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer3_add_norm1', 
             f'Layer 3 Add & Norm 1\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Layer 3 - FFN path
    dot.node('layer3_ffn_up', 
             f'Layer 3 FFN Up\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer3_ffn_gate', 
             f'Layer 3 FFN Gate\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer3_ffn_down', 
             f'Layer 3 FFN Down\\nInput: [batch_size=1024, seq_len={sequence_length}, ffn=16384]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7 (TP=2)', 
             shape='rectangle')
    
    dot.node('layer3_ffn_dropout', 
             f'Layer 3 FFN Dropout\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle')
    
    dot.node('layer3_add_norm2', 
             f'Layer 3 Add & Norm 2\\nInput1: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nInput2: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='hexagon')
    
    # Final output
    dot.node('final_norm', 
             f'Final Layer Norm\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nGPU: base_gpu_0-7', 
             shape='rectangle', fillcolor='lightgreen')
    
    dot.node('output_proj', 
             f'Output Projection\\nInput: [batch_size=1024, seq_len={sequence_length}, hidden=4096]\\nOutput: [batch_size=1024, seq_len={sequence_length}, vocab=32000]\\nGPU: base_gpu_0-7', 
             shape='rectangle', fillcolor='lightgreen')
    
    dot.node('output', 
             f'Total Output\\nInput: [batch_size=1024, seq_len={sequence_length}, vocab=32000]\\nGPU: base_gpu_0-7', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create edges
    dot.edge('input', 'embed')
    dot.edge('embed', 'pos_enc')
    dot.edge('pos_enc', 'kv_cache_replicate')
    
    # Connect to all attention blocks in pool
    for i in range(num_pool_gpus):
        dot.edge('pos_enc', f'q_proj_{i}')
        dot.edge('kv_cache_replicate', f'k_access_{i}')
        dot.edge('kv_cache_replicate', f'v_access_{i}')
        dot.edge(f'q_proj_{i}', f'flash_attn_{i}')
        dot.edge(f'k_access_{i}', f'flash_attn_{i}')
        dot.edge(f'v_access_{i}', f'flash_attn_{i}')
    
    # Collect attention results
    for i in range(num_pool_gpus):
        dot.edge(f'flash_attn_{i}', 'attention_agg')
    
    # Layer 0 attention path
    dot.edge('attention_agg', 'layer0_attn_dropout')
    dot.edge('layer0_attn_dropout', 'layer0_add_norm1')
    dot.edge('pos_enc', 'layer0_add_norm1')
    
    # Layer 0 FFN path
    dot.edge('layer0_add_norm1', 'layer0_ffn_up')
    dot.edge('layer0_add_norm1', 'layer0_ffn_gate')
    dot.edge('layer0_ffn_up', 'layer0_ffn_down')
    dot.edge('layer0_ffn_gate', 'layer0_ffn_down')
    dot.edge('layer0_ffn_down', 'layer0_ffn_dropout')
    dot.edge('layer0_ffn_dropout', 'layer0_add_norm2')
    dot.edge('layer0_add_norm1', 'layer0_add_norm2')
    
    # Layer 1 attention path
    dot.edge('layer0_add_norm2', 'layer1_attn_dropout')
    dot.edge('layer1_attn_dropout', 'layer1_add_norm1')
    dot.edge('layer0_add_norm2', 'layer1_add_norm1')
    
    # Layer 1 FFN path
    dot.edge('layer1_add_norm1', 'layer1_ffn_up')
    dot.edge('layer1_add_norm1', 'layer1_ffn_gate')
    dot.edge('layer1_ffn_up', 'layer1_ffn_down')
    dot.edge('layer1_ffn_gate', 'layer1_ffn_down')
    dot.edge('layer1_ffn_down', 'layer1_ffn_dropout')
    dot.edge('layer1_ffn_dropout', 'layer1_add_norm2')
    dot.edge('layer1_add_norm1', 'layer1_add_norm2')
    
    # Layer 2 attention path
    dot.edge('layer1_add_norm2', 'layer2_attn_dropout')
    dot.edge('layer2_attn_dropout', 'layer2_add_norm1')
    dot.edge('layer1_add_norm2', 'layer2_add_norm1')
    
    # Layer 2 FFN path
    dot.edge('layer2_add_norm1', 'layer2_ffn_up')
    dot.edge('layer2_add_norm1', 'layer2_ffn_gate')
    dot.edge('layer2_ffn_up', 'layer2_ffn_down')
    dot.edge('layer2_ffn_gate', 'layer2_ffn_down')
    dot.edge('layer2_ffn_down', 'layer2_ffn_dropout')
    dot.edge('layer2_ffn_dropout', 'layer2_add_norm2')
    dot.edge('layer2_add_norm1', 'layer2_add_norm2')
    
    # Layer 3 attention path
    dot.edge('layer2_add_norm2', 'layer3_attn_dropout')
    dot.edge('layer3_attn_dropout', 'layer3_add_norm1')
    dot.edge('layer2_add_norm2', 'layer3_add_norm1')
    
    # Layer 3 FFN path
    dot.edge('layer3_add_norm1', 'layer3_ffn_up')
    dot.edge('layer3_add_norm1', 'layer3_ffn_gate')
    dot.edge('layer3_ffn_up', 'layer3_ffn_down')
    dot.edge('layer3_ffn_gate', 'layer3_ffn_down')
    dot.edge('layer3_ffn_down', 'layer3_ffn_dropout')
    dot.edge('layer3_ffn_dropout', 'layer3_add_norm2')
    dot.edge('layer3_add_norm1', 'layer3_add_norm2')
    
    # Final output
    dot.edge('layer3_add_norm2', 'final_norm')
    dot.edge('final_norm', 'output_proj')
    dot.edge('output_proj', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_fa_pool_dag()
    
    # Save DOT file
    dot_path = os.path.join('../outputs/2025-10-19-22-51-33', 'fa_pool_dag.dot')
    with open(dot_path, 'w') as f:
        f.write(dag.source)
    
    # Save SVG
    svg_path = os.path.join('../outputs/2025-10-19-22-51-33', 'fa_pool_dag.svg')
    dag.render(svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"FA Pool DAG generated:")
    print(f"DOT: {dot_path}")
    print(f"SVG: {svg_path}")