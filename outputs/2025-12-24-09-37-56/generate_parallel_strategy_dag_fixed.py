#!/usr/bin/env python3
"""
DAG Generator for LLaMA3-70B Parallel Strategy: PP=2×TP=4×SP=1
Fixed version addressing connectivity issues from previous submission.
"""

import graphviz
from graphviz import Digraph

def create_parallel_strategy_dag_fixed():
    """Create the complete DAG for PP=2×TP=4×SP=1 parallel strategy with proper connectivity"""
    
    # Create the main DAG
    dot = Digraph(name='LLaMA3-70B-Parallel-Strategy-DAG-Fixed')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('graph', bgcolor='white', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute nodes
    
    # ====================================================================================
    # STAGE 0: Layers 0-39 (GPUs 0,1,2,3) with TP=4
    # ====================================================================================
    
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0: Layers 0-39 (GPUs 0,1,2,3)', 
                   style='rounded,filled', fillcolor='lightyellow', fontname='Arial Bold')
        
        # Input processing
        stage0.node('input_stage0', 
                   'Input Embedding\nGPU: [0,1,2,3]\nInput: [batch_size=8, seq_len=4096]\nOutput: [batch_size=8, seq_len=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Create all 40 layers explicitly for proper connectivity
        prev_output = 'input_stage0'
        
        for layer in range(0, 40):
            layer_id = f'layer{layer}'
            
            # RMSNorm (Pre-Attention)
            stage0.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # Attention components
            stage0.node(f'{layer_id}_qkv_proj', 
                       f'Layer {layer}: QKV Projection\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=64, d_k=128]', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_qkv_split', 
                       f'Layer {layer}: QKV Split (TP=4)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=16, d_k=128] per GPU', 
                       shape='parallelogram', fillcolor='lightcyan')
            
            stage0.node(f'{layer_id}_attention', 
                       f'Layer {layer}: Attention Compute\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, heads=16, d_k=128]', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_attn_allreduce', 
                       f'Layer {layer}: Attention All-Reduce\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # RMSNorm (Pre-FFN)
            stage0.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # FFN components
            stage0.node(f'{layer_id}_ffn_gate', 
                       f'Layer {layer}: FFN Gate Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_up', 
                       f'Layer {layer}: FFN Up Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_down', 
                       f'Layer {layer}: FFN Down Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, ffn_dim=2752]\nOutput: [batch=8, seq=4096, hidden=8192] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_allreduce', 
                       f'Layer {layer}: FFN All-Reduce\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=2048] per GPU\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # Connect the layer
            dot.edge(prev_output, f'{layer_id}_norm1')
            dot.edge(f'{layer_id}_norm1', f'{layer_id}_qkv_proj')
            dot.edge(f'{layer_id}_qkv_proj', f'{layer_id}_qkv_split')
            dot.edge(f'{layer_id}_qkv_split', f'{layer_id}_attention')
            dot.edge(f'{layer_id}_attention', f'{layer_id}_attn_allreduce')
            dot.edge(f'{layer_id}_attn_allreduce', f'{layer_id}_norm2')
            dot.edge(f'{layer_id}_norm2', f'{layer_id}_ffn_gate')
            dot.edge(f'{layer_id}_ffn_gate', f'{layer_id}_ffn_up')
            dot.edge(f'{layer_id}_ffn_up', f'{layer_id}_ffn_down')
            dot.edge(f'{layer_id}_ffn_down', f'{layer_id}_ffn_allreduce')
            
            prev_output = f'{layer_id}_ffn_allreduce'
        
        # Stage 0 output
        stage0.node('stage0_output', 
                   'Stage 0 Output\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Connect final layer to stage output
        dot.edge(prev_output, 'stage0_output')
    
    # ====================================================================================
    # STAGE 1: Layers 40-79 (GPUs 4,5,6,7) with TP=4
    # ====================================================================================
    
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1: Layers 40-79 (GPUs 4,5,6,7)', 
                   style='rounded,filled', fillcolor='lightsteelblue', fontname='Arial Bold')
        
        # Stage 1 input (from Stage 0)
        stage1.node('input_stage1', 
                   'Stage 1 Input\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Create all 40 layers explicitly for proper connectivity
        prev_output = 'input_stage1'
        
        for layer in range(40, 80):
            layer_id = f'layer{layer}'
            
            # RMSNorm (Pre-Attention)
            stage1.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # Attention components
            stage1.node(f'{layer_id}_qkv_proj', 
                       f'Layer {layer}: QKV Projection\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=64, d_k=128]', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_qkv_split', 
                       f'Layer {layer}: QKV Split (TP=4)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=16, d_k=128] per GPU', 
                       shape='parallelogram', fillcolor='lightcyan')
            
            stage1.node(f'{layer_id}_attention', 
                       f'Layer {layer}: Attention Compute\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, heads=16, d_k=128]', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_attn_allreduce', 
                       f'Layer {layer}: Attention All-Reduce\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # RMSNorm (Pre-FFN)
            stage1.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # FFN components
            stage1.node(f'{layer_id}_ffn_gate', 
                       f'Layer {layer}: FFN Gate Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_up', 
                       f'Layer {layer}: FFN Up Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_down', 
                       f'Layer {layer}: FFN Down Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, ffn_dim=2752]\nOutput: [batch=8, seq=4096, hidden=8192] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_allreduce', 
                       f'Layer {layer}: FFN All-Reduce\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=2048] per GPU\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # Connect the layer
            dot.edge(prev_output, f'{layer_id}_norm1')
            dot.edge(f'{layer_id}_norm1', f'{layer_id}_qkv_proj')
            dot.edge(f'{layer_id}_qkv_proj', f'{layer_id}_qkv_split')
            dot.edge(f'{layer_id}_qkv_split', f'{layer_id}_attention')
            dot.edge(f'{layer_id}_attention', f'{layer_id}_attn_allreduce')
            dot.edge(f'{layer_id}_attn_allreduce', f'{layer_id}_norm2')
            dot.edge(f'{layer_id}_norm2', f'{layer_id}_ffn_gate')
            dot.edge(f'{layer_id}_ffn_gate', f'{layer_id}_ffn_up')
            dot.edge(f'{layer_id}_ffn_up', f'{layer_id}_ffn_down')
            dot.edge(f'{layer_id}_ffn_down', f'{layer_id}_ffn_allreduce')
            
            prev_output = f'{layer_id}_ffn_allreduce'
        
        # Final output processing
        stage1.node('final_norm', 
                   'Final RMSNorm\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightblue')
        
        stage1.node('logits_projection', 
                   'Logits Projection\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, vocab=128256] per GPU', 
                   fillcolor='lightblue')
        
        stage1.node('logits_allgather', 
                   'Logits All-Gather\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, vocab=32064] per GPU\nOutput: [batch=8, seq=4096, vocab=128256]', 
                   shape='ellipse', fillcolor='lightpink')
        
        stage1.node('output', 
                   'Final Output\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, vocab=128256]\nOutput: [batch=8, seq=4096, vocab=128256]', 
                   fillcolor='lightgreen')
        
        # Connect final layer to final processing
        dot.edge(prev_output, 'final_norm')
        dot.edge('final_norm', 'logits_projection')
        dot.edge('logits_projection', 'logits_allgather')
        dot.edge('logits_allgather', 'output')
    
    # ====================================================================================
    # COMMUNICATION BETWEEN STAGES
    # ====================================================================================
    
    # Pipeline communication (Stage 0 -> Stage 1)
    dot.node('pipeline_comm', 
            'Pipeline Communication\nStage 0 → Stage 1\nGPU: [0,1,2,3] → [4,5,6,7]\nData: [batch=8, seq=4096, hidden=8192]', 
            shape='ellipse', fillcolor='yellow', style='filled,dashed')
    
    # Connect stages
    dot.edge('stage0_output', 'pipeline_comm')
    dot.edge('pipeline_comm', 'input_stage1')
    
    return dot

def create_simplified_parallel_strategy_dag():
    """Create a simplified DAG showing representative layers with proper connectivity"""
    
    # Create the main DAG
    dot = Digraph(name='LLaMA3-70B-Parallel-Strategy-DAG-Simplified')
    dot.attr(rankdir='TB', size='30,40', dpi='300')
    dot.attr('graph', bgcolor='white', fontname='Arial')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')  # Compute nodes
    
    # ====================================================================================
    # STAGE 0: Layers 0-39 (GPUs 0,1,2,3) with TP=4
    # ====================================================================================
    
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0: Layers 0-39 (GPUs 0,1,2,3)', 
                   style='rounded,filled', fillcolor='lightyellow', fontname='Arial Bold')
        
        # Input processing
        stage0.node('input_stage0', 
                   'Input Embedding\nGPU: [0,1,2,3]\nInput: [batch_size=8, seq_len=4096]\nOutput: [batch_size=8, seq_len=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Show representative layers with proper connectivity
        representative_layers = [0, 10, 20, 30]
        prev_output = 'input_stage0'
        
        for i, layer in enumerate(representative_layers):
            layer_id = f'layer{layer}'
            
            # RMSNorm (Pre-Attention)
            stage0.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # Attention components
            stage0.node(f'{layer_id}_qkv_proj', 
                       f'Layer {layer}: QKV Projection\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=64, d_k=128]', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_qkv_split', 
                       f'Layer {layer}: QKV Split (TP=4)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=16, d_k=128] per GPU', 
                       shape='parallelogram', fillcolor='lightcyan')
            
            stage0.node(f'{layer_id}_attention', 
                       f'Layer {layer}: Attention Compute\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, heads=16, d_k=128]', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_attn_allreduce', 
                       f'Layer {layer}: Attention All-Reduce\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # RMSNorm (Pre-FFN)
            stage0.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # FFN components
            stage0.node(f'{layer_id}_ffn_gate', 
                       f'Layer {layer}: FFN Gate Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_up', 
                       f'Layer {layer}: FFN Up Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_down', 
                       f'Layer {layer}: FFN Down Proj\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, ffn_dim=2752]\nOutput: [batch=8, seq=4096, hidden=8192] per GPU', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_ffn_allreduce', 
                       f'Layer {layer}: FFN All-Reduce\nGPU: [0,1,2,3] TP=4\nInput: [batch=8, seq=4096, hidden=2048] per GPU\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # Connect the representative layer
            dot.edge(prev_output, f'{layer_id}_norm1')
            dot.edge(f'{layer_id}_norm1', f'{layer_id}_qkv_proj')
            dot.edge(f'{layer_id}_qkv_proj', f'{layer_id}_qkv_split')
            dot.edge(f'{layer_id}_qkv_split', f'{layer_id}_attention')
            dot.edge(f'{layer_id}_attention', f'{layer_id}_attn_allreduce')
            dot.edge(f'{layer_id}_attn_allreduce', f'{layer_id}_norm2')
            dot.edge(f'{layer_id}_norm2', f'{layer_id}_ffn_gate')
            dot.edge(f'{layer_id}_ffn_gate', f'{layer_id}_ffn_up')
            dot.edge(f'{layer_id}_ffn_up', f'{layer_id}_ffn_down')
            dot.edge(f'{layer_id}_ffn_down', f'{layer_id}_ffn_allreduce')
            
            # For intermediate representative layers, add placeholder connections
            if i < len(representative_layers) - 1:
                next_layer = representative_layers[i + 1]
                # Add intermediate layer placeholder
                stage0.node(f'{layer_id}_intermediate', 
                           f'Layers {layer+1}-{next_layer-1}\nIntermediate Processing\nGPU: [0,1,2,3]', 
                           shape='ellipse', fillcolor='lightgray', style='filled,dashed')
                dot.edge(f'{layer_id}_ffn_allreduce', f'{layer_id}_intermediate')
                prev_output = f'{layer_id}_intermediate'
            else:
                # Last representative layer
                prev_output = f'{layer_id}_ffn_allreduce'
        
        # Stage 0 output
        stage0.node('stage0_output', 
                   'Stage 0 Output\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Connect final representative layer to stage output
        dot.edge(prev_output, 'stage0_output')
    
    # ====================================================================================
    # STAGE 1: Layers 40-79 (GPUs 4,5,6,7) with TP=4
    # ====================================================================================
    
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1: Layers 40-79 (GPUs 4,5,6,7)', 
                   style='rounded,filled', fillcolor='lightsteelblue', fontname='Arial Bold')
        
        # Stage 1 input (from Stage 0)
        stage1.node('input_stage1', 
                   'Stage 1 Input\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightgreen')
        
        # Show representative layers with proper connectivity
        representative_layers = [40, 50, 60, 70]
        prev_output = 'input_stage1'
        
        for i, layer in enumerate(representative_layers):
            layer_id = f'layer{layer}'
            
            # RMSNorm (Pre-Attention)
            stage1.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # Attention components
            stage1.node(f'{layer_id}_qkv_proj', 
                       f'Layer {layer}: QKV Projection\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=64, d_k=128]', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_qkv_split', 
                       f'Layer {layer}: QKV Split (TP=4)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, heads=16, d_k=128] per GPU', 
                       shape='parallelogram', fillcolor='lightcyan')
            
            stage1.node(f'{layer_id}_attention', 
                       f'Layer {layer}: Attention Compute\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, heads=16, d_k=128]', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_attn_allreduce', 
                       f'Layer {layer}: Attention All-Reduce\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, heads=16, d_k=128]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # RMSNorm (Pre-FFN)
            stage1.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            # FFN components
            stage1.node(f'{layer_id}_ffn_gate', 
                       f'Layer {layer}: FFN Gate Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_up', 
                       f'Layer {layer}: FFN Up Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, ffn_dim=2752] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_down', 
                       f'Layer {layer}: FFN Down Proj\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, ffn_dim=2752]\nOutput: [batch=8, seq=4096, hidden=8192] per GPU', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_ffn_allreduce', 
                       f'Layer {layer}: FFN All-Reduce\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=2048] per GPU\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       shape='ellipse', fillcolor='lightpink')
            
            # Connect the representative layer
            dot.edge(prev_output, f'{layer_id}_norm1')
            dot.edge(f'{layer_id}_norm1', f'{layer_id}_qkv_proj')
            dot.edge(f'{layer_id}_qkv_proj', f'{layer_id}_qkv_split')
            dot.edge(f'{layer_id}_qkv_split', f'{layer_id}_attention')
            dot.edge(f'{layer_id}_attention', f'{layer_id}_attn_allreduce')
            dot.edge(f'{layer_id}_attn_allreduce', f'{layer_id}_norm2')
            dot.edge(f'{layer_id}_norm2', f'{layer_id}_ffn_gate')
            dot.edge(f'{layer_id}_ffn_gate', f'{layer_id}_ffn_up')
            dot.edge(f'{layer_id}_ffn_up', f'{layer_id}_ffn_down')
            dot.edge(f'{layer_id}_ffn_down', f'{layer_id}_ffn_allreduce')
            
            # For intermediate representative layers, add placeholder connections
            if i < len(representative_layers) - 1:
                next_layer = representative_layers[i + 1]
                # Add intermediate layer placeholder
                stage1.node(f'{layer_id}_intermediate', 
                           f'Layers {layer+1}-{next_layer-1}\nIntermediate Processing\nGPU: [4,5,6,7]', 
                           shape='ellipse', fillcolor='lightgray', style='filled,dashed')
                dot.edge(f'{layer_id}_ffn_allreduce', f'{layer_id}_intermediate')
                prev_output = f'{layer_id}_intermediate'
            else:
                # Last representative layer
                prev_output = f'{layer_id}_ffn_allreduce'
        
        # Final output processing
        stage1.node('final_norm', 
                   'Final RMSNorm\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightblue')
        
        stage1.node('logits_projection', 
                   'Logits Projection\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, vocab=128256] per GPU', 
                   fillcolor='lightblue')
        
        stage1.node('logits_allgather', 
                   'Logits All-Gather\nGPU: [4,5,6,7] TP=4\nInput: [batch=8, seq=4096, vocab=32064] per GPU\nOutput: [batch=8, seq=4096, vocab=128256]', 
                   shape='ellipse', fillcolor='lightpink')
        
        stage1.node('output', 
                   'Final Output\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, vocab=128256]\nOutput: [batch=8, seq=4096, vocab=128256]', 
                   fillcolor='lightgreen')
        
        # Connect final representative layer to final processing
        dot.edge(prev_output, 'final_norm')
        dot.edge('final_norm', 'logits_projection')
        dot.edge('logits_projection', 'logits_allgather')
        dot.edge('logits_allgather', 'output')
    
    # ====================================================================================
    # COMMUNICATION BETWEEN STAGES
    # ====================================================================================
    
    # Pipeline communication (Stage 0 -> Stage 1)
    dot.node('pipeline_comm', 
            'Pipeline Communication\nStage 0 → Stage 1\nGPU: [0,1,2,3] → [4,5,6,7]\nData: [batch=8, seq=4096, hidden=8192]', 
            shape='ellipse', fillcolor='yellow', style='filled,dashed')
    
    # Connect stages
    dot.edge('stage0_output', 'pipeline_comm')
    dot.edge('pipeline_comm', 'input_stage1')
    
    return dot

def main():
    """Generate both DAG versions and save files"""
    print("Generating LLaMA3-70B Parallel Strategy DAG (Fixed Version)...")
    
    # Create the simplified DAG (more readable)
    dag = create_simplified_parallel_strategy_dag()
    
    # Save DOT file
    dot_file = '../outputs/2025-12-24-09-37-56/llama3_70b_parallel_strategy_fixed.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    print(f"DOT file saved: {dot_file}")
    
    # Render SVG
    svg_file = '../outputs/2025-12-24-09-37-56/llama3_70b_parallel_strategy_fixed.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved: {svg_file}")
    
    print("DAG generation completed successfully!")
    print(f"Strategy: PP=2×TP=4×SP=1")
    print(f"GPUs: 8 (Stage 0: GPUs 0-3, Stage 1: GPUs 4-7)")
    print("Fixed connectivity issues from previous submission")

if __name__ == '__main__':
    main()