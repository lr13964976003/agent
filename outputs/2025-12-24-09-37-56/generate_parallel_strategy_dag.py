#!/usr/bin/env python3
"""
DAG Generator for LLaMA3-70B Parallel Strategy: PP=2×TP=4×SP=1
This script creates a detailed operator-level DAG for the parallel strategy.
"""

import graphviz
from graphviz import Digraph

def create_parallel_strategy_dag():
    """Create the complete DAG for PP=2×TP=4×SP=1 parallel strategy"""
    
    # Create the main DAG
    dot = Digraph(name='LLaMA3-70B-Parallel-Strategy-DAG')
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
        
        # Layer 0-39 processing (showing representative layers)
        for layer in range(0, 40, 10):  # Show every 10th layer for clarity
            layer_id = f'layer{layer}'
            
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
            
            # RMSNorm (no communication needed)
            stage0.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            stage0.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
        
        # Stage 0 output
        stage0.node('stage0_output', 
                   'Stage 0 Output\nGPU: [0,1,2,3]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                   fillcolor='lightgreen')
    
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
        
        # Layer 40-79 processing (showing representative layers)
        for layer in range(40, 80, 10):  # Show every 10th layer for clarity
            layer_id = f'layer{layer}'
            
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
            
            # RMSNorm
            stage1.node(f'{layer_id}_norm1', 
                       f'Layer {layer}: RMSNorm (Pre-Attn)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
            
            stage1.node(f'{layer_id}_norm2', 
                       f'Layer {layer}: RMSNorm (Pre-FFN)\nGPU: [4,5,6,7]\nInput: [batch=8, seq=4096, hidden=8192]\nOutput: [batch=8, seq=4096, hidden=8192]', 
                       fillcolor='lightblue')
        
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
    
    # ====================================================================================
    # COMMUNICATION BETWEEN STAGES
    # ====================================================================================
    
    # Pipeline communication (Stage 0 -> Stage 1)
    dot.node('pipeline_comm', 
            'Pipeline Communication\nStage 0 → Stage 1\nGPU: [0,1,2,3] → [4,5,6,7]\nData: [batch=8, seq=4096, hidden=8192]', 
            shape='ellipse', fillcolor='yellow', style='filled,dashed')
    
    # ====================================================================================
    # EDGES - DATA FLOW
    # ====================================================================================
    
    # Stage 0 flow
    dot.edge('input_stage0', 'layer0_qkv_proj')
    dot.edge('layer0_qkv_proj', 'layer0_qkv_split')
    dot.edge('layer0_qkv_split', 'layer0_attention')
    dot.edge('layer0_attention', 'layer0_attn_allreduce')
    dot.edge('layer0_attn_allreduce', 'layer0_norm1')
    dot.edge('layer0_norm1', 'layer0_ffn_gate')
    dot.edge('layer0_ffn_gate', 'layer0_ffn_up')
    dot.edge('layer0_ffn_up', 'layer0_ffn_down')
    dot.edge('layer0_ffn_down', 'layer0_ffn_allreduce')
    dot.edge('layer0_ffn_allreduce', 'layer0_norm2')
    
    # Connect representative layers (simplified for visualization)
    dot.edge('layer0_norm2', 'layer10_qkv_proj', style='dashed', label='Layers 1-9')
    dot.edge('layer10_norm2', 'layer20_qkv_proj', style='dashed', label='Layers 11-19')
    dot.edge('layer20_norm2', 'layer30_qkv_proj', style='dashed', label='Layers 21-29')
    dot.edge('layer30_norm2', 'stage0_output', style='dashed', label='Layers 31-39')
    
    # Pipeline communication
    dot.edge('stage0_output', 'pipeline_comm')
    dot.edge('pipeline_comm', 'input_stage1')
    
    # Stage 1 flow
    dot.edge('input_stage1', 'layer40_qkv_proj')
    dot.edge('layer40_qkv_proj', 'layer40_qkv_split')
    dot.edge('layer40_qkv_split', 'layer40_attention')
    dot.edge('layer40_attention', 'layer40_attn_allreduce')
    dot.edge('layer40_attn_allreduce', 'layer40_norm1')
    dot.edge('layer40_norm1', 'layer40_ffn_gate')
    dot.edge('layer40_ffn_gate', 'layer40_ffn_up')
    dot.edge('layer40_ffn_up', 'layer40_ffn_down')
    dot.edge('layer40_ffn_down', 'layer40_ffn_allreduce')
    dot.edge('layer40_ffn_allreduce', 'layer40_norm2')
    
    # Connect representative layers in Stage 1
    dot.edge('layer40_norm2', 'layer50_qkv_proj', style='dashed', label='Layers 41-49')
    dot.edge('layer50_norm2', 'layer60_qkv_proj', style='dashed', label='Layers 51-59')
    dot.edge('layer60_norm2', 'layer70_qkv_proj', style='dashed', label='Layers 61-69')
    dot.edge('layer70_norm2', 'final_norm', style='dashed', label='Layers 71-79')
    
    # Final output flow
    dot.edge('final_norm', 'logits_projection')
    dot.edge('logits_projection', 'logits_allgather')
    dot.edge('logits_allgather', 'output')
    
    return dot

def main():
    """Generate the DAG and save files"""
    print("Generating LLaMA3-70B Parallel Strategy DAG...")
    
    # Create the DAG
    dag = create_parallel_strategy_dag()
    
    # Save DOT file
    dot_file = '../outputs/2025-12-24-09-37-56/llama3_70b_parallel_strategy.dot'
    with open(dot_file, 'w') as f:
        f.write(dag.source)
    print(f"DOT file saved: {dot_file}")
    
    # Render SVG
    svg_file = '../outputs/2025-12-24-09-37-56/llama3_70b_parallel_strategy.svg'
    dag.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
    print(f"SVG file saved: {svg_file}")
    
    print("DAG generation completed successfully!")
    print(f"Total nodes: {len(dag.body)}")
    print(f"Strategy: PP=2×TP=4×SP=1")
    print(f"GPUs: 8 (Stage 0: GPUs 0-3, Stage 1: GPUs 4-7)")

if __name__ == '__main__':
    main()