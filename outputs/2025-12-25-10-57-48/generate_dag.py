#!/usr/bin/env python3

import graphviz
import os

def create_llama_dag():
    # Create the DAG
    dot = graphviz.Digraph(comment='Llama3-70B Parallel Strategy Deployment DAG')
    dot.attr(rankdir='TB', size='20,30')
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=64, seq_len=8192, hidden_size=8192]\\nOutput: [batch_size=64, seq_len=8192, hidden_size=8192]', 
             shape='diamond', style='filled', fillcolor='lightpink')
    
    # Stage 0: GPUs 0,1 - Layers 0-19
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Stage 0: GPUs 0,1 (Layers 0-19)', style='rounded, filled', fillcolor='lightgray')
        
        # GPU 0 computations
        with c.subgraph(name='cluster_gpu0') as gpu0:
            gpu0.attr(label='GPU 0', style='rounded')
            
            # Layer 0 operations on GPU 0
            gpu0.node('gpu0_layer0_attn_qkv', 'Layer 0 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu0.node('gpu0_layer0_attn_gate', 'Layer 0 Attention Gate\\nRouting\\nInput: [64, 8192, 3072]\\nOutput: [64, 8192, 1536]', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            gpu0.node('gpu0_layer0_attn_out', 'Layer 0 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu0.node('gpu0_layer0_mlp_gate', 'Layer 0 MLP Gate\\nRouting\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            gpu0.node('gpu0_layer0_mlp_up', 'Layer 0 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu0.node('gpu0_layer0_mlp_down', 'Layer 0 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # GPU 1 computations
        with c.subgraph(name='cluster_gpu1') as gpu1:
            gpu1.attr(label='GPU 1', style='rounded')
            
            # Layer 0 operations on GPU 1
            gpu1.node('gpu1_layer0_attn_qkv', 'Layer 0 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu1.node('gpu1_layer0_attn_gate', 'Layer 0 Attention Gate\\nRouting\\nInput: [64, 8192, 3072]\\nOutput: [64, 8192, 1536]', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            gpu1.node('gpu1_layer0_attn_out', 'Layer 0 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu1.node('gpu1_layer0_mlp_gate', 'Layer 0 MLP Gate\\nRouting\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            
            gpu1.node('gpu1_layer0_mlp_up', 'Layer 0 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu1.node('gpu1_layer0_mlp_down', 'Layer 0 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Communication nodes for tensor parallelism
        c.node('stage0_tp_allreduce1', 'TP All-Reduce\\nAttention Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage0_tp_allreduce2', 'TP All-Reduce\\nMLP Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Stage 0 output
        c.node('stage0_output', 'Stage 0 Output\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='diamond', style='filled', fillcolor='lightcoral')
    
    # Stage 1: GPUs 2,3 - Layers 20-39
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1: GPUs 2,3 (Layers 20-39)', style='rounded, filled', fillcolor='lightgray')
        
        # GPU 2 computations
        with c.subgraph(name='cluster_gpu2') as gpu2:
            gpu2.attr(label='GPU 2', style='rounded')
            
            gpu2.node('gpu2_layer20_attn_qkv', 'Layer 20 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu2.node('gpu2_layer20_attn_out', 'Layer 20 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu2.node('gpu2_layer20_mlp_up', 'Layer 20 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu2.node('gpu2_layer20_mlp_down', 'Layer 20 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # GPU 3 computations
        with c.subgraph(name='cluster_gpu3') as gpu3:
            gpu3.attr(label='GPU 3', style='rounded')
            
            gpu3.node('gpu3_layer20_attn_qkv', 'Layer 20 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu3.node('gpu3_layer20_attn_out', 'Layer 20 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu3.node('gpu3_layer20_mlp_up', 'Layer 20 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu3.node('gpu3_layer20_mlp_down', 'Layer 20 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Communication nodes
        c.node('stage1_tp_allreduce1', 'TP All-Reduce\\nAttention Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage1_tp_allreduce2', 'TP All-Reduce\\nMLP Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Pipeline communication from Stage 0 to Stage 1
        c.node('p2p_send_0_1', 'Pipeline Send\\nStage0→Stage1\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('p2p_recv_0_1', 'Pipeline Receive\\nStage0→Stage1\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage1_output', 'Stage 1 Output\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='diamond', style='filled', fillcolor='lightcoral')
    
    # Stage 2: GPUs 4,5 - Layers 40-59
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Stage 2: GPUs 4,5 (Layers 40-59)', style='rounded, filled', fillcolor='lightgray')
        
        # GPU 4 computations
        with c.subgraph(name='cluster_gpu4') as gpu4:
            gpu4.attr(label='GPU 4', style='rounded')
            
            gpu4.node('gpu4_layer40_attn_qkv', 'Layer 40 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu4.node('gpu4_layer40_attn_out', 'Layer 40 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu4.node('gpu4_layer40_mlp_up', 'Layer 40 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu4.node('gpu4_layer40_mlp_down', 'Layer 40 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # GPU 5 computations
        with c.subgraph(name='cluster_gpu5') as gpu5:
            gpu5.attr(label='GPU 5', style='rounded')
            
            gpu5.node('gpu5_layer40_attn_qkv', 'Layer 40 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu5.node('gpu5_layer40_attn_out', 'Layer 40 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu5.node('gpu5_layer40_mlp_up', 'Layer 40 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu5.node('gpu5_layer40_mlp_down', 'Layer 40 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Communication nodes
        c.node('stage2_tp_allreduce1', 'TP All-Reduce\\nAttention Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage2_tp_allreduce2', 'TP All-Reduce\\nMLP Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Pipeline communication from Stage 1 to Stage 2
        c.node('p2p_send_1_2', 'Pipeline Send\\nStage1→Stage2\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('p2p_recv_1_2', 'Pipeline Receive\\nStage1→Stage2\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage2_output', 'Stage 2 Output\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='diamond', style='filled', fillcolor='lightcoral')
    
    # Stage 3: GPUs 6,7 - Layers 60-79
    with dot.subgraph(name='cluster_stage3') as c:
        c.attr(label='Stage 3: GPUs 6,7 (Layers 60-79)', style='rounded, filled', fillcolor='lightgray')
        
        # GPU 6 computations
        with c.subgraph(name='cluster_gpu6') as gpu6:
            gpu6.attr(label='GPU 6', style='rounded')
            
            gpu6.node('gpu6_layer60_attn_qkv', 'Layer 60 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu6.node('gpu6_layer60_attn_out', 'Layer 60 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu6.node('gpu6_layer60_mlp_up', 'Layer 60 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu6.node('gpu6_layer60_mlp_down', 'Layer 60 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Final output layer
            gpu6.node('gpu6_lm_head', 'LM Head\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 128256]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # GPU 7 computations
        with c.subgraph(name='cluster_gpu7') as gpu7:
            gpu7.attr(label='GPU 7', style='rounded')
            
            gpu7.node('gpu7_layer60_attn_qkv', 'Layer 60 Attention QKV Linear\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 3072]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu7.node('gpu7_layer60_attn_out', 'Layer 60 Attention Output Linear\\nRow Parallel\\nInput: [64, 8192, 1536]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu7.node('gpu7_layer60_mlp_up', 'Layer 60 MLP Up Linear\\nColumn Parallel\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 28672]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            gpu7.node('gpu7_layer60_mlp_down', 'Layer 60 MLP Down Linear\\nRow Parallel\\nInput: [64, 8192, 28672]\\nOutput: [64, 8192, 4096]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Final output layer
            gpu7.node('gpu7_lm_head', 'LM Head\\nColumn Parallel\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 128256]', 
                     shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Communication nodes
        c.node('stage3_tp_allreduce1', 'TP All-Reduce\\nAttention Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('stage3_tp_allreduce2', 'TP All-Reduce\\nMLP Output\\nInput: [64, 8192, 4096]\\nOutput: [64, 8192, 4096]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Pipeline communication from Stage 2 to Stage 3
        c.node('p2p_send_2_3', 'Pipeline Send\\nStage2→Stage3\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('p2p_recv_2_3', 'Pipeline Receive\\nStage2→Stage3\\nInput: [64, 8192, 8192]\\nOutput: [64, 8192, 8192]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Final all-reduce for output
        c.node('final_allreduce', 'Final All-Reduce\\nLM Head Output\\nInput: [64, 8192, 128256]\\nOutput: [64, 8192, 128256]', 
              shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.node('output', 'Final Output\\nInput: [64, 8192, 128256]\\nOutput: [64, 8192, 128256]', 
              shape='diamond', style='filled', fillcolor='lightpink')
    
    # Define edges (connections)
    # Input to Stage 0
    dot.edge('input', 'gpu0_layer0_attn_qkv')
    dot.edge('input', 'gpu1_layer0_attn_qkv')
    
    # Stage 0 internal connections
    dot.edge('gpu0_layer0_attn_qkv', 'gpu0_layer0_attn_gate')
    dot.edge('gpu1_layer0_attn_qkv', 'gpu1_layer0_attn_gate')
    dot.edge('gpu0_layer0_attn_gate', 'gpu0_layer0_attn_out')
    dot.edge('gpu1_layer0_attn_gate', 'gpu1_layer0_attn_out')
    dot.edge('gpu0_layer0_attn_out', 'stage0_tp_allreduce1')
    dot.edge('gpu1_layer0_attn_out', 'stage0_tp_allreduce1')
    dot.edge('stage0_tp_allreduce1', 'gpu0_layer0_mlp_gate')
    dot.edge('stage0_tp_allreduce1', 'gpu0_layer0_mlp_up')
    dot.edge('stage0_tp_allreduce1', 'gpu1_layer0_mlp_gate')
    dot.edge('stage0_tp_allreduce1', 'gpu1_layer0_mlp_up')
    dot.edge('gpu0_layer0_mlp_gate', 'gpu0_layer0_mlp_down')
    dot.edge('gpu1_layer0_mlp_gate', 'gpu1_layer0_mlp_down')
    dot.edge('gpu0_layer0_mlp_up', 'gpu0_layer0_mlp_down')
    dot.edge('gpu1_layer0_mlp_up', 'gpu1_layer0_mlp_down')
    dot.edge('gpu0_layer0_mlp_down', 'stage0_tp_allreduce2')
    dot.edge('gpu1_layer0_mlp_down', 'stage0_tp_allreduce2')
    dot.edge('stage0_tp_allreduce2', 'stage0_output')
    
    # Pipeline communication Stage 0 to Stage 1
    dot.edge('stage0_output', 'p2p_send_0_1')
    dot.edge('p2p_send_0_1', 'p2p_recv_0_1')
    dot.edge('p2p_recv_0_1', 'gpu2_layer20_attn_qkv')
    dot.edge('p2p_recv_0_1', 'gpu3_layer20_attn_qkv')
    
    # Stage 1 internal connections (simplified - similar to Stage 0)
    dot.edge('gpu2_layer20_attn_qkv', 'gpu2_layer20_attn_out')
    dot.edge('gpu3_layer20_attn_qkv', 'gpu3_layer20_attn_out')
    dot.edge('gpu2_layer20_attn_out', 'stage1_tp_allreduce1')
    dot.edge('gpu3_layer20_attn_out', 'stage1_tp_allreduce1')
    dot.edge('stage1_tp_allreduce1', 'gpu2_layer20_mlp_up')
    dot.edge('stage1_tp_allreduce1', 'gpu3_layer20_mlp_up')
    dot.edge('gpu2_layer20_mlp_up', 'gpu2_layer20_mlp_down')
    dot.edge('gpu3_layer20_mlp_up', 'gpu3_layer20_mlp_down')
    dot.edge('gpu2_layer20_mlp_down', 'stage1_tp_allreduce2')
    dot.edge('gpu3_layer20_mlp_down', 'stage1_tp_allreduce2')
    dot.edge('stage1_tp_allreduce2', 'stage1_output')
    
    # Pipeline communication Stage 1 to Stage 2
    dot.edge('stage1_output', 'p2p_send_1_2')
    dot.edge('p2p_send_1_2', 'p2p_recv_1_2')
    dot.edge('p2p_recv_1_2', 'gpu4_layer40_attn_qkv')
    dot.edge('p2p_recv_1_2', 'gpu5_layer40_attn_qkv')
    
    # Stage 2 internal connections (simplified - similar to Stage 0)
    dot.edge('gpu4_layer40_attn_qkv', 'gpu4_layer40_attn_out')
    dot.edge('gpu5_layer40_attn_qkv', 'gpu5_layer40_attn_out')
    dot.edge('gpu4_layer40_attn_out', 'stage2_tp_allreduce1')
    dot.edge('gpu5_layer40_attn_out', 'stage2_tp_allreduce1')
    dot.edge('stage2_tp_allreduce1', 'gpu4_layer40_mlp_up')
    dot.edge('stage2_tp_allreduce1', 'gpu5_layer40_mlp_up')
    dot.edge('gpu4_layer40_mlp_up', 'gpu4_layer40_mlp_down')
    dot.edge('gpu5_layer40_mlp_up', 'gpu5_layer40_mlp_down')
    dot.edge('gpu4_layer40_mlp_down', 'stage2_tp_allreduce2')
    dot.edge('gpu5_layer40_mlp_down', 'stage2_tp_allreduce2')
    dot.edge('stage2_tp_allreduce2', 'stage2_output')
    
    # Pipeline communication Stage 2 to Stage 3
    dot.edge('stage2_output', 'p2p_send_2_3')
    dot.edge('p2p_send_2_3', 'p2p_recv_2_3')
    dot.edge('p2p_recv_2_3', 'gpu6_layer60_attn_qkv')
    dot.edge('p2p_recv_2_3', 'gpu7_layer60_attn_qkv')
    
    # Stage 3 internal connections (simplified - similar to Stage 0)
    dot.edge('gpu6_layer60_attn_qkv', 'gpu6_layer60_attn_out')
    dot.edge('gpu7_layer60_attn_qkv', 'gpu7_layer60_attn_out')
    dot.edge('gpu6_layer60_attn_out', 'stage3_tp_allreduce1')
    dot.edge('gpu7_layer60_attn_out', 'stage3_tp_allreduce1')
    dot.edge('stage3_tp_allreduce1', 'gpu6_layer60_mlp_up')
    dot.edge('stage3_tp_allreduce1', 'gpu7_layer60_mlp_up')
    dot.edge('gpu6_layer60_mlp_up', 'gpu6_layer60_mlp_down')
    dot.edge('gpu7_layer60_mlp_up', 'gpu7_layer60_mlp_down')
    dot.edge('gpu6_layer60_mlp_down', 'stage3_tp_allreduce2')
    dot.edge('gpu7_layer60_mlp_down', 'stage3_tp_allreduce2')
    
    # Final LM Head and output
    dot.edge('stage3_tp_allreduce2', 'gpu6_lm_head')
    dot.edge('stage3_tp_allreduce2', 'gpu7_lm_head')
    dot.edge('gpu6_lm_head', 'final_allreduce')
    dot.edge('gpu7_lm_head', 'final_allreduce')
    dot.edge('final_allreduce', 'output')
    
    return dot

def main():
    # Create the output directory if it doesn't exist
    os.makedirs('../outputs/2025-12-25-10-57-48', exist_ok=True)
    
    # Generate the DAG
    dag = create_llama_dag()
    
    # Save as DOT file
    dot_file_path = '../outputs/2025-12-25-10-57-48/llama70b_parallel_dag.dot'
    dag.save(dot_file_path)
    
    # Save as SVG image
    svg_file_path = '../outputs/2025-12-25-10-57-48/llama70b_parallel_dag.svg'
    dag.render('../outputs/2025-12-25-10-57-48/llama70b_parallel_dag', format='svg', cleanup=True)
    
    print(f"DAG generated successfully!")
    print(f"DOT file: {dot_file_path}")
    print(f"SVG file: {svg_file_path}")
    
    # Return the paths in JSON format
    import json
    result = {
        "dot_file": dot_file_path,
        "svg_file": svg_file_path,
        "graphviz_code": dag.source
    }
    
    with open('../outputs/2025-12-25-10-57-48/dag_paths.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nGenerated files:")
    print(f"DOT: {dot_file_path}")
    print(f"SVG: {svg_file_path}")
    print(f"JSON: ../outputs/2025-12-25-10-57-48/dag_paths.json")

if __name__ == "__main__":
    main()