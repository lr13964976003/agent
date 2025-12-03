#!/usr/bin/env python3

import graphviz
import os

def create_deployment_dag():
    """Create a comprehensive DAG for the hybrid tensor-parallel pipeline deployment strategy"""
    
    # Create directed graph
    dot = graphviz.Digraph(comment='Hybrid Tensor-Parallel Pipeline Deployment Strategy')
    dot.attr(rankdir='TB', size='20,30', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation  
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Stage 0: GPU 0 - Input Processing and Embedding
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Stage 0: GPU 0 - Input Processing', style='rounded,filled', fillcolor='lightgray')
        
        # Input node
        c.node('input', 'Input\nInput: [batch_size=1, seq_len=1024]\nOutput: [batch_size=1, seq_len=1024]', 
               shape='ellipse', fillcolor='lightblue')
        
        # Tokenization (computation)
        c.node('tokenize', 'Tokenize\nInput: [batch_size=1, seq_len=1024]\nOutput: [batch_size=1, seq_len=1024, vocab_size]',
               shape='rectangle', fillcolor='lightgreen')
        
        # Embedding layer (computation)
        c.node('embed', 'Embedding\nInput: [batch_size=1, seq_len=1024, vocab_size]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
               shape='rectangle', fillcolor='lightgreen')
        
        # Position encoding (computation)
        c.node('pos_enc', 'Position Encoding\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Communication: Send to Stage 1
    dot.node('comm_stage0_to_1', 'Send to Stage 1\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
             shape='ellipse', fillcolor='lightblue')
    
    # Stage 1: GPUs 1-2 - Expert Layer (Tensor Parallel)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Stage 1: GPUs 1-2 - Expert Layer (Tensor Parallel)', style='rounded,filled', fillcolor='lightgray')
        
        # GPU 1 computations
        with c.subgraph(name='cluster_gpu1') as gpu1:
            gpu1.attr(label='GPU 1 - Tensor Parallel Part 1', style='rounded,filled', fillcolor='lightcyan')
            
            # Layer normalization (computation)
            gpu1.node('ln_gpu1', 'LayerNorm\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # First linear layer - column parallel (computation)
            gpu1.node('linear1_gpu1', 'Linear1 (Col-Parallel)\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, ffn=8192/2]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # GELU activation (computation)
            gpu1.node('gelu_gpu1', 'GELU\nInput: [batch_size=1, seq_len=1024, ffn=4096]\nOutput: [batch_size=1, seq_len=1024, ffn=4096]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # Second linear layer - row parallel (computation)
            gpu1.node('linear2_gpu1', 'Linear2 (Row-Parallel)\nInput: [batch_size=1, seq_len=1024, ffn=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
                     shape='rectangle', fillcolor='lightgreen')
        
        # GPU 2 computations  
        with c.subgraph(name='cluster_gpu2') as gpu2:
            gpu2.attr(label='GPU 2 - Tensor Parallel Part 2', style='rounded,filled', fillcolor='lightcyan')
            
            # Layer normalization (computation)
            gpu2.node('ln_gpu2', 'LayerNorm\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # First linear layer - column parallel (computation)
            gpu2.node('linear1_gpu2', 'Linear1 (Col-Parallel)\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, ffn=8192/2]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # GELU activation (computation)
            gpu2.node('gelu_gpu2', 'GELU\nInput: [batch_size=1, seq_len=1024, ffn=4096]\nOutput: [batch_size=1, seq_len=1024, ffn=4096]',
                     shape='rectangle', fillcolor='lightgreen')
            
            # Second linear layer - row parallel (computation)
            gpu2.node('linear2_gpu2', 'Linear2 (Row-Parallel)\nInput: [batch_size=1, seq_len=1024, ffn=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
                     shape='rectangle', fillcolor='lightgreen')
    
    # Communication: All-reduce sum for tensor parallel output
    dot.node('all_reduce', 'All-Reduce Sum\nInput: [batch_size=1, seq_len=1024, hidden=4096] x2\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
             shape='ellipse', fillcolor='lightblue')
    
    # Aggregation node
    dot.node('agg', 'Aggregate\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
             shape='parallelogram', fillcolor='lightyellow')
    
    # Communication: Send to Stage 2
    dot.node('comm_stage1_to_2', 'Send to Stage 2\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
             shape='ellipse', fillcolor='lightblue')
    
    # Stage 2: GPU 0 - Aggregation and Output
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Stage 2: GPU 0 - Aggregation and Output', style='rounded,filled', fillcolor='lightgray')
        
        # Residual connection (routing/aggregation)
        c.node('residual', 'Residual Add\nInput: [batch_size=1, seq_len=1024, hidden=4096] x2\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
               shape='parallelogram', fillcolor='lightyellow')
        
        # Final layer normalization (computation)
        c.node('final_ln', 'Final LayerNorm\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, hidden=4096]',
               shape='rectangle', fillcolor='lightgreen')
        
        # Output projection (computation)
        c.node('output_proj', 'Output Projection\nInput: [batch_size=1, seq_len=1024, hidden=4096]\nOutput: [batch_size=1, seq_len=1024, vocab_size]',
               shape='rectangle', fillcolor='lightgreen')
        
        # Softmax (computation)
        c.node('softmax', 'Softmax\nInput: [batch_size=1, seq_len=1024, vocab_size]\nOutput: [batch_size=1, seq_len=1024, vocab_size]',
               shape='rectangle', fillcolor='lightgreen')
    
    # Output node
    dot.node('output', 'Output\nInput: [batch_size=1, seq_len=1024, vocab_size]\nOutput: [batch_size=1, seq_len=1024, vocab_size]',
             shape='ellipse', fillcolor='lightblue')
    
    # Define edges (connections)
    # Stage 0 flow
    dot.edge('input', 'tokenize')
    dot.edge('tokenize', 'embed')
    dot.edge('embed', 'pos_enc')
    dot.edge('pos_enc', 'comm_stage0_to_1')
    
    # Stage 1 flow - split to both GPUs
    dot.edge('comm_stage0_to_1', 'ln_gpu1')
    dot.edge('comm_stage0_to_1', 'ln_gpu2')
    
    # GPU 1 computation flow
    dot.edge('ln_gpu1', 'linear1_gpu1')
    dot.edge('linear1_gpu1', 'gelu_gpu1')
    dot.edge('gelu_gpu1', 'linear2_gpu1')
    
    # GPU 2 computation flow
    dot.edge('ln_gpu2', 'linear1_gpu2')
    dot.edge('linear1_gpu2', 'gelu_gpu2')
    dot.edge('gelu_gpu2', 'linear2_gpu2')
    
    # Tensor parallel aggregation
    dot.edge('linear2_gpu1', 'all_reduce')
    dot.edge('linear2_gpu2', 'all_reduce')
    dot.edge('all_reduce', 'agg')
    dot.edge('agg', 'comm_stage1_to_2')
    
    # Stage 2 flow
    dot.edge('comm_stage1_to_2', 'residual')
    # Note: residual also needs input from original path (dashed line)
    dot.edge('pos_enc', 'residual', style='dashed', label='residual_connection')
    dot.edge('residual', 'final_ln')
    dot.edge('final_ln', 'output_proj')
    dot.edge('output_proj', 'softmax')
    dot.edge('softmax', 'output')
    
    return dot

def create_micro_batch_dag():
    """Create DAG showing micro-batch scheduling for pipeline parallelism"""
    
    dot = graphviz.Digraph(comment='Micro-Batch Pipeline Scheduling')
    dot.attr(rankdir='LR', size='15,10', dpi='300')
    dot.attr('node', fontname='Arial', fontsize='10')
    
    # Define time steps
    time_steps = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    
    for i, step in enumerate(time_steps):
        with dot.subgraph(name=f'cluster_{step}') as c:
            c.attr(label=f'Time Step {step}', style='rounded,filled', fillcolor='lightgray')
            
            # GPU 0 activities
            if i < 6:  # GPU 0 has work until T6
                if i < 2:  # Input processing
                    c.node(f'gpu0_input_{i}', f'GPU 0\nInput Proc\nMicro-batch {i}', 
                          shape='rectangle', fillcolor='lightgreen')
                elif i < 4:  # Output processing
                    c.node(f'gpu0_output_{i}', f'GPU 0\nOutput Proc\nMicro-batch {i-2}', 
                          shape='rectangle', fillcolor='lightgreen')
                else:  # Final cleanup
                    c.node(f'gpu0_final_{i}', f'GPU 0\nFinal\nMicro-batch {i-2}', 
                          shape='rectangle', fillcolor='lightgreen')
            
            # GPU 1-2 activities (tensor parallel expert)
            if 1 <= i <= 5:  # GPUs 1-2 have work from T1 to T5
                mb_idx = i - 1
                c.node(f'gpu1_expert_{i}', f'GPU 1\nExpert TP\nMicro-batch {mb_idx}', 
                      shape='rectangle', fillcolor='lightcyan')
                c.node(f'gpu2_expert_{i}', f'GPU 2\nExpert TP\nMicro-batch {mb_idx}', 
                      shape='rectangle', fillcolor='lightcyan')
                
                # Communication between GPUs 1-2
                c.node(f'comm_tp_{i}', f'All-Reduce\nMicro-batch {mb_idx}', 
                      shape='ellipse', fillcolor='lightblue')
                
                # Connect them
                dot.edge(f'gpu1_expert_{i}', f'comm_tp_{i}')
                dot.edge(f'gpu2_expert_{i}', f'comm_tp_{i}')
            
            # Pipeline communication
            if i < 6:
                dot.edge(f'gpu0_input_{i}', f'gpu1_expert_{i+1}', style='dashed', label='pipeline')
    
    return dot

def main():
    """Generate both DAGs and save them"""
    
    # Create main deployment DAG
    deployment_dag = create_deployment_dag()
    
    # Save DOT file
    dot_file_path = '../outputs/2025-12-03-10-02-49/hybrid_tensor_parallel_pipeline_deployment.dot'
    with open(dot_file_path, 'w') as f:
        f.write(deployment_dag.source)
    
    # Render SVG
    svg_file_path = '../outputs/2025-12-03-10-02-49/hybrid_tensor_parallel_pipeline_deployment.svg'
    deployment_dag.render(svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    # Create micro-batch scheduling DAG
    micro_batch_dag = create_micro_batch_dag()
    
    # Save DOT file for micro-batch
    micro_dot_file_path = '../outputs/2025-12-03-10-02-49/micro_batch_scheduling.dot'
    with open(micro_dot_file_path, 'w') as f:
        f.write(micro_batch_dag.source)
    
    # Render SVG for micro-batch
    micro_svg_file_path = '../outputs/2025-12-03-10-02-49/micro_batch_scheduling.svg'
    micro_batch_dag.render(micro_svg_file_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Generated DAGs:")
    print(f"1. Main deployment DAG: {dot_file_path}")
    print(f"2. Main deployment SVG: {svg_file_path}")
    print(f"3. Micro-batch scheduling DAG: {micro_dot_file_path}")
    print(f"4. Micro-batch scheduling SVG: {micro_svg_file_path}")
    
    return {
        'deployment_dot': dot_file_path,
        'deployment_svg': svg_file_path,
        'micro_batch_dot': micro_dot_file_path,
        'micro_batch_svg': micro_svg_file_path
    }

if __name__ == '__main__':
    file_paths = main()
    print("DAG generation completed successfully!")