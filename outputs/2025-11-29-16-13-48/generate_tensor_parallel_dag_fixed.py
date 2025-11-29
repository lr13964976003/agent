#!/usr/bin/env python3

import graphviz

def create_tensor_parallel_dag():
    """Create detailed DAG showing tensor parallelism within MLP layers"""
    
    dot = graphviz.Digraph(comment='Tensor Parallel MLP Layers - Detailed DAG')
    dot.attr(rankdir='TB', splines='ortho', bgcolor='white')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication
    dot.attr('node', shape='box', style='filled', fillcolor='lightgreen')     # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model parameters
    batch_size = 128
    seq_len = 10000
    hidden_size = 4096
    ffn_hidden_size = 16384
    
    # Input node
    dot.node('input', f'Input\\n(batch={batch_size}, seq={seq_len}, hidden={hidden_size})', 
             shape='box', fillcolor='lightcoral')
    
    # Layer 1 detailed tensor parallelism
    with dot.subgraph(name='cluster_layer1_detailed') as c:
        c.attr(label='Layer 1 - Tensor Parallel MLP (Detailed)', style='rounded,dashed', bgcolor='lightblue')
        
        # Input processing
        c.node('l1_input_split', 'Split Input\\nColumn-wise\\n(hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        
        # First Linear Layer - Column Parallel
        c.node('l1_linear1_gpu0', f'Linear1 Col-Parallel A\\nGPU 0\\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        c.node('l1_linear1_gpu1', f'Linear1 Col-Parallel B\\nGPU 1\\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        
        # Concatenation after first linear
        c.node('l1_concat1', 'Concatenate\\nFFN Outputs', shape='parallelogram', fillcolor='lightyellow')
        
        # GELU Activation
        c.node('l1_gelu', 'GELU Activation\\n(Element-wise)', shape='box', fillcolor='lightgreen')
        
        # Second Linear Layer - Row Parallel
        c.node('l1_split2', 'Split Intermediate\\nRow-wise\\n(ffn_hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        c.node('l1_linear2_gpu0', f'Linear2 Row-Parallel A\\nGPU 0\\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        c.node('l1_linear2_gpu1', f'Linear2 Row-Parallel B\\nGPU 1\\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        
        # All-reduce for final output
        c.node('l1_allreduce', 'All-Reduce Sum\\nFinal Output', shape='ellipse', fillcolor='lightblue')
        
        # Connections within layer 1
        c.edge('l1_input_split', 'l1_linear1_gpu0')
        c.edge('l1_input_split', 'l1_linear1_gpu1')
        c.edge('l1_linear1_gpu0', 'l1_concat1')
        c.edge('l1_linear1_gpu1', 'l1_concat1')
        c.edge('l1_concat1', 'l1_gelu')
        c.edge('l1_gelu', 'l1_split2')
        c.edge('l1_split2', 'l1_linear2_gpu0')
        c.edge('l1_split2', 'l1_linear2_gpu1')
        c.edge('l1_linear2_gpu0', 'l1_allreduce')
        c.edge('l1_linear2_gpu1', 'l1_allreduce')
    
    # Connect input to layer 1
    dot.edge('input', 'l1_input_split')
    
    # Output
    dot.node('output', f'Output\\n(batch={batch_size}, seq={seq_len}, hidden={hidden_size})', 
             shape='box', fillcolor='lightcoral')
    dot.edge('l1_allreduce', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_tensor_parallel_dag()
    
    # Save as DOT file
    dag.save('../outputs/2025-11-29-16-13-48/tensor_parallel_detailed_dag.dot')
    
    # Save as SVG image
    dag.render('../outputs/2025-11-29-16-13-48/tensor_parallel_detailed_dag', format='svg', cleanup=True)
    
    print("Tensor Parallel DAG generated successfully!")
    print(f"DOT file: ../outputs/2025-11-29-16-13-48/tensor_parallel_detailed_dag.dot")
    print(f"SVG file: ../outputs/2025-11-29-16-13-48/tensor_parallel_detailed_dag.svg")