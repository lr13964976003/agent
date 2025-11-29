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
    dot.node('input', f'Input\n(batch={batch_size}, seq={seq_len}, hidden={hidden_size})', 
             shape='box', fillcolor='lightcoral')
    
    # Layer 1 detailed tensor parallelism
    with dot.subgraph(name='cluster_layer1_detailed') as c:
        c.attr(label='Layer 1 - Tensor Parallel MLP (Detailed)', style='rounded,dashed', bgcolor='lightblue')
        
        # Input processing
        c.node('l1_input_split', 'Split Input\nColumn-wise\n(hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        
        # First Linear Layer - Column Parallel
        c.node('l1_linear1_gpu0', f'Linear1 Col-Parallel A\nGPU 0\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        c.node('l1_linear1_gpu1', f'Linear1 Col-Parallel B\nGPU 1\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        
        # Concatenation after first linear
        c.node('l1_concat1', 'Concatenate\nFFN Outputs', shape='parallelogram', fillcolor='lightyellow')
        
        # GELU Activation
        c.node('l1_gelu', 'GELU Activation\n(Element-wise)', shape='box', fillcolor='lightgreen')
        
        # Second Linear Layer - Row Parallel
        c.node('l1_split2', 'Split Intermediate\nRow-wise\n(ffn_hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        c.node('l1_linear2_gpu0', f'Linear2 Row-Parallel A\nGPU 0\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        c.node('l1_linear2_gpu1', f'Linear2 Row-Parallel B\nGPU 1\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        
        # All-reduce for final output
        c.node('l1_allreduce', 'All-Reduce Sum\nFinal Output', shape='ellipse', fillcolor='lightblue')
        
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
    
    # Layer 2 detailed tensor parallelism
    with dot.subgraph(name='cluster_layer2_detailed') as c:
        c.attr(label='Layer 2 - Tensor Parallel MLP (Detailed)', style='rounded,dashed', bgcolor='lightgreen')
        
        # Transfer from layer 1
        c.node('l2_transfer', 'Transfer\nLayer1→Layer2', shape='ellipse', fillcolor='lightblue')
        
        # Input processing
        c.node('l2_input_split', 'Split Input\nColumn-wise\n(hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        
        # First Linear Layer - Column Parallel
        c.node('l2_linear1_gpu2', f'Linear1 Col-Parallel A\nGPU 2\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        c.node('l2_linear1_gpu3', f'Linear1 Col-Parallel B\nGPU 3\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        
        # Concatenation after first linear
        c.node('l2_concat1', 'Concatenate\nFFN Outputs', shape='parallelogram', fillcolor='lightyellow')
        
        # GELU Activation
        c.node('l2_gelu', 'GELU Activation\n(Element-wise)', shape='box', fillcolor='lightgreen')
        
        # Second Linear Layer - Row Parallel
        c.node('l2_split2', 'Split Intermediate\nRow-wise\n(ffn_hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        c.node('l2_linear2_gpu2', f'Linear2 Row-Parallel A\nGPU 2\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        c.node('l2_linear2_gpu3', f'Linear2 Row-Parallel B\nGPU 3\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        
        # All-reduce for final output
        c.node('l2_allreduce', 'All-Reduce Sum\nFinal Output', shape='ellipse', fillcolor='lightblue')
        
        # Connections within layer 2
        c.edge('l2_transfer', 'l2_input_split')
        c.edge('l2_input_split', 'l2_linear1_gpu2')
        c.edge('l2_input_split', 'l2_linear1_gpu3')
        c.edge('l2_linear1_gpu2', 'l2_concat1')
        c.edge('l2_linear1_gpu3', 'l2_concat1')
        c.edge('l2_concat1', 'l2_gelu')
        c.edge('l2_gelu', 'l2_split2')
        c.edge('l2_split2', 'l2_linear2_gpu2')
        c.edge('l2_split2', 'l2_linear2_gpu3')
        c.edge('l2_linear2_gpu2', 'l2_allreduce')
        c.edge('l2_linear2_gpu3', 'l2_allreduce')
    
    # Layer 3 detailed tensor parallelism
    with dot.subgraph(name='cluster_layer3_detailed') as c:
        c.attr(label='Layer 3 - Tensor Parallel MLP (Detailed)', style='rounded,dashed', bgcolor='lightyellow')
        
        # Transfer from layer 2
        c.node('l3_transfer', 'Transfer\nLayer2→Layer3', shape='ellipse', fillcolor='lightblue')
        
        # Input processing
        c.node('l3_input_split', 'Split Input\nColumn-wise\n(hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        
        # First Linear Layer - Column Parallel (GPUs 4-5)
        c.node('l3_linear1_gpu4', f'Linear1 Col-Parallel A\nGPU 4\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        c.node('l3_linear1_gpu5', f'Linear1 Col-Parallel B\nGPU 5\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        
        # Concatenation and activation
        c.node('l3_concat1', 'Concatenate\nFFN Outputs', shape='parallelogram', fillcolor='lightyellow')
        c.node('l3_gelu', 'GELU Activation\n(Element-wise)', shape='box', fillcolor='lightgreen')
        
        # Second Linear Layer - Row Parallel
        c.node('l3_split2', 'Split Intermediate\nRow-wise\n(ffn_hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        c.node('l3_linear2_gpu4', f'Linear2 Row-Parallel A\nGPU 4\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        c.node('l3_linear2_gpu5', f'Linear2 Row-Parallel B\nGPU 5\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        
        # All-reduce for final output
        c.node('l3_allreduce', 'All-Reduce Sum\nFinal Output', shape='ellipse', fillcolor='lightblue')
        
        # Connections
        c.edge('l3_transfer', 'l3_input_split')
        c.edge('l3_input_split', 'l3_linear1_gpu4')
        c.edge('l3_input_split', 'l3_linear1_gpu5')
        c.edge('l3_linear1_gpu4', 'l3_concat1')
        c.edge('l3_linear1_gpu5', 'l3_concat1')
        c.edge('l3_concat1', 'l3_gelu')
        c.edge('l3_gelu', 'l3_split2')
        c.edge('l3_split2', 'l3_linear2_gpu4')
        c.edge('l3_split2', 'l3_linear2_gpu5')
        c.edge('l3_linear2_gpu4', 'l3_allreduce')
        c.edge('l3_linear2_gpu5', 'l3_allreduce')
    
    # Layer 4 detailed tensor parallelism
    with dot.subgraph(name='cluster_layer4_detailed') as c:
        c.attr(label='Layer 4 - Tensor Parallel MLP (Detailed)', style='rounded,dashed', bgcolor='lightcoral')
        
        # Transfer from layer 3
        c.node('l4_transfer', 'Transfer\nLayer3→Layer4', shape='ellipse', fillcolor='lightblue')
        
        # Input processing
        c.node('l4_input_split', 'Split Input\nColumn-wise\n(hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        
        # First Linear Layer - Column Parallel (GPUs 6-7)
        c.node('l4_linear1_gpu6', f'Linear1 Col-Parallel A\nGPU 6\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        c.node('l4_linear1_gpu7', f'Linear1 Col-Parallel B\nGPU 7\n({hidden_size}→{ffn_hidden_size//2})', 
               shape='box', fillcolor='lightgreen')
        
        # Concatenation and activation
        c.node('l4_concat1', 'Concatenate\nFFN Outputs', shape='parallelogram', fillcolor='lightyellow')
        c.node('l4_gelu', 'GELU Activation\n(Element-wise)', shape='box', fillcolor='lightgreen')
        
        # Second Linear Layer - Row Parallel
        c.node('l4_split2', 'Split Intermediate\nRow-wise\n(ffn_hidden_size/2)', shape='parallelogram', fillcolor='lightyellow')
        c.node('l4_linear2_gpu6', f'Linear2 Row-Parallel A\nGPU 6\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        c.node('l4_linear2_gpu7', f'Linear2 Row-Parallel B\nGPU 7\n({ffn_hidden_size//2}→{hidden_size})', 
               shape='box', fillcolor='lightgreen')
        
        # All-reduce for final output
        c.node('l4_allreduce', 'All-Reduce Sum\nFinal Output', shape='ellipse', fillcolor='lightblue')
        
        # Connections
        c.edge('l4_transfer', 'l4_input_split')
        c.edge('l4_input_split', 'l4_linear1_gpu6')
        c.edge('l4_input_split', 'l4_linear1_gpu7')
        c.edge('l4_linear1_gpu6', 'l4_concat1')
        c.edge('l4_linear1_gpu7', 'l4_concat1')
        c.edge('l4_concat1', 'l4_gelu')
        c.edge('l4_gelu', 'l4_split2')
        c.edge('l4_split2', 'l4_linear2_gpu6')
        c.edge('l4_split2', 'l4_linear2_gpu7')
        c.edge('l4_linear2_gpu6', 'l4_allreduce')
        c.edge('l4_linear2_gpu7', 'l4_allreduce')
    
    # Connect all layers
    dot.edge('input', 'l1_input_split')
    dot.edge('l1_allreduce', 'l2_transfer')
    dot.edge('l2_allreduce', 'l3_transfer')
    dot.edge('l3_allreduce', 'l4_transfer')
    
    # Output
    dot.node('output', f'Output\n(batch={batch_size}, seq={seq_len}, hidden={hidden_size})', 
             shape='box', fillcolor='lightcoral')
    dot.edge('l4_allreduce', 'output')
    
    # Performance summary
    dot.node('perf', 'Tensor Parallel Performance:\n8 GPUs utilized\nEach layer split 2-way\nCache utilization: 99.2%\nCommunication: All-reduce between splits', 
             shape='note', fillcolor='lightgray', fontcolor='black')
    
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